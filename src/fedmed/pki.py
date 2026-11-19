"""A small private PKI for the federation.

Creates a federation root CA, a server certificate for the coordinator, one client
certificate per hospital (CN = site id, EKU = clientAuth), a CRL, and an Ed25519 key
for signing access tokens. In production the CA key would live in an HSM or an offline
machine. Here it is a file with 0600 permissions, which is fine for a proof of concept.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ORG = "fedmed federation (proof of concept)"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _write_key(path: Path, key) -> None:
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)


def _write_cert(path: Path, cert: x509.Certificate) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(Path(path).read_bytes())


def load_key(path: Path):
    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)


def create_ca(out: Path, days: int = 3650) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP384R1())
    name = x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORG),
                      x509.NameAttribute(NameOID.COMMON_NAME, "fedmed Federation Root CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - dt.timedelta(minutes=5)).not_valid_after(_now() + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                     crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA384())
    )
    _write_key(out / "ca.key", key)
    _write_cert(out / "ca.crt", cert)
    return cert, key


def issue_cert(
    out: Path, name: str, ca_cert: x509.Certificate, ca_key, server: bool = False,
    sans: list[str] | None = None, days: int = 365,
) -> x509.Certificate:
    key = ec.generate_private_key(ec.SECP256R1())
    b = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORG),
                                 x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(ca_cert.subject).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - dt.timedelta(minutes=5)).not_valid_after(_now() + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=True, key_cert_sign=False,
                                     crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage(
            [ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
    )
    if server:
        alt: list[x509.GeneralName] = []
        for s in sans or ["localhost"]:
            try:
                alt.append(x509.IPAddress(ipaddress.ip_address(s)))
            except ValueError:
                alt.append(x509.DNSName(s))
        b = b.add_extension(x509.SubjectAlternativeName(alt), critical=False)
    cert = b.sign(ca_key, hashes.SHA384())
    _write_key(out / f"{name}.key", key)
    _write_cert(out / f"{name}.crt", cert)
    return cert


def write_crl(out: Path, ca_cert, ca_key, revoked_serials: list[int]) -> None:
    b = (x509.CertificateRevocationListBuilder().issuer_name(ca_cert.subject)
         .last_update(_now()).next_update(_now() + dt.timedelta(days=30)))
    for s in revoked_serials:
        b = b.add_revoked_certificate(
            x509.RevokedCertificateBuilder().serial_number(s).revocation_date(_now()).build())
    (out / "crl.pem").write_bytes(b.sign(ca_key, hashes.SHA384()).public_bytes(serialization.Encoding.PEM))


def revoked_serials(pki: Path) -> list[int]:
    p = Path(pki) / "crl.pem"
    if not p.exists():
        return []
    return [r.serial_number for r in x509.load_pem_x509_crl(p.read_bytes())]


def revoke(pki: Path, cert_path: Path) -> int:
    pki = Path(pki)
    ca_cert, ca_key = load_cert(pki / "ca.crt"), load_key(pki / "ca.key")
    serial = load_cert(cert_path).serial_number
    write_crl(pki, ca_cert, ca_key, sorted(set(revoked_serials(pki)) | {serial}))
    return serial


def init_pki(out: Path, sites: list[str], server_sans: list[str], extra_clients: tuple[str, ...] = ()) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ca_cert, ca_key = create_ca(out)
    issue_cert(out, "server", ca_cert, ca_key, server=True, sans=server_sans)
    for name in [*sites, *extra_clients]:
        issue_cert(out, name, ca_cert, ca_key)
    write_crl(out, ca_cert, ca_key, [])
    tok = ed25519.Ed25519PrivateKey.generate()
    _write_key(out / "token_signing.key", tok)
    (out / "token_signing.pub").write_bytes(
        tok.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
