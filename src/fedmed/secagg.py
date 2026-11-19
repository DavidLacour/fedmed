"""Secure aggregation with pairwise additive masks (simplified Bonawitz et al., CCS 2017).

Every pair of sites (i, j) agrees on a shared secret with X25519. Both expand it with
ChaCha20 into the same pseudo-random mask m_ij. Site i adds m_ij for every j > i and
subtracts it for every j < i. In the sum over all sites every mask cancels, so the
coordinator learns sum_i(n_i * delta_i) and sum_i(n_i), and nothing about any single
site's update.

Hardening beyond the textbook version:
* Each site signs its X25519 public key with the private key of its mTLS client
  certificate. Peers verify the signature and the certificate chain against the
  federation CA, so a malicious coordinator cannot swap in its own keys (MITM) to
  unmask an individual update.
* Masks are bound to (run id, round, ordered pair), so they are never reused.

Limitations (documented, not hidden): no dropout recovery. If a site disappears after
masking, the round must be restarted; the full protocol uses Shamir secret sharing of
the seeds for this. Arithmetic is fixed point over Z_{2^64}.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.x509.oid import NameOID

FRAC_BITS = 24  # fixed-point precision: 2^-24 ~ 6e-8


def encode_fixed(vec: np.ndarray) -> np.ndarray:
    q = np.round(vec.astype(np.float64) * (1 << FRAC_BITS)).astype(np.int64)
    return q.view(np.uint64)


def decode_fixed(vec: np.ndarray) -> np.ndarray:
    return (vec.view(np.int64).astype(np.float64) / (1 << FRAC_BITS)).astype(np.float32)


def prg(seed: bytes, n: int) -> np.ndarray:
    """ChaCha20 keystream as n uint64 values. Seed is a fresh 32-byte HKDF output."""
    enc = Cipher(algorithms.ChaCha20(seed, b"\x00" * 16), mode=None).encryptor()
    return np.frombuffer(enc.update(b"\x00" * (8 * n)), dtype="<u8").astype(np.uint64)


@dataclass
class RosterEntry:
    site: str
    public_key: bytes  # raw X25519
    cert_pem: bytes = b""
    signature: bytes = b""


class SecAggClient:
    def __init__(self, site: str, signing_key: ec.EllipticCurvePrivateKey | None = None):
        self.site = site
        self._sk = x25519.X25519PrivateKey.generate()
        self.public_key = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.signature = b""
        if signing_key is not None:
            self.signature = signing_key.sign(self._signed_blob(site, self.public_key), ec.ECDSA(hashes.SHA256()))

    @staticmethod
    def _signed_blob(site: str, pk: bytes) -> bytes:
        return b"fedmed-secagg-v1|" + site.encode() + b"|" + pk

    def mask(self, vec: np.ndarray, roster: list[RosterEntry], run_id: str, rnd: int) -> np.ndarray:
        """Return fixed-point vec + sum of pairwise masks (mod 2^64)."""
        out = encode_fixed(vec)
        sites = sorted(e.site for e in roster)
        if self.site not in sites:
            raise ValueError("own site missing from roster")
        for e in roster:
            if e.site == self.site:
                continue
            shared = self._sk.exchange(x25519.X25519PublicKey.from_public_bytes(e.public_key))
            a, b = sorted((self.site, e.site))
            info = f"fedmed-mask|{run_id}|{rnd}|{a}|{b}".encode()
            seed = HKDF(hashes.SHA256(), 32, salt=None, info=info).derive(shared)
            m = prg(seed, len(out))
            with np.errstate(over="ignore"):
                out = out + m if self.site < e.site else out - m
        return out


def unmask_sum(masked: list[np.ndarray]) -> np.ndarray:
    """Coordinator side: modular sum of masked vectors -> float sum of plain vectors."""
    total = np.zeros_like(masked[0])
    with np.errstate(over="ignore"):
        for m in masked:
            total = total + m
    return decode_fixed(total)


def verify_roster(roster: list[RosterEntry], ca_cert_pem: bytes) -> None:
    """Raise if any roster key is not signed by a CA-issued cert whose CN is that site."""
    ca = x509.load_pem_x509_certificate(ca_cert_pem)
    seen = set()
    for e in roster:
        if e.site in seen:
            raise ValueError(f"duplicate roster entry for {e.site}")
        seen.add(e.site)
        cert = x509.load_pem_x509_certificate(e.cert_pem)
        cert.verify_directly_issued_by(ca)
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        if cn != e.site:
            raise ValueError(f"roster cert CN {cn!r} does not match site {e.site!r}")
        pub = cert.public_key()
        if not isinstance(pub, ec.EllipticCurvePublicKey):
            raise ValueError("expected EC client certificate")
        pub.verify(e.signature, SecAggClient._signed_blob(e.site, e.public_key), ec.ECDSA(hashes.SHA256()))


def fingerprint(pk: bytes) -> str:
    return hashlib.sha256(pk).hexdigest()[:16]
