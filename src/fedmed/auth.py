"""Authentication and authorisation on top of mTLS.

Three independent checks guard every API call:

1. **mTLS**: the TLS handshake only succeeds with a client certificate issued by the
   federation CA and absent from the CRL. The certificate CN is the site identity.
2. **Certificate-bound access token** (RFC 8705 style): an EdDSA-signed JWT whose
   ``cnf.x5t#S256`` claim must equal the SHA-256 thumbprint of the certificate presented
   on *this* connection, and whose ``sub`` must equal the certificate CN. A stolen
   token is useless without the matching private key, and a stolen key is useless
   once its token expires or is revoked.
3. **Role-based authorisation**: a policy file maps each site to one role and each role
   to permissions. The policy, not the token, is the source of truth, so permissions can
   be withdrawn centrally without re-issuing certificates.
"""

from __future__ import annotations

import base64
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import jwt
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

ISSUER = "fedmed-coordinator"
AUDIENCE = "fedmed-api"

DEFAULT_POLICY = {
    "roles": {
        "trainer": ["task:read", "model:read", "update:submit", "metrics:submit", "quality:submit",
                    "secagg:register", "secagg:read"],
        "observer": ["status:read"],
        "admin": ["status:read", "task:read"],
    },
    "sites": {},
    "revoked_tokens": [],
}


class AuthError(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def cert_thumbprint(cert_der: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(cert_der).digest()).rstrip(b"=").decode()


def cert_cn(cert_der: bytes) -> str:
    cert = x509.load_der_x509_certificate(cert_der)
    return cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value


def issue_token(signing_key_path: Path, site: str, role: str, cert_path: Path, ttl_hours: float = 72) -> str:
    key = serialization.load_pem_private_key(Path(signing_key_path).read_bytes(), password=None)
    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    der = cert.public_bytes(serialization.Encoding.DER)
    if cert_cn(der) != site:
        raise ValueError("certificate CN does not match site")
    now = int(time.time())
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": site, "role": role,
        "iat": now, "nbf": now - 60, "exp": now + int(ttl_hours * 3600),
        "jti": str(uuid.uuid4()),
        "cnf": {"x5t#S256": cert_thumbprint(der)},
    }
    return jwt.encode(claims, key, algorithm="EdDSA")


@dataclass
class Policy:
    roles: dict[str, list[str]]
    sites: dict[str, str]
    revoked_tokens: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path | None, default_sites: dict[str, str] | None = None) -> Policy:
        raw = dict(DEFAULT_POLICY)
        if path:
            raw.update(yaml.safe_load(Path(path).read_text()) or {})
        sites = {**(default_sites or {}), **(raw.get("sites") or {})}
        return cls(raw["roles"], sites, set(raw.get("revoked_tokens") or []))

    def permissions(self, site: str) -> set[str]:
        role = self.sites.get(site)
        return set(self.roles.get(role, [])) if role else set()


@dataclass
class Principal:
    site: str
    role: str
    permissions: set[str]
    token_id: str


class Authenticator:
    def __init__(self, token_public_key_path: Path, policy: Policy, leeway_s: int = 30):
        self._pub = serialization.load_pem_public_key(Path(token_public_key_path).read_bytes())
        self.policy = policy
        self.leeway = leeway_s

    def authenticate(self, cert_der: bytes | None, authorization: str | None) -> Principal:
        if not cert_der:  # unreachable when the TLS context requires client certs; defence in depth
            raise AuthError(401, "no client certificate")
        site = cert_cn(cert_der)
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthError(401, "missing bearer token")
        try:
            claims = jwt.decode(
                authorization[7:], self._pub, algorithms=["EdDSA"], audience=AUDIENCE, issuer=ISSUER,
                leeway=self.leeway, options={"require": ["exp", "iat", "sub", "jti", "cnf"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError(401, f"invalid token: {type(exc).__name__}") from exc
        if claims["sub"] != site:
            raise AuthError(401, "token subject does not match certificate")
        if claims.get("cnf", {}).get("x5t#S256") != cert_thumbprint(cert_der):
            raise AuthError(401, "token not bound to this certificate")
        if claims["jti"] in self.policy.revoked_tokens:
            raise AuthError(401, "token revoked")
        role = self.policy.sites.get(site)
        if role is None:
            raise AuthError(403, "site not in policy")
        if claims.get("role") != role:
            raise AuthError(403, "token role does not match policy")
        return Principal(site, role, self.policy.permissions(site), claims["jti"])

    @staticmethod
    def authorise(principal: Principal, permission: str) -> None:
        if permission not in principal.permissions:
            raise AuthError(403, f"{principal.site} ({principal.role}) lacks {permission}")
