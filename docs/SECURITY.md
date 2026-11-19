# Security setup: PKI, mTLS, tokens and authorisation

## 1. The federation PKI (`fedmed pki init`)

```
certs/
├── ca.crt / ca.key              federation root CA, ECDSA P-384, CA:TRUE pathlen:0
├── server.crt / server.key      coordinator, EKU=serverAuth, SAN=localhost,127.0.0.1,coordinator
├── hospital-{a..e}.crt / .key   one per site, CN=<site id>, EKU=clientAuth, ECDSA P-256
├── auditor.crt / .key           read-only observer identity
├── crl.pem                      certificate revocation list, signed by the CA
├── token_signing.key / .pub     Ed25519 key pair for access tokens
└── tokens/<site>.jwt            issued by `fedmed token`
```

* Private keys are written with mode `0600` and never sent over the network.
* **Who gets what.** The coordinator receives `server.*`, `ca.crt`, `crl.pem` and
  `token_signing.pub` only. Each hospital receives `ca.crt`, *its own* key, certificate
  and token. `ca.key` and `token_signing.key` stay with the federation operator and,
  in a real deployment, would go offline or into an HSM.
* Inspect a certificate: `openssl x509 -in certs/hospital-a.crt -noout -text`.

Equivalent OpenSSL commands, if you prefer to create the certificates by hand:

```bash
openssl ecparam -name secp384r1 -genkey -noout -out ca.key
openssl req -x509 -new -key ca.key -sha384 -days 3650 -subj "/O=fedmed/CN=fedmed Federation Root CA" \
  -addext "basicConstraints=critical,CA:TRUE,pathlen:0" -addext "keyUsage=critical,keyCertSign,cRLSign" -out ca.crt
openssl ecparam -name prime256v1 -genkey -noout -out hospital-a.key
openssl req -new -key hospital-a.key -subj "/O=fedmed/CN=hospital-a" -out hospital-a.csr
openssl x509 -req -in hospital-a.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 365 -sha384 \
  -extfile <(printf "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyAgreement\nextendedKeyUsage=clientAuth") \
  -out hospital-a.crt
```

## 2. Transport: mutual TLS 1.3

Coordinator (`server/app.py::server_ssl_context`):
* `minimum_version = TLSv1_3`
* `verify_mode = CERT_REQUIRED`, trusting only `ca.crt`
* `VERIFY_CRL_CHECK_LEAF` against `crl.pem`

Node (`node/agent.py::client_ssl_context`): trusts only `ca.crt`, checks the hostname
(`check_hostname=True`) and presents its own certificate.

Revoke a hospital: `fedmed pki revoke certs/hospital-c.crt`, then restart the
coordinator. The next handshake from that certificate fails.

## 3. Authentication: certificate-bound access tokens

`fedmed token --site hospital-a --role trainer` issues an EdDSA-signed JWT:

```json
{ "iss": "fedmed-coordinator", "aud": "fedmed-api", "sub": "hospital-a", "role": "trainer",
  "iat": 1795000000, "nbf": 1794999940, "exp": 1795259200, "jti": "8d5f…",
  "cnf": { "x5t#S256": "<base64url SHA-256 of hospital-a.crt DER>" } }
```

On every request the coordinator checks:
1. the signature (EdDSA only, which blocks `alg=none` and HS256 confusion), plus `aud`, `iss`, `exp` and `nbf`;
2. `sub` equals the CN of the certificate on this TLS connection;
3. `cnf.x5t#S256` equals the thumbprint of that certificate (RFC 8705 binding);
4. `jti` is not in the policy's `revoked_tokens`.

So a leaked token cannot be used without the matching private key, and a leaked key
cannot be used once its token expires (72 h by default) or is revoked.

## 4. Authorisation: role-based access control

`deploy/config/policy.yaml` is the single source of truth:

| Role | Permissions |
|---|---|
| `trainer` (hospitals) | `task:read model:read update:submit metrics:submit quality:submit secagg:register secagg:read` |
| `observer` (auditor) | `status:read` |

Every route declares the one permission it needs (`ROUTES` in `server/app.py`), and
unknown routes return 404 before authentication runs. The token's role must equal
the policy's role for that site, so permissions can be withdrawn centrally
without re-issuing certificates. Write endpoints also check that the payload's site equals
the authenticated site and that the site is a participant in the current run.

## 5. Audit log

Every request, allowed or denied, is appended to `runs/coordinator/audit.jsonl` with
`ts`, `site`, `method`, `path`, `status` and `reason`. The log holds no payloads.

## 6. Secure-software practices used in the code

* No pickle and no `torch.load` on network input; the wire format has a dtype allowlist and bounds checks.
* Request size limit of 64 MB; malformed input returns 400 without echoing internals.
* Containers run as non-root with a read-only root filesystem, all capabilities dropped and
  `no-new-privileges`. Data and keys are mounted read-only.
* K8s: `runAsNonRoot`, `seccompProfile: RuntimeDefault`, `automountServiceAccountToken: false`,
  NetworkPolicies.
* CI runs `ruff` with the `bandit` (S) rule set, and `pip-audit` for dependency CVEs.
* **Dependency confusion found during development.** Flower's runtime dependency installer
  pulled an unrelated PyPI project named `fedmed` in place of the local package. The
  distribution was renamed to `fedmed-imaging`, and the Flower app no longer lists it as a
  dependency (see `flower_app/pyproject.toml`).
