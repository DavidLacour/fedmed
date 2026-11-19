# Threat model

This is a proof of concept. The document says what the design protects against, what it
does not, and which test shows each claim. Each "Tested by" entry points to a test that
passes in CI.

## Assets

| Asset | Where it lives | Leaves the hospital? |
|---|---|---|
| Images and labels (patient data) | hospital node disk, mounted read-only | **Never** |
| File names, per-file quality findings | `local_quality_report.json` on the node | **Never** |
| Aggregate data-quality summary | sent to the coordinator | Yes: counts only, small cells suppressed (`<5`), optional Laplace noise |
| Model update (Δw) for one hospital | node memory | Yes. With secure aggregation it leaves only **masked**. With DP it is a DP-SGD output. |
| Global model | coordinator | Sent to the participants |
| Private keys (TLS, token signing, CA) | `certs/` with mode 0600, or K8s Secrets with mode 0440 | Never sent over the network |

## Adversaries and what each can learn

### 1. Honest-but-curious coordinator (the main adversary)
The coordinator follows the protocol but tries to learn about individual patients from
what it receives.

| Without protection | With fedmed's defences |
|---|---|
| Sees each hospital's update Δwᵢ. Gradient-inversion attacks can reconstruct training images from a single client's update, especially for small batches. Membership inference on Δwᵢ can show whether a specific patient was in a hospital's data. | **Secure aggregation** (`secagg.py`): the coordinator receives only Σᵢ nᵢΔwᵢ and Σᵢ nᵢ. A single masked update is uniform noise over ℤ₂⁶⁴. **DP-SGD** (`train.py`, Opacus): every hospital clips per-sample gradients and adds Gaussian noise, so even the *sum*, and the final model, satisfy (ε, δ)-DP at the record level, with the ε reported per site. |
| Sees each site's sample count nᵢ. | Hidden inside the secure sum. |
| Sees label histograms if they are sent in the clear. | Only the suppressed, optionally noised histogram from the quality tool. |

Tested by: `test_masks_cancel_in_the_sum`, `test_single_masked_update_reveals_nothing_useful`,
`test_masks_change_every_round`, `test_dp_budget_accumulates_across_rounds`,
`test_shared_quality_report_leaks_no_filenames_and_suppresses_small_cells`.

**What it can still learn:** the aggregate update each round (this is the point of FL),
the global model, timing and message sizes, which sites participate, and the suppressed
histograms. Without DP, the aggregate of 5 sites still leaks information. Secure
aggregation alone is *not* a privacy guarantee, which is why DP is layered on top.

### 2. Malicious coordinator (active)
| Attack | Defence | Status |
|---|---|---|
| Swap a hospital's secure-aggregation public key for its own key (man in the middle), then unmask that hospital's update | Each X25519 key is **signed with the hospital's mTLS certificate key**. Every node verifies all signatures and the certificate chain to the federation CA, and checks that CN = site, before masking. | Defended. `test_coordinator_cannot_substitute_keys`, `test_roster_rejects_cert_for_other_site` |
| Run "rounds" with a roster that contains only one real hospital, so that the sum equals that hospital's update | Not prevented. The full Bonawitz protocol has a minimum-threshold check, and a node could refuse rosters smaller than *t*. | **Open**. Mitigated by DP. |
| Send a crafted global model to a single hospital (model-inversion or "trap" weights) | Not prevented. Would need verifiable broadcast of the global model. | **Open**. Mitigated by DP. |

### 3. Malicious or compromised hospital
| Attack | Defence | Status |
|---|---|---|
| Impersonate another hospital | mTLS identity comes from the certificate CN; the token must name the same subject and be bound to that certificate's thumbprint. | Defended. `test_stolen_token_is_useless_with_another_certificate` |
| Submit NaN or huge updates (bug or poisoning) | Shape, finiteness and L2-norm bound checks; wrong-round and duplicate submissions rejected | Defended for plain updates. `test_update_validation` |
| Model poisoning *within* the norm bound, or any poisoning when secure aggregation is on | Masked updates cannot be inspected: robust aggregation and secure aggregation are fundamentally in tension. Fixes include secure range proofs, or robust aggregation inside an MPC/TEE. | **Open** (documented trade-off) |
| Upload pickled code to exploit the server | Wire format is JSON plus raw arrays with a dtype allowlist and bounds checks. Nothing is ever unpickled. | Defended. `test_pack_refuses_object_arrays_and_unpack_rejects_tampering`, `test_update_validation` |
| Path traversal through `labels.csv` | File names are checked against their basename before opening | Defended. `test_quality_tool_finds_planted_defects` |

### 4. Network attacker
| Attack | Defence | Tested by |
|---|---|---|
| Eavesdrop or modify traffic | TLS 1.3 only, both directions | all security tests run over real TLS |
| Connect without a certificate | Handshake fails (`CERT_REQUIRED`) | `test_no_client_certificate_fails_handshake` |
| Certificate from another CA | Handshake fails | `test_certificate_from_rogue_ca_is_rejected` |
| Impersonate the coordinator | Nodes pin the federation CA and verify the hostname | `test_client_refuses_impostor_server` |
| Use a stolen but revoked certificate | CRL checked at handshake (`VERIFY_CRL_CHECK_LEAF`) | `test_revoked_certificate_is_rejected` |
| Replay an old token | `exp`/`nbf` checked, plus a `jti` denylist in the policy | `test_expired_token_is_rejected` |
| Forge a token, e.g. by switching it to HS256 (algorithm confusion) | Only `EdDSA` accepted, verified with the coordinator's public key | `test_forged_token_is_rejected` |
| Pivot from one hospital to another | Docker: one bridge network per hospital. K8s: NetworkPolicy allows hospitals to reach only the coordinator. | `run_federation.sh` isolation check |

### 5. Insider at the coordinator with read access to its disk
Finds `server.key`, the token **verification** key, the CRL, run history and the audit
log. It does **not** find the CA key or the token-signing key: the coordinator never
receives either one (see the compose and Helm mounts), so it cannot mint identities or tokens.

## Explicitly out of scope for this PoC
* **Dropout tolerance in secure aggregation.** If a hospital disappears after masking, the round
  cannot be unmasked (`test_missing_site_breaks_unmasking`). The full protocol fixes this
  with Shamir secret sharing of the seeds.
* **Threshold or multi-key HE.** The CKKS demo uses one key pair, so it shows cost, not a
  deployable trust model.
* **Opacus `secure_mode`** (cryptographically secure noise RNG) is off for speed. Production
  training should enable it.
* **HSM-backed CA, automated certificate rotation (ACME/step-ca), OIDC for humans, and
  SBOM/image signing.** These are listed in the README as next steps.
* **Clinical validity.** DermaMNIST is a 28×28 downsample of HAM10000. The model is a
  research toy, not a diagnostic device.
