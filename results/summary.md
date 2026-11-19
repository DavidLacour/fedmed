## Aggregation (balanced accuracy, mean of last 10 rounds, mean ± sd over seeds)

| Method | α = 0.1 (strongly non-IID) | α = 0.5 (moderately non-IID) | α = 100 (≈ IID) |
|---|---|---|---|
| FedAvg | 0.355 ± 0.016 | 0.494 ± 0.006 | 0.530 ± 0.006 |
| FedProx μ=0.01 | 0.354 ± 0.019 | 0.497 ± 0.007 | 0.528 ± 0.008 |
| FedProx μ=0.1 | 0.336 ± 0.020 | 0.491 ± 0.004 | 0.525 ± 0.011 |
| SCAFFOLD | 0.384 ± 0.008 | 0.501 ± 0.022 | 0.526 ± 0.009 |
| Pooled data (no federation) | 0.560 | | |
| Each hospital alone, α=0.1 (mean of 5) | 0.309 | | |
| Each hospital alone, α=0.5 (mean of 5) | 0.387 | | |

## Differential privacy

| setting | noise σ | ε (worst site) | balanced acc | accuracy |
|---|---|---|---|---|
| C=10, B=128, 50 rounds | 3 | 9.6 | 0.182 | 0.600 |
| C=10, B=128, 50 rounds | 2 | 16.6 | 0.168 | 0.640 |
| C=10, B=128, 50 rounds | 1.5 | 25.5 | 0.194 | 0.589 |
| C=10, B=128, 50 rounds | 1 | 51.4 | 0.157 | 0.656 |
| C=10, B=128, 50 rounds | 0.7 | 108.3 | 0.178 | 0.663 |
| C=10, B=128, 50 rounds | 0.5 | 240.1 | 0.207 | 0.646 |
| C=1, B=32, 100 rounds | 3 | 5.6 | 0.145 | 0.473 |
| C=1, B=32, 100 rounds | 2 | 9.5 | 0.164 | 0.544 |
| C=1, B=32, 100 rounds | 1.5 | 14.5 | 0.167 | 0.601 |
| C=1, B=32, 100 rounds | 1 | 29.6 | 0.154 | 0.656 |
| C=1, B=32, 100 rounds | 0.8 | 48.5 | 0.144 | 0.664 |
| C=1, B=32, 100 rounds | 0.6 | 104.7 | 0.143 | 0.668 |
| C=10 settings, no DP | – | ∞ | 0.428 | 0.671 |

## Communication

| Uplink encoding | MB / site / round | total uplink (100 rounds) | balanced acc | rounds to 0.30 bal. acc | energy (kWh, CodeCarbon est.) |
|---|---|---|---|---|---|
| float32 (baseline) | 0.379 | 189 MB | 0.488 | 11 | 0.0056 |
| fp16 | 0.189 | 95 MB | 0.489 | 11 | 0.0054 |
| int8 blockwise | 0.096 | 48 MB | 0.488 | 11 | 0.0055 |
| top-10% + error feedback | 0.076 | 38 MB | 0.490 | 11 | 0.0053 |
| top-1% + error feedback | 0.008 | 4 MB | 0.479 | 31 | 0.0053 |
| secure aggregation (masked) | 0.757 | 378 MB | 0.487 | 11 | 0.0048 |
| CKKS homomorphic encryption (benchmark only) | 7.95 | – | – | – | – |

## Scalability (10 rounds, α = 0.5)

| sites | client time / round, plain (s) | client time / round, secagg (s) | uplink / round, plain (MB) | uplink / round, secagg (MB) |
|---|---|---|---|---|
| 3 | 2.46 | 2.46 | 1.1 | 2.3 |
| 5 | 1.81 | 1.87 | 1.9 | 3.8 |
| 10 | 1.76 | 1.81 | 3.8 | 7.6 |
| 20 | 1.42 | 1.21 | 7.6 | 15.1 |
