# RTX 3080: token-weighted generation throughput

Source: `C:\Users\Harman\Documents\Python\llama.cpp benchmarks\results\nvidia-rtx-3080\mtp-speed-bench-results`

Each cell is weighted mean ± one weighted population standard deviation, in tokens/s.
Weights are `completion_tokens`; speed is `predicted_per_second` (server generation speed).
`n_predict` below refers to the filename's MTP `n` / `runs.csv`'s `mtp_n_max` (0–5).
OSL is the configured output sequence length.

Weighted mean: `μ = Σ(tokens_i × speed_i) / Σ(tokens_i)`.
Weighted population SD: `σ = sqrt(Σ(tokens_i × (speed_i − μ)²) / Σ(tokens_i))`.
This SD describes the supplied runs; it is not a confidence interval or standard error.

Configurations: 72. Included runs: 2160.

## throughput_1k — high entropy

| n_predict | OSL 1024 | OSL 2048 | OSL 4096 |
|---:|---:|---:|---:|
| 0 | 129.92 ± 0.69 | 128.27 ± 0.27 | 128.28 ± 0.34 |
| 1 | 161.41 ± 3.82 | 161.82 ± 4.30 | 167.89 ± 3.89 |
| 2 | 175.09 ± 9.26 | 181.99 ± 8.84 | 196.33 ± 10.31 |
| 3 | 172.07 ± 12.80 | 183.48 ± 14.78 | 207.12 ± 15.67 |
| 4 | 162.94 ± 18.52 | 177.12 ± 20.63 | 204.48 ± 23.33 |
| 5 | 152.41 ± 16.67 | 167.64 ± 25.07 | 204.81 ± 27.39 |

## throughput_1k — low entropy

| n_predict | OSL 1024 | OSL 2048 | OSL 4096 |
|---:|---:|---:|---:|
| 0 | 129.28 ± 0.79 | 127.91 ± 0.73 | 127.85 ± 0.57 |
| 1 | 165.87 ± 5.65 | 165.03 ± 4.25 | 168.13 ± 4.27 |
| 2 | 188.00 ± 15.31 | 189.33 ± 10.41 | 199.04 ± 10.16 |
| 3 | 190.46 ± 24.59 | 194.35 ± 19.07 | 211.52 ± 18.75 |
| 4 | 184.65 ± 26.73 | 195.71 ± 21.44 | 217.24 ± 22.16 |
| 5 | 179.22 ± 31.32 | 189.11 ± 24.37 | 216.62 ± 24.74 |

## throughput_2k — high entropy

| n_predict | OSL 1024 | OSL 2048 | OSL 4096 |
|---:|---:|---:|---:|
| 0 | 129.20 ± 0.21 | 128.73 ± 0.14 | 127.93 ± 0.17 |
| 1 | 160.83 ± 3.96 | 162.53 ± 4.88 | 166.28 ± 3.53 |
| 2 | 179.17 ± 7.62 | 185.67 ± 9.63 | 197.80 ± 8.96 |
| 3 | 177.63 ± 12.65 | 186.10 ± 14.09 | 206.22 ± 13.89 |
| 4 | 166.87 ± 15.21 | 176.17 ± 15.19 | 202.14 ± 18.08 |
| 5 | 158.14 ± 17.00 | 172.14 ± 24.65 | 205.30 ± 24.28 |

## throughput_2k — low entropy

| n_predict | OSL 1024 | OSL 2048 | OSL 4096 |
|---:|---:|---:|---:|
| 0 | 128.84 ± 0.98 | 127.61 ± 0.53 | 127.29 ± 0.31 |
| 1 | 163.94 ± 4.75 | 164.65 ± 3.49 | 168.29 ± 3.48 |
| 2 | 187.44 ± 13.10 | 189.70 ± 10.61 | 198.74 ± 9.23 |
| 3 | 190.28 ± 18.28 | 195.33 ± 16.36 | 209.49 ± 16.09 |
| 4 | 187.66 ± 24.22 | 194.54 ± 17.24 | 216.59 ± 19.57 |
| 5 | 183.43 ± 26.80 | 190.69 ± 19.68 | 218.75 ± 23.59 |
