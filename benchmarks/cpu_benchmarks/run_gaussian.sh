#!/bin/bash
# Main CPU experiment of the paper: Gaussian logits, batch 32, K = 10 ... 10^6, six values of sigma (about 7 minutes).
set -eo pipefail; cd "$(dirname "$0")"; mkdir -p results
python bench_cpu_e2e.py --N 32 --kmin 10 --kmax 1e6 --stds 0.001,0.01,0.1,1,10,100 --dist gaussian --out results/gaussian.json | tee results/gaussian.log
python plot_cpu_sigma.py results/gaussian.json results/speed_sigma_all_cpu | tee results/gaussian_summary.txt
python plot_cpu_sigma_row.py results/gaussian.json results/speed_sigma_row_cpu
