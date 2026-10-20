#!/bin/bash
# Other logit distributions (about 4 minutes): Laplace, and a two-component Gaussian mixture.
# Reduced grid: K = 10, 10^2, ..., 10^6 and shorter timing windows (--quick).
set -eo pipefail; cd "$(dirname "$0")"; mkdir -p results
for dist in laplace mixture; do
  python bench_cpu_e2e.py --N 32 --kmin 10 --kmax 1e6 --stds 0.001,0.01,0.1,1,10,100 --dist $dist --quick --out results/$dist.json | tee results/$dist.log
  python plot_cpu_sigma.py results/$dist.json results/speed_sigma_all_cpu_$dist | tee results/${dist}_summary.txt
done
