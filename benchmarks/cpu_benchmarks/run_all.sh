#!/bin/bash
set -eo pipefail; cd "$(dirname "$0")"
bash run_tests.sh
bash run_gaussian.sh
bash run_other_distributions.sh
echo "DONE. Figures and numbers are in results/."
