#!/bin/bash
# Correctness tests of the CPU kernel (about 3 minutes).
set -eo pipefail; cd "$(dirname "$0")"
python -c "import fwcpp, torch; print('SIMD path:', fwcpp.simd(), '| torch', torch.__version__, '| threads', torch.get_num_threads())"
python test_cpp.py          # 224,000 samples against a float64 reference computed by sorting; autograd checks
python test_dense.py        # rows with nearly equal logits
python test_cpp_train.py    # gradient through a linear layer; a short SGD run
