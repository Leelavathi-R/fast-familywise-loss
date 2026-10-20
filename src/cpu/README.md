# Familywise loss: CPU kernel and experiments

This folder contains the CPU implementation of the familywise (FW) loss described in the paper
(single-pass threshold search), its correctness tests, and the scripts that produce every CPU
timing figure and number.

## Requirements

- Linux, x86-64 CPU with AVX2 or AVX-512
- a C++17 compiler (`g++` 9 or newer)
- Python 3.9 or newer with the packages in `requirements.txt` (PyTorch 2.x, NumPy, Matplotlib)

No GPU is needed. The kernel is compiled automatically on first import (about one minute) with
`-O3 -march=native`.

    pip install -r requirements.txt

## Quick start

    bash run_all.sh        # tests + all experiments, about 15 minutes on a 2-core machine

or step by step:

| Command | What it does | Time | Output in `results/` |
|---|---|---|---|
| `bash run_tests.sh` | correctness tests | 3 min | printed report |
| `bash run_gaussian.sh` | main experiment: Gaussian logits | 7 min | `gaussian.json`, `speed_sigma_all_cpu.pdf`, `speed_sigma_row_cpu.pdf`, `gaussian_summary.txt` |
| `bash run_other_distributions.sh` | Laplace and Gaussian-mixture logits | 4 min | `laplace.json`, `mixture.json`, `speed_sigma_all_cpu_laplace.pdf`, `speed_sigma_all_cpu_mixture.pdf`, `*_summary.txt` |

The first line printed by `run_tests.sh` must report `SIMD path: avx512` or `SIMD path: avx2`.
If it reports `scalar`, the CPU or the compiler flags do not provide AVX2 and the timings of the
paper do not apply.

## Using the loss

    from fwcpp import familywise_loss
    loss = familywise_loss(logits, target, reduction="mean")   # logits: [N, K] float32 CPU tensor, target: [N] int64
    loss.backward()

It is a drop-in replacement for `torch.nn.functional.cross_entropy` on CPU (`reduction` is `"mean"`
or `"sum"`).

## Experimental protocol

- One timed call is `logits.grad = None; loss(logits, target, reduction="sum").backward()`,
  identical for all methods.
- FW (ours) and cross entropy (PyTorch) are timed alternately, and the order is swapped every round.
  The naive FW implementation (sort + cumulative sum, autograd) is timed separately.
- Reported values are medians; bands in the figures are the 20-80 quantiles.
- Before timing, each configuration is checked against the naive implementation in double precision
  (loss: relative tolerance 1e-5; gradient: absolute tolerance 1e-6).
- Setup: batch size 32, single precision, sigma in {0.001, 0.01, 0.1, 1, 10, 100}.
  - Gaussian: logits ~ N(0, sigma^2); K = 10 ... 10^6, 11 values.
  - Laplace: mean 0, standard deviation sigma; K = 10, 10^2, ..., 10^6.
  - Mixture: each logit is drawn from N(0, 1) or from N(2, sigma^2) with probability 1/2 each;
    K = 10, 10^2, ..., 10^6.

`plot_cpu_sigma.py` prints, for each sigma, the range of the speed-up over cross entropy, the mean
number of non-zero gradient entries (kappa-bar), and the counts quoted in the paper ("FW faster than
CE: x / y", "disjoint 20-80 bands").

## Shipped results

`results/` contains the measurements used in the paper, taken on a 2-core Intel Xeon (2.30 GHz,
AVX-512) virtual machine with PyTorch 2.14 and GCC 13.3:

| File | Configuration |
|---|---|
| `gaussian.json` | AVX-512 build, 2 threads |
| `gaussian_avx2_1thread.json` | AVX2 build, 1 thread |
| `laplace.json`, `mixture.json` | AVX-512 build, 2 threads, reduced grid |

To reproduce the AVX2 single-thread run:

    FW_CPU_CFLAGS="-mavx2 -mfma" OMP_NUM_THREADS=1 python bench_cpu_e2e.py --N 32 --kmin 10 --kmax 1e6 \
        --stds 0.001,0.01,0.1,1,10,100 --out results/gaussian_avx2_1thread.json
    python plot_cpu_sigma.py results/gaussian_avx2_1thread.json results/speed_sigma_all_cpu_avx2

Running the scripts overwrites the files in `results/` with measurements from your machine.
Absolute times depend on the hardware; the comparison between methods is what the paper reports.

## Files

| File | Content |
|---|---|
| `fw_cpu.cpp` | the kernel: chunked threshold search, vectorized for AVX-512 and AVX2 with a scalar fallback, sparse backward pass, median-of-medians safeguard |
| `fwcpp.py` | builds and loads the kernel; exposes `familywise_loss` |
| `bench_cpu_e2e.py` | timing of FW (ours), cross entropy and naive FW; options `--dist gaussian\|laplace\|mixture`, `--N`, `--kmin`, `--kmax`, `--stds`, `--quick`, `--out` |
| `plot_cpu_sigma.py` | 2x3 figure (one panel per sigma) and the printed summary |
| `plot_cpu_sigma_row.py` | the same six panels in one row (main-text figure) |
| `test_cpp.py`, `test_cpu_gen.py` | 224,000 samples against a float64 reference computed by sorting: Gaussian, ties, constant, sorted, two-level and outlier inputs, magnitudes up to 1e30, float32 and float64, K from 1 to 40,000; autograd checks |
| `test_dense.py` | rows with nearly equal logits (dense gradient) |
| `test_cpp_train.py` | gradient through a linear layer and a short SGD run |

## Options

- `FW_CPU_CFLAGS` selects the instruction set at build time, e.g. `"-mavx2 -mfma"` (AVX2 only) or
  `"-mno-avx"` (scalar fallback). Default: `-march=native`.
- `OMP_NUM_THREADS` sets the number of threads used by both PyTorch and the kernel.
