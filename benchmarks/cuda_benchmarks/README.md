
# CUDA Benchmarking

This directory contains scripts for benchmarking the CUDA implementation of Familywise Loss against PyTorch Cross Entropy, sorting-based FWLoss, and top-k-based FWLoss.

## Prerequisites

Follow the [CUDA installation instructions](../../src/cuda/README.md) before running the benchmarks.

## Benchmark Configuration

The following parameters can be modified in `CE_FW_perf.py`:

- **Batch size (`B`):** Number of samples per batch.
- **Number of classes (`K`):** List of class counts used in the experiments.
- **Scaling factors (`sigma`):** List of input distribution scaling parameters.
- **Input distribution:** Change the data generation function to select the desired distribution.

Available data generation functions in `data_io.py`:

- `generate_data()` — Gaussian distribution.
- `generate_data_laplace()` — Laplace distribution.
- `generate_data_gaussian_mixture()` — Gaussian mixture distribution.

For example, to use Gaussian inputs:

```python
inputs, targets = generate_data(B, k, scaling_factor, device)
```
When changing the input distribution, update the corresponding result filenames and plotting function in `CE_FW_perf.py`.

## Experimental Settings

The default benchmark configuration is:

- **Batch size:** 32
- **Number of classes:** 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000
- **Scaling factors:** 0.001, 0.01, 0.1, 1.0, 10.0, 100.0
- **Warm-up iterations:** 20
- **Timed iterations:** 200
- **Timing method:** CUDA events
- **Reported statistic:** Median execution time (ms)
## Running Benchmarks

From the repository root:

```bash
python benchmarks/cuda_benchmarks/CE_FW_perf.py
```

## Results

The benchmark compares the combined forward and backward execution times of PyTorch Cross Entropy, sorting-based FWLoss, top-k-based FWLoss, and the CUDA implementation.

The script also performs correctness checks against the reference implementation.

Generated outputs are saved in:

- `results/` — CSV files containing execution times in milliseconds.
- `figs/` — Performance comparison plots.

## Reproducibility

Benchmark runtimes may vary depending on the GPU model, CUDA version, PyTorch version, and system configuration.

