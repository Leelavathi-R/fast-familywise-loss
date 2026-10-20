
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

## Running Benchmarks

From the repository root:

```bash
python benchmarks/cuda_benchmarks/CE_FW_perf.py
```

## Results

The benchmark results and generated figures are saved in:

- `results/` — CSV files containing execution times.
- `figs/` — Performance plots.

The benchmark measures the combined forward and backward execution time in milliseconds, reporting the median over 200 trials after 20 warm-up iterations.

