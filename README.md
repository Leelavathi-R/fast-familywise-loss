# Fast-Familywise-Loss
Efficient CPU and GPU implementations of Familywise Loss for multiclass classification.

## Implementations
The repository includes:

- **CPU:** Single-pass threshold search [Build and Install](src/cpu/README.md)
- **CUDA (GPU):** FWLoss via radix partitioning  [Build and Install](src/cuda/README.md)
- **Reference:** A baseline implementation for correctness verification and performance comparison. [`reference_impl.py`](reference_impl.py)

## Benchmarking
The repository includes scripts for evaluating the performance of FWLoss across different numbers of classes and input distributions.

- **CPU:**  [Benchmark instructions](benchmarks/cpu_benchmarks/README.md)
- **CUDA (GPU):** [Benchmark instructions](benchmarks/cuda_benchmarks/README.md)

## License
