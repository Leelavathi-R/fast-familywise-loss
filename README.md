# Fast-Familywise-Loss
Efficient CPU and GPU implementations of Familywise Loss for multiclass classification.

## Implementations
- **CPU:** [Build and Install](src/cpu/README.md)
- **CUDA (GPU):** [Build and Install](src/cuda/README.md)

## Benchmarking
- **CPU:** Single-pass threshold search [Benchmark instructions](benchmarks/cpu_benchmarks/README.md)
- **CUDA (GPU):** FWLoss via radix partitioning [Benchmark instructions](benchmarks/cuda_benchmarks/README.md)
- **Reference:** A baseline implementation for correctness verification and performance comparison. [`reference_impl.py`](reference_impl.py)

## License
