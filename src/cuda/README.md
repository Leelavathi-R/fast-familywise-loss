# Building the CUDA Extension

## Requirements

- Python
- PyTorch with CUDA support
- CUDA Toolkit (including `nvcc`)
- Compatible C++ compiler

## Installation
Clone the repository and navigate to its root directory:

```bash
git clone https://github.com/Leelavathi-R/fast-familywise-loss.git
cd fast-familywise-loss
```

Create and activate a Conda environment:

```bash
conda create -n fwloss python=3.11 -y
conda activate fwloss
```
Install PyTorch with CUDA support, following the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Verify that PyTorch detects your GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
```

Install the CUDA extension in editable mode:

```bash
pip install -e . --no-build-isolation
```

PyTorch generally detects the GPU architecture automatically.

### CUDA Architecture Issues

If installation fails due to an unsupported or unknown CUDA architecture, check your GPU's compute capability:

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```

For example, `(8, 9)` corresponds to compute capability `8.9`. You can explicitly specify the architecture:

```bash
TORCH_CUDA_ARCH_LIST="8.9" pip install -e . --no-build-isolation
```

Replace `8.9` with the compute capability supported by your GPU and CUDA toolkit.

## Verify Installation

Verify that the extension can be imported:

```bash
python -c "import fw_radix_cuda; print('CUDA extension loaded successfully')"
```
