from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fw_radix_cuda",
    ext_modules=[
        CUDAExtension(
            name="fw_radix_cuda",
            sources=[
                "./src/cuda/binding.cpp",
                "./src/cuda/fw_radix_argmax.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)