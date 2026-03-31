#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="cuda_decode_small_ops_ext",
    ext_modules=[
        CUDAExtension(
            name="cuda_decode_small_ops_ext",
            sources=[
                "decode_small_ops_ext.cpp",
                "decode_small_ops_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
