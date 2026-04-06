#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_decode_attention_ext",
    ext_modules=[
        CUDAExtension(
            name="cuda_decode_attention_ext",
            sources=[
                "decode_attention_ext.cpp",
                "decode_attention_kernel.cu",
                "decode_attention_v4_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
