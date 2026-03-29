#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="cuda_decode_gemv_ext",
    ext_modules=[
        CUDAExtension(
            name="cuda_decode_gemv_ext",
            sources=[
                "decode_gemv_ext.cpp",
                "decode_gemv_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                # Keep optimized build, but add lineinfo so Nsight Compute can
                # correlate SASS/metrics back to CUDA source with low overhead.
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
