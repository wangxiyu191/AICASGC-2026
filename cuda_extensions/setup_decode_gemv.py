#!/usr/bin/env python3
"""Build script for cuda_decode_gemv_ext.

decode_gemv_gridsync.cu uses cooperative_groups and must be compiled with
-rdc=true (relocatable device code) + a device-link step.  PyTorch's
BuildExtension does not handle this automatically, so we pre-compile and
device-link the gridsync TU separately and pass the resulting objects via
extra_objects.
"""

import glob
import os
import subprocess
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import torch
from torch.utils.cpp_extension import include_paths as torch_include_paths

# ---------------------------------------------------------------------------
# Pre-compile decode_gemv_gridsync.cu with -rdc=true + device linking
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRIDSYNC_SRC = os.path.join(_SCRIPT_DIR, "decode_gemv_gridsync.cu")
_GRIDSYNC_OBJ = os.path.join(_SCRIPT_DIR, "decode_gemv_gridsync.o")
_GRIDSYNC_DLINK = os.path.join(_SCRIPT_DIR, "decode_gemv_gridsync_dlink.o")

_NVCC = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
_ARCH = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0").strip()
# Convert "8.0" -> "80", "8.0;8.6" -> first arch
_ARCH_NUM = _ARCH.split(";")[0].split(",")[0].replace(".", "")
_GENCODE = f"-gencode=arch=compute_{_ARCH_NUM},code=sm_{_ARCH_NUM}"

# Gather include paths
_INCLUDES: list[str] = []
for p in torch_include_paths("cuda"):
    _INCLUDES += ["-I", p]
# Python include
_INCLUDES += ["-I", os.path.join(sys.prefix, "include")]
# Virtual env include
_venv_inc = os.path.join(sys.prefix, "include",
                         f"python{sys.version_info.major}.{sys.version_info.minor}")
if os.path.isdir(_venv_inc):
    _INCLUDES += ["-I", _venv_inc]
# System Python include
import sysconfig
_sys_inc = sysconfig.get_path("include")
if _sys_inc and os.path.isdir(_sys_inc):
    _INCLUDES += ["-I", _sys_inc]

_DEFINES = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    '-DPYBIND11_COMPILER_TYPE=\\"_gcc\\"',
    '-DPYBIND11_STDLIB=\\"_libstdcpp\\"',
    '-DPYBIND11_BUILD_ABI=\\"_cxxabi1018\\"',
    "-DTORCH_EXTENSION_NAME=cuda_decode_gemv_ext",
]


def _need_rebuild(target: str, deps: list[str]) -> bool:
    if not os.path.exists(target):
        return True
    t_mtime = os.path.getmtime(target)
    return any(os.path.getmtime(d) > t_mtime for d in deps if os.path.exists(d))


def _precompile_gridsync():
    """Compile decode_gemv_gridsync.cu with -rdc=true and device-link it."""
    deps = [_GRIDSYNC_SRC]

    if _need_rebuild(_GRIDSYNC_OBJ, deps):
        cmd = [
            _NVCC, "-c", "-rdc=true",
            "-O3", "--use_fast_math", "-lineinfo",
            "--extra-device-vectorization",
            "-Xptxas", "--allow-expensive-optimizations=true",
            _GENCODE,
            "--expt-relaxed-constexpr",
            "--compiler-options", "-fPIC",
            "-std=c++17",
        ] + _INCLUDES + _DEFINES + [
            _GRIDSYNC_SRC, "-o", _GRIDSYNC_OBJ,
        ]
        print(f"[gridsync] compiling {_GRIDSYNC_SRC} ...")
        subprocess.run(cmd, check=True, cwd=_SCRIPT_DIR)

    if _need_rebuild(_GRIDSYNC_DLINK, [_GRIDSYNC_OBJ]):
        cmd = [
            _NVCC, "-dlink",
            "-o", _GRIDSYNC_DLINK, _GRIDSYNC_OBJ,
            _GENCODE,
            "--compiler-options", "-fPIC",
            "-lcudadevrt",
        ]
        print(f"[gridsync] device-linking ...")
        subprocess.run(cmd, check=True, cwd=_SCRIPT_DIR)


_precompile_gridsync()


def _invalidate_build_cache_if_needed():
    """Remove cached .so from build/ when gridsync objects are newer, forcing relink.

    setuptools tracks source->object mtime for files listed in ``sources``,
    but ``extra_objects`` (the pre-compiled gridsync .o / _dlink.o) are not
    monitored.  If only gridsync.cu changed, the old .so in build/lib.*/ is
    considered up-to-date and simply *copied* to the inplace directory without
    relinking.  Deleting the stale cached .so solves this.
    """
    objs = [f for f in (_GRIDSYNC_OBJ, _GRIDSYNC_DLINK) if os.path.exists(f)]
    if not objs:
        return
    gridsync_mtime = max(os.path.getmtime(f) for f in objs)

    pattern = os.path.join(_SCRIPT_DIR, "build", "lib.*",
                           "cuda_decode_gemv_ext*.so")
    for cached_so in glob.glob(pattern):
        if os.path.getmtime(cached_so) < gridsync_mtime:
            print(f"[gridsync] removing stale build cache: {cached_so}")
            os.remove(cached_so)


_invalidate_build_cache_if_needed()

# ---------------------------------------------------------------------------
# Main extension (no -rdc=true for decode_gemv_kernel.cu)
# ---------------------------------------------------------------------------

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
                "nvcc": [
                    "-O3", "--use_fast_math", "-lineinfo",
                    "--extra-device-vectorization",
                    "-Xptxas", "--allow-expensive-optimizations=true",
                ],
            },
            extra_objects=[_GRIDSYNC_OBJ, _GRIDSYNC_DLINK],
            libraries=["cudadevrt"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
