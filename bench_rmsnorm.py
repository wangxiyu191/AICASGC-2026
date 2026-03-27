#!/usr/bin/env python3
"""Benchmark RMSNorm implementations: PyTorch vs Triton (minimal) vs torch.compile."""
import torch
import time
import triton
import triton.language as tl


# === 1. Minimal Triton RMSNorm ===
@triton.jit
def _rms_norm_fwd_kernel(
    X, Y, W, stride, N: tl.constexpr, eps, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def triton_rms_norm(x, weight, eps=1e-6):
    orig_shape = x.shape
    out = torch.empty_like(x)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_2d = x.reshape(M, N)
    out_2d = out.reshape(M, N)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_fwd_kernel[(M,)](x_2d, out_2d, weight, x_2d.stride(0), N, eps, BLOCK_N=BLOCK_N)
    return out


# === 2. PyTorch baseline (current implementation) ===
def pytorch_rms_norm(x, weight, eps=1e-6):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


# === 3. torch.compile version ===
@torch.compile(mode='max-autotune', fullgraph=True)
def compiled_rms_norm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


# === 4. flash_attn triton version ===
from flash_attn.ops.triton.layer_norm import rms_norm_fn as fa_rms_norm_fn

def fa_rms_norm(x, weight, eps=1e-6):
    return fa_rms_norm_fn(x, weight, bias=None, eps=eps)


def benchmark_fn(fn, args, warmup=100, iters=1000):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def main():
    configs = [
        ('decode:  bs=1,   dim=1536', (1, 1536)),
        ('decode:  bs=1,   dim=128',  (1, 128)),
        ('prefill: bs=690, dim=1536', (690, 1536)),
        ('prefill: bs=690, dim=128',  (690, 128)),
    ]

    header = f"{'Config':40s}  {'PyTorch':>9s}  {'Triton':>9s}  {'Compiled':>9s}  {'FA-Triton':>9s}  {'T-spdup':>8s}  {'C-spdup':>8s}"
    print(header)
    print('-' * len(header))

    for name, shape in configs:
        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        w = torch.ones(shape[-1], dtype=torch.float16, device='cuda')
        eps = 1e-6

        t_pt = benchmark_fn(pytorch_rms_norm, (x, w, eps))
        t_tr = benchmark_fn(triton_rms_norm, (x, w, eps))
        t_cp = benchmark_fn(compiled_rms_norm, (x, w, eps), warmup=200)
        t_fa = benchmark_fn(fa_rms_norm, (x, w, eps))

        # Correctness
        ref = pytorch_rms_norm(x, w, eps)
        diff_tr = (triton_rms_norm(x, w, eps).float() - ref.float()).abs().max().item()
        diff_cp = (compiled_rms_norm(x, w, eps).float() - ref.float()).abs().max().item()

        print(f'{name:40s}  {t_pt:7.1f}us  {t_tr:7.1f}us  {t_cp:7.1f}us  {t_fa:7.1f}us  '
              f'{t_pt/t_tr:7.2f}x  {t_pt/t_cp:7.2f}x  '
              f'(diff: tr={diff_tr:.0e}, cp={diff_cp:.0e})')


if __name__ == '__main__':
    main()
