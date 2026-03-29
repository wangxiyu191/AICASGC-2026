#!/usr/bin/env python3
"""Benchmark decode-phase linear implementations on real captured tensors.

This tool mirrors benchmark-style prefill + custom decode, captures one layer's
decode-time `qkv` / `gate_up` / `o_proj` / `down_proj` / `lm_head` inputs,
then benchmarks a few linear-call variants.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from datasets import load_from_disk
from transformers.cache_utils import StaticCache

from benchmark import get_system_info
from bench_decode_attention import _prepare_inputs
from evaluation_wrapper import VLMModel
from optimized_layer import set_small_decode_triton_enabled


class LinearCapture:
    def __init__(self, decode_step: int, target_layer, lm_head):
        self.decode_step = decode_step
        self.qkv_call_index = 0
        self.gate_up_call_index = 0
        self.o_proj_call_index = 0
        self.down_proj_call_index = 0
        self.lm_head_call_index = 0
        self._seen_prefill_lm_head = False
        self.payload = {}
        self._qkv_weight_ptr = target_layer.qkv_weight.data_ptr()
        self._gate_up_weight_ptr = target_layer.gate_up_weight.data_ptr()
        self._lm_head_weight_ptr = lm_head.weight.data_ptr()

    def maybe_capture_f_linear(self, x, weight, bias):
        if x.shape[1] != 1:
            return
        weight_ptr = weight.data_ptr()
        if weight_ptr == self._qkv_weight_ptr:
            if self.qkv_call_index == self.decode_step and "qkv" not in self.payload:
                self.payload["qkv"] = {
                    "x": x.detach().clone(),
                    "weight": weight.detach().clone(),
                    "bias": bias.detach().clone() if bias is not None else None,
                }
            self.qkv_call_index += 1
        elif weight_ptr == self._gate_up_weight_ptr:
            if self.gate_up_call_index == self.decode_step and "gate_up" not in self.payload:
                self.payload["gate_up"] = {
                    "x": x.detach().clone(),
                    "weight": weight.detach().clone(),
                    "bias": bias.detach().clone() if bias is not None else None,
                }
            self.gate_up_call_index += 1

    def maybe_capture_o_proj(self, module, args):
        x = args[0]
        if x.shape[1] != 1:
            return
        if self.o_proj_call_index == self.decode_step and "o_proj" not in self.payload:
            self.payload["o_proj"] = {
                "x": x.detach().clone(),
                "weight": module.weight.detach().clone(),
                "bias": module.bias.detach().clone() if module.bias is not None else None,
            }
        self.o_proj_call_index += 1

    def maybe_capture_down_proj(self, module, args):
        x = args[0]
        if x.shape[1] != 1:
            return
        if self.down_proj_call_index == self.decode_step and "down_proj" not in self.payload:
            self.payload["down_proj"] = {
                "x": x.detach().clone(),
                "weight": module.weight.detach().clone(),
                "bias": module.bias.detach().clone() if module.bias is not None else None,
            }
        self.down_proj_call_index += 1

    def maybe_capture_lm_head(self, module, args):
        x = args[0]
        if x.shape[1] != 1:
            return
        if not self._seen_prefill_lm_head:
            self._seen_prefill_lm_head = True
            return
        if self.lm_head_call_index == self.decode_step and "lm_head" not in self.payload:
            self.payload["lm_head"] = {
                "x": x.detach().clone(),
                "weight": module.weight.detach().clone(),
                "bias": module.bias.detach().clone() if module.bias is not None else None,
            }
        self.lm_head_call_index += 1


def _build_bucketed_mask(vlm: VLMModel, inputs, max_new_tokens: int, bucketed: bool):
    prompt_length = inputs.input_ids.shape[1]
    engine = vlm.model._custom_decode
    max_cache_len = prompt_length + max_new_tokens
    if bucketed:
        max_cache_len = engine._get_graph_bucket_len(max_cache_len)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        prompt_attention_mask = torch.ones((1, prompt_length), dtype=torch.bool, device=vlm.device)
    else:
        prompt_attention_mask = attention_mask.to(device=vlm.device, dtype=torch.bool)

    full_attention_mask = torch.zeros((1, max_cache_len), dtype=torch.bool, device=vlm.device)
    full_attention_mask[:, :prompt_length] = prompt_attention_mask
    return prompt_attention_mask, full_attention_mask, max_cache_len


def _run_capture(
    vlm: VLMModel,
    inputs,
    decode_step: int,
    max_new_tokens: int,
    bucketed: bool,
    layer_idx: int,
):
    prompt_length = inputs.input_ids.shape[1]
    prompt_attention_mask, full_attention_mask, max_cache_len = _build_bucketed_mask(
        vlm,
        inputs,
        max_new_tokens=max_new_tokens,
        bucketed=bucketed,
    )
    static_cache = StaticCache(config=vlm._model.config, max_cache_len=max_cache_len)
    model = vlm._model
    target_layer = model.model.language_model.layers[layer_idx]
    capture = LinearCapture(decode_step=decode_step, target_layer=target_layer, lm_head=model.lm_head)
    o_handle = target_layer.o_proj.register_forward_pre_hook(capture.maybe_capture_o_proj)
    d_handle = target_layer.down_proj.register_forward_pre_hook(capture.maybe_capture_down_proj)
    l_handle = model.lm_head.register_forward_pre_hook(capture.maybe_capture_lm_head)

    previous_small_decode_triton = set_small_decode_triton_enabled(model, False)
    orig_f_linear = F.linear

    def _capturing_linear(x, weight, bias=None):
        capture.maybe_capture_f_linear(x, weight, bias)
        return orig_f_linear(x, weight, bias)

    try:
        F.linear = _capturing_linear
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                past_key_values=static_cache,
                use_cache=True,
                logits_to_keep=1,
            )

            rope_deltas = outputs.rope_deltas
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            current_position = prompt_length
            engine = vlm.model._custom_decode

            for step in range(max_new_tokens - 1):
                full_attention_mask[:, current_position] = True
                position_ids = engine._build_decode_position_ids(
                    current_position=current_position,
                    rope_deltas=rope_deltas,
                    device=torch.device(vlm.device),
                )
                cache_position = torch.tensor([current_position], dtype=torch.long, device=vlm.device)
                outputs = model(
                    input_ids=next_tokens,
                    attention_mask=full_attention_mask,
                    position_ids=position_ids,
                    past_key_values=static_cache,
                    use_cache=True,
                    cache_position=cache_position,
                    logits_to_keep=1,
                )
                next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                if {"qkv", "gate_up", "o_proj", "down_proj", "lm_head"} <= capture.payload.keys():
                    return capture.payload, {
                        "prompt_length": int(prompt_length),
                        "max_cache_len": int(max_cache_len),
                        "captured_decode_step": int(step),
                        "layer_idx": int(layer_idx),
                    }
                current_position += 1
    finally:
        F.linear = orig_f_linear
        set_small_decode_triton_enabled(model, previous_small_decode_triton)
        o_handle.remove()
        d_handle.remove()
        l_handle.remove()

    raise RuntimeError(
        "Failed to capture linear tensors. "
        f"decode_step={decode_step}, layer_idx={layer_idx}"
    )


def _module_linear(x, weight, bias):
    return F.linear(x, weight, bias)


def _f_linear_contig(x, weight, bias):
    return F.linear(x, weight.contiguous(), bias)


def _matmul_t(x, weight, bias):
    x2d = x.reshape(-1, x.shape[-1])
    y2d = x2d @ weight.t()
    if bias is not None:
        y2d = y2d + bias
    return y2d.view(*x.shape[:-1], weight.shape[0])


@triton.jit
def _matvec_fwd_kernel(
    X,
    W,
    BIAS,
    OUT,
    K,
    N,
    stride_w_n,
    stride_w_k,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        w_ptrs = W + offs_m[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
        w = tl.load(w_ptrs, mask=(offs_m[:, None] < N) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_m, mask=offs_m < N, other=0.0).to(tl.float32)
        acc += bias
    tl.store(OUT + offs_m, acc.to(OUT.dtype.element_ty), mask=offs_m < N)


def _triton_matvec(x, weight, bias):
    x_vec = x.reshape(-1).contiguous()
    out_vec = torch.empty((weight.shape[0],), dtype=x.dtype, device=x.device)
    block_m = 64
    block_k = 256 if weight.shape[1] >= 4096 else 128
    _matvec_fwd_kernel[(triton.cdiv(weight.shape[0], block_m),)](
        x_vec,
        weight,
        bias if bias is not None else out_vec,
        out_vec,
        weight.shape[1],
        weight.shape[0],
        weight.stride(0),
        weight.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return out_vec.view(*x.shape[:-1], weight.shape[0])


@triton.jit
def _splitk_matvec_fwd_kernel(
    X,
    W,
    OUT_FP32,
    K,
    N,
    CHUNK_K,
    stride_w_n,
    stride_w_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_split = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    chunk_start = pid_split * CHUNK_K
    chunk_end = tl.minimum(chunk_start + CHUNK_K, K)
    for start_k in range(chunk_start, chunk_end, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_k, mask=offs_k < chunk_end, other=0.0).to(tl.float32)
        w_ptrs = W + offs_m[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
        w = tl.load(w_ptrs, mask=(offs_m[:, None] < N) & (offs_k[None, :] < chunk_end), other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.atomic_add(OUT_FP32 + offs_m, acc, mask=offs_m < N)


@triton.jit
def _splitk_matvec_finalize_kernel(
    OUT_FP32,
    BIAS,
    OUT,
    N,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.load(OUT_FP32 + offs_m, mask=offs_m < N, other=0.0)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_m, mask=offs_m < N, other=0.0).to(tl.float32)
    tl.store(OUT + offs_m, acc.to(OUT.dtype.element_ty), mask=offs_m < N)


def _triton_splitk_matvec_impl(x, weight, bias, split_k: int, block_m: int, block_k: int):
    x_vec = x.reshape(-1).contiguous()
    n = weight.shape[0]
    k = weight.shape[1]
    chunk_k = triton.cdiv(k, split_k)
    scratch = torch.zeros((n,), dtype=torch.float32, device=x.device)
    out_vec = torch.empty((n,), dtype=x.dtype, device=x.device)
    _splitk_matvec_fwd_kernel[(triton.cdiv(n, block_m), split_k)](
        x_vec,
        weight,
        scratch,
        k,
        n,
        chunk_k,
        weight.stride(0),
        weight.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    _splitk_matvec_finalize_kernel[(triton.cdiv(n, block_m),)](
        scratch,
        bias if bias is not None else out_vec,
        out_vec,
        n,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )
    return out_vec.view(*x.shape[:-1], n)


def _triton_splitk4_matvec(x, weight, bias):
    block_k = 256 if weight.shape[1] >= 4096 else 128
    return _triton_splitk_matvec_impl(x, weight, bias, split_k=4, block_m=128, block_k=block_k)


def _triton_splitk8_matvec(x, weight, bias):
    block_k = 256 if weight.shape[1] >= 4096 else 128
    return _triton_splitk_matvec_impl(x, weight, bias, split_k=8, block_m=128, block_k=block_k)


def _bench_impl(fn, x, weight, bias, warmup: int, iters: int):
    for _ in range(warmup):
        fn(x, weight, bias)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(x, weight, bias)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return out, elapsed_ms


def _bench_impl_graph(fn, x, weight, bias, warmup: int, iters: int):
    """Benchmark using CUDAGraph replay to eliminate Python dispatch overhead."""
    # Eager warmup to JIT-compile Triton kernels
    for _ in range(warmup):
        fn(x, weight, bias)
    torch.cuda.synchronize()

    # Capture CUDAGraph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(x, weight, bias)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Warmup replays
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    # Timed replays
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return out, elapsed_ms


def _run_benchmarks(payload, warmup: int, iters: int, use_graph: bool = False):
    variants = {
        "f_linear": _module_linear,
        "f_linear_contiguous_weight": _f_linear_contig,
        "matmul_weight_t": _matmul_t,
        "triton_matvec": _triton_matvec,
        "triton_splitk4_matvec": _triton_splitk4_matvec,
        "triton_splitk8_matvec": _triton_splitk8_matvec,
    }
    bench_fn = _bench_impl_graph if use_graph else _bench_impl
    results = {}
    ref_out, ref_ms = bench_fn(
        variants["f_linear"],
        payload["x"],
        payload["weight"],
        payload["bias"],
        warmup=warmup,
        iters=iters,
    )
    results["f_linear"] = {"time_ms": ref_ms, "status": "ok"}
    for name, fn in variants.items():
        if name == "f_linear":
            continue
        try:
            out, ms = bench_fn(
                fn,
                payload["x"],
                payload["weight"],
                payload["bias"],
                warmup=warmup,
                iters=iters,
            )
            diff = (out - ref_out).abs().float()
            results[name] = {
                "time_ms": ms,
                "status": "ok",
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
                "speedup_vs_ref": ref_ms / ms if ms > 0 else None,
            }
        except Exception as exc:
            results[name] = {
                "status": "error",
                "error": str(exc),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark decode linear implementations on real captured tensors")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--decode-step", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--graph", action="store_true", default=True, help="Use CUDAGraph replay to eliminate dispatch overhead (default)")
    parser.add_argument("--no-graph", dest="graph", action="store_false", help="Use eager mode (includes Python dispatch overhead)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"results/linear_microbench_sample{args.sample_idx}_"
            f"layer{args.layer_idx}_step{args.decode_step}.json"
        )

    print("[LinearBench] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    print(
        "[LinearBench] Capturing tensors: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"layer_idx={args.layer_idx}, decode_step={args.decode_step}, bucketed={args.bucketed}"
    )
    payloads, capture_meta = _run_capture(
        vlm,
        inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        layer_idx=args.layer_idx,
    )

    bench_mode = "graph" if args.graph else "eager"
    print(
        f"[LinearBench] Benchmarking implementations (mode={bench_mode})..."
    )

    output = {
        "system_info": get_system_info(),
        "metadata": {
            "question_id": int(sample["question_id"]),
            "sample_idx": int(args.sample_idx),
            "layer_idx": int(args.layer_idx),
            "decode_step": int(args.decode_step),
            "bucketed": bool(args.bucketed),
            "bench_mode": bench_mode,
            "prompt_length": capture_meta["prompt_length"],
            "max_cache_len": capture_meta["max_cache_len"],
            "captured_decode_step": capture_meta["captured_decode_step"],
            "qkv_input_shape": list(payloads["qkv"]["x"].shape),
            "qkv_weight_shape": list(payloads["qkv"]["weight"].shape),
            "qkv_weight_is_contiguous": bool(payloads["qkv"]["weight"].is_contiguous()),
            "gate_up_input_shape": list(payloads["gate_up"]["x"].shape),
            "gate_up_weight_shape": list(payloads["gate_up"]["weight"].shape),
            "gate_up_weight_is_contiguous": bool(payloads["gate_up"]["weight"].is_contiguous()),
            "o_proj_input_shape": list(payloads["o_proj"]["x"].shape),
            "o_proj_weight_shape": list(payloads["o_proj"]["weight"].shape),
            "o_proj_weight_is_contiguous": bool(payloads["o_proj"]["weight"].is_contiguous()),
            "down_proj_input_shape": list(payloads["down_proj"]["x"].shape),
            "down_proj_weight_shape": list(payloads["down_proj"]["weight"].shape),
            "down_proj_weight_is_contiguous": bool(payloads["down_proj"]["weight"].is_contiguous()),
            "lm_head_input_shape": list(payloads["lm_head"]["x"].shape),
            "lm_head_weight_shape": list(payloads["lm_head"]["weight"].shape),
            "lm_head_weight_is_contiguous": bool(payloads["lm_head"]["weight"].is_contiguous()),
            "dtype": str(payloads["o_proj"]["x"].dtype),
            "device": str(payloads["o_proj"]["x"].device),
        },
        "qkv_benchmarks": _run_benchmarks(payloads["qkv"], warmup=args.warmup, iters=args.iters, use_graph=args.graph),
        "gate_up_benchmarks": _run_benchmarks(payloads["gate_up"], warmup=args.warmup, iters=args.iters, use_graph=args.graph),
        "o_proj_benchmarks": _run_benchmarks(payloads["o_proj"], warmup=args.warmup, iters=args.iters, use_graph=args.graph),
        "down_proj_benchmarks": _run_benchmarks(payloads["down_proj"], warmup=args.warmup, iters=args.iters, use_graph=args.graph),
        "lm_head_benchmarks": _run_benchmarks(payloads["lm_head"], warmup=args.warmup, iters=args.iters, use_graph=args.graph),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    for target in ("qkv", "gate_up", "o_proj", "down_proj", "lm_head"):
        print(f"[LinearBench] {target}")
        for name, info in output[f"{target}_benchmarks"].items():
            if info.get("status") != "ok":
                print(f"  {name:<24} ERROR  {info.get('error', 'unknown error')}")
                continue
            speedup = info.get("speedup_vs_ref")
            speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
            print(f"  {name:<24} {info['time_ms']:.4f} ms  {speedup_text}")

    print(f"[LinearBench] Saved to: {output_path}")


if __name__ == "__main__":
    main()
