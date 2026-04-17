#!/usr/bin/env python3
"""Sweep launch_blocks for the current fused_post_attn_mlp_softsync decode kernel.

Captures one real decode-time payload from the benchmark path, then benchmarks
the exact softsync MLP kernel:
  fused_post_attn_mlp_softsync_k2048_k6144_fwd_kernel<1024, 32>
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")
os.environ.setdefault("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC", "0")
os.environ.setdefault("ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT", "0")
os.environ.setdefault("ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_SOFTSYNC", "0")
os.environ.setdefault("ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT", "1")
os.environ.setdefault("ENABLE_CUDA_FUSED_ADD_RMSNORM_GATEUP", "0")
os.environ.setdefault("ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ", "0")
os.environ.setdefault("ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ_MEGA", "0")
os.environ.setdefault("FUSED_POST_ATTN_MLP_THREADS", "1024")

import torch
from datasets import load_from_disk
from transformers.cache_utils import StaticCache

from benchmark import get_system_info
from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    benchmark_style_cuda_reset,
    make_l2_flush_runner,
)
from bench_decode_attention import _prepare_inputs
import cuda_decode_gemv
from evaluation_wrapper import VLMModel
from optimized_layer import set_cuda_mlp_fusion_flags, set_small_decode_triton_enabled


@dataclass
class CapturedPayload:
    residual: torch.Tensor
    attn_output: torch.Tensor
    norm_weight: torch.Tensor
    gate_up_weight: torch.Tensor
    down_proj_weight: torch.Tensor
    down_proj_bias: torch.Tensor | None
    eps: float
    threads_per_block: int


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


def _make_workspace(device: torch.device, dtype: torch.dtype, n: int = 2048) -> dict:
    return {
        "type": "fused_post_attn_mlp_softsync",
        "barrier_count": torch.zeros((1,), dtype=torch.int32, device=device),
        "barrier_generation": torch.zeros((1,), dtype=torch.int32, device=device),
        "hidden_out": torch.empty((6144,), dtype=dtype, device=device),
        "out": torch.empty((n,), dtype=dtype, device=device),
    }


def _parse_launch_blocks(spec: str | None, num_sms: int) -> list[int]:
    if spec is None or not spec.strip():
        return list(range(1, num_sms + 1))
    values: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))
    blocks = sorted(v for v in values if 1 <= v <= num_sms)
    if not blocks:
        raise ValueError(f"no valid launch_blocks in [1, {num_sms}]")
    return blocks


class SoftsyncCapture:
    def __init__(self, decode_step: int):
        self.decode_step = decode_step
        self.decode_call_index = 0
        self.capture_armed = False
        self.payload: CapturedPayload | None = None

    def layer_pre_hook(self, module, args):
        hidden_states = args[0]
        if hidden_states.shape[1] != 1:
            return
        self.capture_armed = self.decode_call_index == self.decode_step
        self.decode_call_index += 1

    def layer_post_hook(self, module, args, output):
        self.capture_armed = False

    def maybe_capture(
        self,
        residual: torch.Tensor,
        attn_output: torch.Tensor,
        norm_weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        down_proj_bias: torch.Tensor | None,
        eps: float,
        threads_per_block: int,
    ) -> None:
        if not self.capture_armed or self.payload is not None:
            return
        self.payload = CapturedPayload(
            residual=residual.detach().clone(),
            attn_output=attn_output.detach().clone(),
            norm_weight=norm_weight.detach().clone(),
            gate_up_weight=gate_up_weight.detach(),
            down_proj_weight=down_proj_weight.detach(),
            down_proj_bias=None if down_proj_bias is None else down_proj_bias.detach().clone(),
            eps=float(eps),
            threads_per_block=int(threads_per_block),
        )
        self.capture_armed = False


def _capture_payload(
    vlm: VLMModel,
    inputs,
    decode_step: int,
    max_new_tokens: int,
    bucketed: bool,
    layer_idx: int,
) -> tuple[CapturedPayload, dict]:
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
    capture = SoftsyncCapture(decode_step=decode_step)
    pre_handle = target_layer.register_forward_pre_hook(capture.layer_pre_hook)
    post_handle = target_layer.register_forward_hook(capture.layer_post_hook)

    previous_small_decode_triton = set_small_decode_triton_enabled(model, True)
    previous_cuda_flags = set_cuda_mlp_fusion_flags(
        add_rmsnorm_gateup=False,
        full_layer_softsync=False,
        post_attn_mlp_persistent=True,
        o_proj_post_attn_mlp_persistent=False,
        o_proj_post_attn_mlp_softsync=False,
        silu_downproj=False,
        gateup_silu_downproj=False,
        gateup_silu_downproj_mega=False,
    )
    orig_softsync = cuda_decode_gemv.fused_post_attn_mlp_softsync_half

    def _capturing_softsync(
        residual,
        attn_output,
        norm_weight,
        gate_up_weight,
        down_proj_weight,
        bias,
        eps,
        workspace=None,
        threads_per_block=1024,
        launch_blocks=None,
    ):
        capture.maybe_capture(
            residual,
            attn_output,
            norm_weight,
            gate_up_weight,
            down_proj_weight,
            bias,
            eps,
            threads_per_block,
        )
        return orig_softsync(
            residual,
            attn_output,
            norm_weight,
            gate_up_weight,
            down_proj_weight,
            bias,
            eps,
            workspace=workspace,
            threads_per_block=threads_per_block,
            launch_blocks=launch_blocks,
        )

    try:
        cuda_decode_gemv.fused_post_attn_mlp_softsync_half = _capturing_softsync
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
                if capture.payload is not None:
                    return capture.payload, {
                        "prompt_length": int(prompt_length),
                        "max_cache_len": int(max_cache_len),
                        "captured_decode_step": int(step),
                        "layer_idx": int(layer_idx),
                        "bucketed": bool(bucketed),
                    }
                next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                current_position += 1
    finally:
        cuda_decode_gemv.fused_post_attn_mlp_softsync_half = orig_softsync
        set_small_decode_triton_enabled(model, previous_small_decode_triton)
        set_cuda_mlp_fusion_flags(
            add_rmsnorm_gateup=previous_cuda_flags[0],
            full_layer_softsync=previous_cuda_flags[1],
            post_attn_mlp_persistent=previous_cuda_flags[2],
            o_proj_post_attn_mlp_persistent=previous_cuda_flags[3],
            o_proj_post_attn_mlp_softsync=previous_cuda_flags[4],
            silu_downproj=previous_cuda_flags[5],
            gateup_silu_downproj=previous_cuda_flags[6],
            gateup_silu_downproj_mega=previous_cuda_flags[7],
        )
        pre_handle.remove()
        post_handle.remove()

    raise RuntimeError(
        "Failed to capture fused_post_attn_mlp_softsync payload. "
        f"decode_step={decode_step}, layer_idx={layer_idx}"
    )


def _run_kernel(
    payload: CapturedPayload,
    workspace: dict,
    launch_blocks: int,
):
    return cuda_decode_gemv.fused_post_attn_mlp_softsync_half(
        payload.residual,
        payload.attn_output,
        payload.norm_weight,
        payload.gate_up_weight,
        payload.down_proj_weight,
        payload.down_proj_bias,
        payload.eps,
        workspace=workspace,
        threads_per_block=payload.threads_per_block,
        launch_blocks=launch_blocks,
    )


def _bench_launch_blocks(
    payload: CapturedPayload,
    launch_blocks: int,
    warmup: int,
    iters: int,
    pre_replay,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    workspace = _make_workspace(payload.residual.device, payload.residual.dtype)

    benchmark_style_cuda_reset(pre_replay=pre_replay)
    for _ in range(warmup):
        _run_kernel(payload, workspace, launch_blocks)
    torch.cuda.synchronize()

    benchmark_style_cuda_reset(pre_replay=pre_replay)
    start_events = []
    end_events = []
    hidden = None
    out = None
    for _ in range(iters):
        if pre_replay is not None:
            pre_replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        hidden, out = _run_kernel(payload, workspace, launch_blocks)
        end.record()
        start_events.append(start)
        end_events.append(end)
    torch.cuda.synchronize()
    elapsed_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / iters
    return elapsed_ms, hidden.detach().clone(), out.detach().clone()


def main():
    parser = argparse.ArgumentParser(description="Sweep launch_blocks for post-attn MLP softsync kernel")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--l2-flush-mb", type=int, default=DEFAULT_BENCHMARK_L2_FLUSH_MB)
    parser.add_argument("--launch-blocks", type=str, default=None, help="Comma-separated values and/or ranges, e.g. 32,48,64-108")
    parser.add_argument(
        "--output",
        type=str,
        default=f"results/post_attn_mlp_softsync_sms_sweep_sample0_layer14_step16_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    launch_blocks_list = _parse_launch_blocks(args.launch_blocks, num_sms)

    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    payload, capture_meta = _capture_payload(
        vlm,
        inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        layer_idx=args.layer_idx,
    )

    pre_replay = make_l2_flush_runner(device, args.l2_flush_mb)
    default_workspace = _make_workspace(device, payload.residual.dtype)
    ref_hidden, ref_out = cuda_decode_gemv.fused_post_attn_mlp_softsync_half(
        payload.residual,
        payload.attn_output,
        payload.norm_weight,
        payload.gate_up_weight,
        payload.down_proj_weight,
        payload.down_proj_bias,
        payload.eps,
        workspace=default_workspace,
        threads_per_block=payload.threads_per_block,
        launch_blocks=num_sms,
    )
    torch.cuda.synchronize()

    results = []
    for launch_blocks in launch_blocks_list:
        elapsed_ms, hidden, out = _bench_launch_blocks(
            payload,
            launch_blocks=launch_blocks,
            warmup=args.warmup,
            iters=args.iters,
            pre_replay=pre_replay,
        )
        max_abs_diff = float((out - ref_out).abs().max().item())
        hidden_max_abs_diff = float((hidden - ref_hidden).abs().max().item())
        results.append(
            {
                "launch_blocks": int(launch_blocks),
                "elapsed_ms": float(elapsed_ms),
                "max_abs_diff_vs_default": max_abs_diff,
                "hidden_max_abs_diff_vs_default": hidden_max_abs_diff,
            }
        )
        print(
            f"launch_blocks={launch_blocks:3d}  "
            f"elapsed_ms={elapsed_ms:.4f}  "
            f"out_diff={max_abs_diff:.3e}  hidden_diff={hidden_max_abs_diff:.3e}",
            flush=True,
        )

    results.sort(key=lambda item: item["elapsed_ms"])
    best = results[0]
    default_result = next(item for item in results if item["launch_blocks"] == num_sms)

    payload_meta = {
        "residual_shape": list(payload.residual.shape),
        "attn_output_shape": list(payload.attn_output.shape),
        "norm_weight_shape": list(payload.norm_weight.shape),
        "gate_up_weight_shape": list(payload.gate_up_weight.shape),
        "down_proj_weight_shape": list(payload.down_proj_weight.shape),
        "down_proj_bias_shape": None if payload.down_proj_bias is None else list(payload.down_proj_bias.shape),
        "eps": payload.eps,
        "threads_per_block": payload.threads_per_block,
    }
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": get_system_info(),
        "benchmark": {
            "script": "bench_post_attn_mlp_softsync_sms_sweep.py",
            "sample_idx": args.sample_idx,
            "layer_idx": args.layer_idx,
            "decode_step": args.decode_step,
            "max_new_tokens": args.max_new_tokens,
            "bucketed": args.bucketed,
            "warmup": args.warmup,
            "iters": args.iters,
            "l2_flush_mb": args.l2_flush_mb,
            "num_sms": num_sms,
            "launch_blocks_list": launch_blocks_list,
        },
        "capture_meta": capture_meta,
        "payload_meta": payload_meta,
        "best": {
            "launch_blocks": int(best["launch_blocks"]),
            "elapsed_ms": float(best["elapsed_ms"]),
            "speedup_vs_default": float(default_result["elapsed_ms"] / best["elapsed_ms"]),
        },
        "default": {
            "launch_blocks": int(default_result["launch_blocks"]),
            "elapsed_ms": float(default_result["elapsed_ms"]),
        },
        "results": sorted(results, key=lambda item: item["launch_blocks"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(
        "\nBest launch_blocks="
        f"{best['launch_blocks']} elapsed_ms={best['elapsed_ms']:.4f} "
        f"(default {default_result['launch_blocks']} -> {default_result['elapsed_ms']:.4f} ms, "
        f"speedup {default_result['elapsed_ms'] / best['elapsed_ms']:.4f}x)"
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
