#!/usr/bin/env python3
"""Benchmark one decode-layer chain on real benchmark-path payloads.

This tool captures real tensors from one decode step / one layer, then builds a
single CUDAGraph replay that approximates the layer's decode forward path:

input_rmsnorm -> qkv_gemv -> q_norm_rope -> k_norm_rope -> attention
-> o_proj_gemv -> post_attn_add_rmsnorm -> gate_up_gemv -> silu_mul
-> down_proj_gemv -> residual_add

Notes:
- KV cache update is not replayed here. Attention uses the already-captured
  post-update q/k/v tensors from the real decode path.
- Replay intentionally calls torch.cuda.empty_cache() + synchronize() before
  each replay to mimic the user's requested benchmark-style stress case.
- Timing only uses one CUDA event pair around each whole-graph replay. No
  internal per-stage events are inserted into the graph.
- Optional VizTracer support traces the replay loop directly without adding any
  extra CUDA events inside the graph.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")
os.environ.setdefault("ENABLE_CUDA_NOSPLIT_GEMV", "1")

import torch
from datasets import load_from_disk
from transformers.cache_utils import StaticCache

import cuda_decode_gemv
import optimized_layer
from benchmark import get_system_info
from bench_decode_attention import (
    _build_bucketed_mask as _build_bucketed_mask_attention,
    _prepare_inputs,
)
from evaluation_wrapper import VLMModel
from optimized_layer import set_small_decode_triton_enabled


class LayerChainCapture:
    def __init__(self, decode_step: int, target_layer):
        self.decode_step = decode_step
        self.target_layer = target_layer
        self.decode_call_index = 0
        self.payload = {}
        self._active_target_call = False
        self._o_proj_weight_ptr = target_layer.o_proj_weight.data_ptr()
        self._down_proj_weight_ptr = target_layer.down_proj_weight.data_ptr()

    def layer_pre_hook(self, _module, args, kwargs):
        hidden_states = args[0] if args else kwargs["hidden_states"]
        if hidden_states.shape[1] != 1:
            return
        if self.decode_call_index == self.decode_step:
            self._active_target_call = True
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            if position_embeddings is None:
                raise RuntimeError("failed to capture position_embeddings for target layer")
            self.payload.update({
                "hidden_states": hidden_states.detach().clone(),
                "cos": position_embeddings[0].detach().clone(),
                "sin": position_embeddings[1].detach().clone(),
            })
        self.decode_call_index += 1

    def layer_post_hook(self, _module, _args, _kwargs, _output):
        self._active_target_call = False

    def maybe_capture_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        num_key_value_groups: int,
        cache_position: torch.Tensor | None,
    ):
        if not self._active_target_call or q.shape[2] != 1 or "attention" in self.payload:
            return
        self.payload["attention"] = {
            "q": q.detach().clone(),
            "k": k.detach().clone(),
            "v": v.detach().clone(),
            "attention_mask": attention_mask.detach().clone() if attention_mask is not None else None,
            "scaling": float(scaling),
            "num_key_value_groups": int(num_key_value_groups),
            "cache_position": cache_position.detach().clone() if cache_position is not None else None,
        }

    def maybe_capture_gemv(self, x, weight, bias):
        if not self._active_target_call or x.shape[-2] != 1:
            return
        weight_ptr = weight.data_ptr()
        if weight_ptr == self._o_proj_weight_ptr and "o_proj_input" not in self.payload:
            self.payload["o_proj_input"] = x.detach().clone()
        elif weight_ptr == self._down_proj_weight_ptr and "down_proj_input" not in self.payload:
            self.payload["down_proj_input"] = x.detach().clone()

    def is_complete(self) -> bool:
        return {
            "hidden_states",
            "cos",
            "sin",
            "attention",
            "o_proj_input",
            "down_proj_input",
        } <= self.payload.keys()


def _run_layer_chain_capture(
    vlm: VLMModel,
    inputs,
    decode_step: int,
    max_new_tokens: int,
    bucketed: bool,
    layer_idx: int,
):
    model = vlm._model
    target_layer = model.model.language_model.layers[layer_idx]
    capture = LayerChainCapture(decode_step=decode_step, target_layer=target_layer)
    layer_pre_handle = target_layer.register_forward_pre_hook(
        capture.layer_pre_hook,
        with_kwargs=True,
    )
    layer_post_handle = target_layer.register_forward_hook(
        capture.layer_post_hook,
        with_kwargs=True,
    )

    original_attention = optimized_layer._small_decode_attention
    original_cuda_nosplit_gemv = optimized_layer._cuda_nosplit_gemv

    def _capturing_attention(
        q,
        k,
        v,
        attention_mask,
        scaling,
        num_key_value_groups,
        prefer_small_triton=False,
        cache_position=None,
        splitkv_workspace=None,
    ):
        capture.maybe_capture_attention(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
        return original_attention(
            q,
            k,
            v,
            attention_mask,
            scaling,
            num_key_value_groups,
            prefer_small_triton=prefer_small_triton,
            cache_position=cache_position,
            splitkv_workspace=splitkv_workspace,
        )

    def _capturing_cuda_nosplit_gemv(x, weight, bias):
        capture.maybe_capture_gemv(x, weight, bias)
        return original_cuda_nosplit_gemv(x, weight, bias)

    prompt_length = inputs.input_ids.shape[1]
    prompt_attention_mask, full_attention_mask, max_cache_len = _build_bucketed_mask_attention(
        vlm,
        inputs,
        max_new_tokens=max_new_tokens,
        bucketed=bucketed,
    )
    static_cache = StaticCache(config=vlm._model.config, max_cache_len=max_cache_len)
    previous_small_decode_triton = set_small_decode_triton_enabled(model, True)
    optimized_layer._small_decode_attention = _capturing_attention
    optimized_layer._cuda_nosplit_gemv = _capturing_cuda_nosplit_gemv

    try:
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
                if capture.is_complete():
                    return {
                        "layer_input": capture.payload["hidden_states"],
                        "cos": capture.payload["cos"],
                        "sin": capture.payload["sin"],
                        "attention": capture.payload["attention"],
                        "o_proj_input": capture.payload["o_proj_input"],
                        "down_proj_input": capture.payload["down_proj_input"],
                    }, {
                        "prompt_length": int(prompt_length),
                        "max_cache_len": int(max_cache_len),
                        "captured_decode_step": int(step),
                        "layer_idx": int(layer_idx),
                    }
                current_position += 1
    finally:
        optimized_layer._small_decode_attention = original_attention
        optimized_layer._cuda_nosplit_gemv = original_cuda_nosplit_gemv
        set_small_decode_triton_enabled(model, previous_small_decode_triton)
        layer_pre_handle.remove()
        layer_post_handle.remove()

    raise RuntimeError(
        "Failed to capture target layer-chain payload. "
        f"decode_step={decode_step}, layer_idx={layer_idx}"
    )


def _benchmark_style_cuda_reset():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _decode_gemv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    if (
        x.device.type == "cuda"
        and weight.device.type == "cuda"
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.ndim >= 2
        and x.shape[-2] == 1
    ):
        return cuda_decode_gemv.nosplit_matvec_half(x, weight, bias)
    return torch.nn.functional.linear(x, weight, bias)


def _event_stats_ms(values_ms: list[float]) -> dict[str, float]:
    ordered = sorted(values_ms)
    idx_95 = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean_ms": float(mean(values_ms)),
        "median_ms": float(median(values_ms)),
        "p95_ms": float(ordered[idx_95]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _run_layer_chain_graph(target_layer, payload):
    prefer_small_triton = target_layer._prefer_small_decode_triton
    if getattr(target_layer, "_splitkv_workspace", None) is None and os.environ.get(
        "ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV", "1"
    ) == "1":
        target_layer._ensure_splitkv_workspace()

    layer_input = payload["layer_input"]
    cos = payload["cos"]
    sin = payload["sin"]
    attn = payload["attention"]
    input_shape = layer_input.shape[:-1]
    hidden_shape_q = (*input_shape, target_layer.num_heads, target_layer.head_dim)
    hidden_shape_kv = (*input_shape, target_layer.num_kv_heads, target_layer.head_dim)

    outputs = {}
    _benchmark_style_cuda_reset()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden_states = optimized_layer._rms_norm(
                layer_input,
                target_layer.input_layernorm_weight,
                target_layer.rms_norm_eps,
                prefer_small_triton=prefer_small_triton,
            )

            qkv = _decode_gemv(hidden_states, target_layer.qkv_weight, None)

            q, k_cur, _v_cur = qkv.split([target_layer.q_size, target_layer.kv_size, target_layer.kv_size], dim=-1)

            q = optimized_layer._fused_rms_norm_rope(
                q.view(hidden_shape_q),
                target_layer.q_norm_weight,
                target_layer.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=prefer_small_triton,
            ).transpose(1, 2)

            _k_cur = optimized_layer._fused_rms_norm_rope(
                k_cur.view(hidden_shape_kv),
                target_layer.k_norm_weight,
                target_layer.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=prefer_small_triton,
            ).transpose(1, 2)

            attn_output = optimized_layer._small_decode_attention(
                attn["q"],
                attn["k"],
                attn["v"],
                attn["attention_mask"],
                attn["scaling"],
                attn["num_key_value_groups"],
                prefer_small_triton=prefer_small_triton,
                cache_position=attn["cache_position"],
                splitkv_workspace=target_layer._splitkv_workspace,
            )
            if attn_output is None:
                raise RuntimeError("layer-chain benchmark expected small decode attention path")

            attn_output = attn_output.reshape(*input_shape, -1)
            if not attn_output.is_contiguous():
                attn_output = attn_output.contiguous()
            outputs["o_proj_input_simulated"] = attn_output

            attn_output = _decode_gemv(attn_output, target_layer.o_proj_weight, target_layer.o_proj_bias)

            residual, hidden_states = optimized_layer._fused_add_rms_norm(
                layer_input,
                attn_output,
                target_layer.post_attn_layernorm_weight,
                target_layer.rms_norm_eps,
                prefer_small_triton=prefer_small_triton,
            )

            gate_up = _decode_gemv(hidden_states, target_layer.gate_up_weight, None)

            gate, up = gate_up.chunk(2, dim=-1)

            hidden_states = optimized_layer._fused_silu_mul(
                gate,
                up,
                prefer_small_triton=prefer_small_triton,
            )
            outputs["down_proj_input_simulated"] = hidden_states

            hidden_states = _decode_gemv(hidden_states, target_layer.down_proj_weight, target_layer.down_proj_bias)

            hidden_states = hidden_states + residual

            outputs["q_after_rope"] = q
            outputs["final_hidden_states"] = hidden_states
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # Validate against a real replay result instead of capture-time graph-private buffers.
    graph.replay()
    torch.cuda.synchronize()

    q_diff = (outputs["q_after_rope"] - attn["q"]).abs().float()
    o_proj_input_diff = (outputs["o_proj_input_simulated"] - payload["o_proj_input"]).abs().float()
    down_proj_input_diff = (outputs["down_proj_input_simulated"] - payload["down_proj_input"]).abs().float()

    return {
        "graph": graph,
        "outputs": outputs,
        "validation": {
            "q_after_rope_vs_captured_q_max_abs_diff": float(q_diff.max().item()),
            "q_after_rope_vs_captured_q_mean_abs_diff": float(q_diff.mean().item()),
            "attention_output_vs_captured_o_proj_input_max_abs_diff": float(o_proj_input_diff.max().item()),
            "attention_output_vs_captured_o_proj_input_mean_abs_diff": float(o_proj_input_diff.mean().item()),
            "silu_mul_output_vs_captured_down_proj_input_max_abs_diff": float(down_proj_input_diff.max().item()),
            "silu_mul_output_vs_captured_down_proj_input_mean_abs_diff": float(down_proj_input_diff.mean().item()),
        },
    }


def _measure_graph_replay(graph, warmup: int, iters: int) -> dict[str, float]:
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        _benchmark_style_cuda_reset()
        total_start.record()
        graph.replay()
        total_end.record()
        torch.cuda.synchronize()

    total_values_ms = []
    for _ in range(iters):
        _benchmark_style_cuda_reset()
        total_start.record()
        graph.replay()
        total_end.record()
        torch.cuda.synchronize()
        total_values_ms.append(total_start.elapsed_time(total_end))
    return _event_stats_ms(total_values_ms)


def _trace_graph_replay(graph, warmup: int, trace_iters: int, trace_output: Path):
    from viztracer import VizTracer

    for _ in range(warmup):
        _benchmark_style_cuda_reset()
        graph.replay()
        torch.cuda.synchronize()

    tracer = VizTracer(log_torch=True, output_file=str(trace_output), max_stack_depth=15, verbose=0)
    tracer.start()
    try:
        for _ in range(trace_iters):
            _benchmark_style_cuda_reset()
            graph.replay()
            torch.cuda.synchronize()
    finally:
        tracer.stop()
        tracer.save()


def main():
    parser = argparse.ArgumentParser(description="Benchmark one decode-layer chain on real benchmark-path payloads")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=36)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--viztrace", action="store_true", help="Capture VizTracer trace for the replay loop.")
    parser.add_argument("--viztrace-iters", type=int, default=1, help="Number of traced graph replays.")
    parser.add_argument("--viztrace-output", type=str, default=None, help="Trace JSON output path under traces/.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"results/layer_chain_microbench_sample{args.sample_idx}_"
            f"layer{args.layer_idx}_step{args.decode_step}.json"
        )

    print("[LayerChain] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    print(
        "[LayerChain] Capturing payloads: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"layer_idx={args.layer_idx}, decode_step={args.decode_step}, bucketed={args.bucketed}"
    )
    payload, capture_meta = _run_layer_chain_capture(
        vlm,
        inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        layer_idx=args.layer_idx,
    )

    target_layer = vlm._model.model.language_model.layers[args.layer_idx]
    print(
        "[LayerChain] Benchmarking one-layer chain "
        f"(warmup={args.warmup}, iters={args.iters}, replay_reset=empty_cache+synchronize)..."
    )
    previous_small_decode_triton = set_small_decode_triton_enabled(vlm._model, True)
    try:
        graph_result = _run_layer_chain_graph(
            target_layer=target_layer,
            payload=payload,
        )
    finally:
        set_small_decode_triton_enabled(vlm._model, previous_small_decode_triton)

    total_replay_timing = _measure_graph_replay(
        graph=graph_result["graph"],
        warmup=args.warmup,
        iters=args.iters,
    )

    viztrace_output = None
    if args.viztrace:
        if args.viztrace_output is None:
            args.viztrace_output = (
                f"traces/layer_chain/layer_chain_viztrace_"
                f"sample{args.sample_idx}_layer{args.layer_idx}_step{args.decode_step}.json"
            )
        viztrace_output = Path(args.viztrace_output)
        viztrace_output.parent.mkdir(parents=True, exist_ok=True)
        _trace_graph_replay(
            graph=graph_result["graph"],
            warmup=args.warmup,
            trace_iters=args.viztrace_iters,
            trace_output=viztrace_output,
        )

    output = {
        "system_info": get_system_info(),
        "metadata": {
            "question_id": int(sample["question_id"]),
            "sample_idx": int(args.sample_idx),
            "layer_idx": int(args.layer_idx),
            "decode_step": int(args.decode_step),
            "bucketed": bool(args.bucketed),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "prompt_length": int(capture_meta["prompt_length"]),
            "max_cache_len": int(capture_meta["max_cache_len"]),
            "captured_decode_step": int(capture_meta["captured_decode_step"]),
            "replay_reset": "torch.cuda.empty_cache + torch.cuda.synchronize",
            "timing_mode": "cuda_event_total_replay_only",
            "viztrace_enabled": bool(args.viztrace),
            "viztrace_iters": int(args.viztrace_iters) if args.viztrace else 0,
            "viztrace_output": None if viztrace_output is None else str(viztrace_output),
            "note": "cache update omitted; attention uses captured post-update q/k/v",
        },
        "results": {
            "total_replay_timing": total_replay_timing,
            "validation": graph_result["validation"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print("[LayerChain] Total replay timing:")
    total_stats = output["results"]["total_replay_timing"]
    print(f"  {'total':<24} {total_stats['mean_ms']*1000.0:8.2f} us")
    print("[LayerChain] Validation:")
    for name, value in output["results"]["validation"].items():
        print(f"  {name:<48} {value:.6f}")
    if viztrace_output is not None:
        print(f"[LayerChain] VizTracer saved to: {viztrace_output}")
    print(f"[LayerChain] Saved to: {output_path}")


if __name__ == "__main__":
    main()
