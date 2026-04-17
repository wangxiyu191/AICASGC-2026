#!/usr/bin/env python3
"""Benchmark fixed-seq prefill text subgraphs with eager attention in the middle.

This is a tooling-only prototype:
- Reuses the real benchmark-path prefill fastpath to capture the input entering
  a chosen decoder layer range.
- Keeps attention eager.
- Captures only the pre-attention and post-attention non-attention segments of
  each layer with `torch.cuda.CUDAGraph`.
- Uses one fixed sample length at a time, so no bucket management is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers.cache_utils import StaticCache

from bench_decode_attention import _prepare_inputs
from benchmark import get_system_info
from benchmark_cuda_utils import DEFAULT_BENCHMARK_L2_FLUSH_MB, make_l2_flush_runner
from evaluation_wrapper import VLMModel
from optimized_layer import (
    _add,
    _cuda_nosplit_gemv,
    _cuda_partition_gemv,
    _fused_add_rms_norm,
    _fused_qk_rms_norm_rope,
    _fused_rms_norm_rope,
    _fused_silu_mul,
    _fused_silu_mul_packed,
    _rms_norm,
    _triton_decode_gemv,
)


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


def _shape_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape_list(v) for v in value]
    return None


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _maybe_clone_tensor(value):
    return None if value is None else value.detach().clone()


def _copy_if_present(dst: torch.Tensor | None, src: torch.Tensor | None):
    if dst is not None and src is not None:
        dst.copy_(src)


def _capture_prefill_text_payload(
    vlm: VLMModel,
    inputs,
    start_layer_idx: int,
):
    model = vlm._model
    model_core = model.model
    language_model = model_core.language_model
    captured: dict[str, object] = {}

    def _layer_pre_hook(_module, args, kwargs):
        if captured.get("start_hidden") is not None:
            return
        hidden_states = args[0] if args else kwargs["hidden_states"]
        position_embeddings = kwargs.get("position_embeddings")
        captured["start_hidden"] = hidden_states.detach().clone()
        captured["attention_mask"] = _maybe_clone_tensor(kwargs.get("attention_mask"))
        captured["position_ids"] = _maybe_clone_tensor(kwargs.get("position_ids"))
        captured["position_embeddings"] = (
            tuple(t.detach().clone() for t in position_embeddings)
            if position_embeddings is not None
            else None
        )
        captured["prefill_static_cache_no_mask"] = bool(kwargs.get("prefill_static_cache_no_mask", False))

    def _norm_hook(_module, _args, output):
        captured["ref_norm_out"] = output.detach().clone()

    handles = [
        language_model.layers[start_layer_idx].register_forward_pre_hook(_layer_pre_hook, with_kwargs=True),
        language_model.norm.register_forward_hook(_norm_hook),
    ]

    prompt_attention_mask = inputs.get("attention_mask")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
    else:
        prompt_attention_mask = prompt_attention_mask.to(dtype=torch.bool)

    static_cache = StaticCache(
        config=model.config,
        max_cache_len=int(inputs.input_ids.shape[1]),
    )
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
    finally:
        for handle in handles:
            handle.remove()

    required = ("start_hidden", "position_embeddings", "ref_norm_out")
    missing = [key for key in required if captured.get(key) is None]
    if missing:
        raise RuntimeError(f"Failed to capture prefill text payload fields: {missing}")

    captured["logits"] = outputs.logits.detach().clone()
    captured["rope_deltas"] = _maybe_clone_tensor(getattr(outputs, "rope_deltas", None))
    return captured


def _run_layers_current(
    language_model,
    start_hidden: torch.Tensor,
    position_embeddings,
    start_layer_idx: int,
    end_layer_idx: int,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    prefill_static_cache_no_mask: bool,
):
    hidden_states = start_hidden.clone()
    example_inputs: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for layer_idx in range(start_layer_idx, end_layer_idx + 1):
            example_inputs[layer_idx] = hidden_states.detach().clone()
            hidden_states = language_model.layers[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=None,
                position_embeddings=position_embeddings,
                prefill_static_cache_no_mask=prefill_static_cache_no_mask,
            )
        hidden_states = language_model.norm(hidden_states)
    return hidden_states, example_inputs


class PrefillLayerSubgraphRunner:
    def __init__(
        self,
        layer,
        example_hidden: torch.Tensor,
        position_embeddings,
        warmup_steps: int = 2,
    ):
        self.layer = layer
        self.position_embeddings = position_embeddings
        self.hidden_in_buf = example_hidden.detach().clone()
        self.attn_out_buf = torch.empty_like(self.hidden_in_buf)
        self.pre_graph = None
        self.post_graph = None
        self.q_out = None
        self.k_out = None
        self.v_out = None
        self.hidden_out = None
        self._capture(max(1, warmup_steps))

    def _run_pre_segment(self):
        input_shape = self.hidden_in_buf.shape[:-1]
        hidden_shape_q = (*input_shape, self.layer.num_heads, self.layer.head_dim)
        hidden_shape_kv = (*input_shape, self.layer.num_kv_heads, self.layer.head_dim)

        hidden_states = _rms_norm(
            self.hidden_in_buf,
            self.layer.input_layernorm_weight,
            self.layer.rms_norm_eps,
            prefer_small_triton=False,
        )
        qkv = _cuda_partition_gemv(hidden_states, f"layer{self.layer.layer_idx}.qkv")
        if qkv is None:
            qkv = _cuda_nosplit_gemv(hidden_states, self.layer.qkv_weight, None)
        if qkv is None:
            qkv = F.linear(hidden_states, self.layer.qkv_weight)
        q, k, v = qkv.split([self.layer.q_size, self.layer.kv_size, self.layer.kv_size], dim=-1)

        q_view = q.view(hidden_shape_q)
        k_view = k.view(hidden_shape_kv)
        cos, sin = self.position_embeddings
        fused_qk = _fused_qk_rms_norm_rope(
            q_view,
            k_view,
            self.layer.q_norm_weight,
            self.layer.k_norm_weight,
            self.layer.rms_norm_eps,
            cos,
            sin,
            prefer_small_triton=False,
        )
        if fused_qk is None:
            q_view = _fused_rms_norm_rope(
                q_view,
                self.layer.q_norm_weight,
                self.layer.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=False,
            )
            k_view = _fused_rms_norm_rope(
                k_view,
                self.layer.k_norm_weight,
                self.layer.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=False,
            )
        else:
            q_view, k_view = fused_qk

        self.q_out = q_view.transpose(1, 2)
        self.k_out = k_view.transpose(1, 2)
        self.v_out = v.view(hidden_shape_kv).transpose(1, 2)

    def _run_attention(self):
        attn_output, _ = self.layer._attention_interface(
            self.layer,
            self.q_out,
            self.k_out,
            self.v_out,
            None,
            dropout=0.0,
            scaling=self.layer.scaling,
        )
        attn_output = attn_output.reshape(*self.hidden_in_buf.shape[:-1], -1)
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()
        return attn_output

    def _run_post_segment(self):
        attn_output = _cuda_partition_gemv(self.attn_out_buf, f"layer{self.layer.layer_idx}.o_proj")
        if attn_output is None:
            attn_output = _cuda_nosplit_gemv(self.attn_out_buf, self.layer.o_proj_weight, self.layer.o_proj_bias)
        if attn_output is None:
            attn_output = _triton_decode_gemv(
                self.attn_out_buf,
                self.layer.o_proj_weight,
                self.layer.o_proj_bias,
                prefer_small_triton=False,
            )
        if attn_output is None:
            attn_output = self.layer.o_proj(self.attn_out_buf)

        residual, hidden_states = _fused_add_rms_norm(
            self.hidden_in_buf,
            attn_output,
            self.layer.post_attn_layernorm_weight,
            self.layer.rms_norm_eps,
            prefer_small_triton=False,
        )

        gate_up = _cuda_partition_gemv(hidden_states, f"layer{self.layer.layer_idx}.gate_up")
        if gate_up is None:
            gate_up = _cuda_nosplit_gemv(hidden_states, self.layer.gate_up_weight, None)
        if gate_up is None:
            gate_up = F.linear(hidden_states, self.layer.gate_up_weight)

        hidden_states = _fused_silu_mul_packed(gate_up, prefer_small_triton=False)
        if hidden_states is None:
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = _fused_silu_mul(gate, up, prefer_small_triton=False)

        down_proj = _cuda_nosplit_gemv(hidden_states, self.layer.down_proj_weight, self.layer.down_proj_bias)
        if down_proj is None:
            down_proj = _triton_decode_gemv(
                hidden_states,
                self.layer.down_proj_weight,
                self.layer.down_proj_bias,
                prefer_small_triton=False,
            )
        if down_proj is None:
            down_proj = self.layer.down_proj(hidden_states)

        self.hidden_out = _add(residual, down_proj, prefer_small_triton=False)

    def _capture_graph(self, run_fn, graph_name: str):
        side_stream = torch.cuda.Stream(device=self.hidden_in_buf.device)
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                run_fn()
        torch.cuda.current_stream(device=self.hidden_in_buf.device).wait_stream(side_stream)
        torch.cuda.synchronize(self.hidden_in_buf.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_fn()
        torch.cuda.synchronize(self.hidden_in_buf.device)
        setattr(self, graph_name, graph)

    def _capture(self, warmup_steps: int):
        self._capture_graph(self._run_pre_segment, "pre_graph")
        for _ in range(warmup_steps):
            self.pre_graph.replay()
            attn_output = self._run_attention()
            self.attn_out_buf.copy_(attn_output)
        torch.cuda.synchronize(self.hidden_in_buf.device)
        self.pre_graph.replay()
        self.attn_out_buf.copy_(self._run_attention())
        self._capture_graph(self._run_post_segment, "post_graph")

    def replay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.hidden_in_buf.copy_(hidden_states)
        self.pre_graph.replay()
        self.attn_out_buf.copy_(self._run_attention())
        self.post_graph.replay()
        return self.hidden_out


class PrefillGenericSubgraphRunner:
    def __init__(
        self,
        template_layer,
        example_hidden: torch.Tensor,
        position_embeddings,
        warmup_steps: int = 2,
    ):
        self.template_layer = template_layer
        self.position_embeddings = position_embeddings
        self.hidden_in_buf = example_hidden.detach().clone()
        self.attn_out_buf = torch.empty_like(self.hidden_in_buf)
        self.pre_graph = None
        self.post_graph = None
        self.q_out = None
        self.k_out = None
        self.v_out = None
        self.hidden_out = None

        self.input_layernorm_weight_buf = torch.empty_like(template_layer.input_layernorm_weight)
        self.q_norm_weight_buf = torch.empty_like(template_layer.q_norm_weight)
        self.k_norm_weight_buf = torch.empty_like(template_layer.k_norm_weight)
        self.qkv_weight_buf = torch.empty_like(template_layer.qkv_weight)
        self.o_proj_weight_buf = torch.empty_like(template_layer.o_proj_weight)
        self.o_proj_bias_buf = (
            None if template_layer.o_proj_bias is None else torch.empty_like(template_layer.o_proj_bias)
        )
        self.post_attn_layernorm_weight_buf = torch.empty_like(template_layer.post_attn_layernorm_weight)
        self.gate_up_weight_buf = torch.empty_like(template_layer.gate_up_weight)
        self.down_proj_weight_buf = torch.empty_like(template_layer.down_proj_weight)
        self.down_proj_bias_buf = (
            None if template_layer.down_proj_bias is None else torch.empty_like(template_layer.down_proj_bias)
        )

        self._stage_pre_weights(template_layer)
        self._stage_post_weights(template_layer)
        self._capture(max(1, warmup_steps))

    def _stage_pre_weights(self, layer):
        self.input_layernorm_weight_buf.copy_(layer.input_layernorm_weight)
        self.q_norm_weight_buf.copy_(layer.q_norm_weight)
        self.k_norm_weight_buf.copy_(layer.k_norm_weight)
        self.qkv_weight_buf.copy_(layer.qkv_weight)

    def _stage_post_weights(self, layer):
        self.o_proj_weight_buf.copy_(layer.o_proj_weight)
        _copy_if_present(self.o_proj_bias_buf, layer.o_proj_bias)
        self.post_attn_layernorm_weight_buf.copy_(layer.post_attn_layernorm_weight)
        self.gate_up_weight_buf.copy_(layer.gate_up_weight)
        self.down_proj_weight_buf.copy_(layer.down_proj_weight)
        _copy_if_present(self.down_proj_bias_buf, layer.down_proj_bias)

    def _run_pre_segment(self):
        input_shape = self.hidden_in_buf.shape[:-1]
        hidden_shape_q = (*input_shape, self.template_layer.num_heads, self.template_layer.head_dim)
        hidden_shape_kv = (*input_shape, self.template_layer.num_kv_heads, self.template_layer.head_dim)

        hidden_states = _rms_norm(
            self.hidden_in_buf,
            self.input_layernorm_weight_buf,
            self.template_layer.rms_norm_eps,
            prefer_small_triton=False,
        )
        qkv = F.linear(hidden_states, self.qkv_weight_buf)
        q, k, v = qkv.split([self.template_layer.q_size, self.template_layer.kv_size, self.template_layer.kv_size], dim=-1)

        q_view = q.view(hidden_shape_q)
        k_view = k.view(hidden_shape_kv)
        cos, sin = self.position_embeddings
        fused_qk = _fused_qk_rms_norm_rope(
            q_view,
            k_view,
            self.q_norm_weight_buf,
            self.k_norm_weight_buf,
            self.template_layer.rms_norm_eps,
            cos,
            sin,
            prefer_small_triton=False,
        )
        if fused_qk is None:
            q_view = _fused_rms_norm_rope(
                q_view,
                self.q_norm_weight_buf,
                self.template_layer.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=False,
            )
            k_view = _fused_rms_norm_rope(
                k_view,
                self.k_norm_weight_buf,
                self.template_layer.rms_norm_eps,
                cos,
                sin,
                prefer_small_triton=False,
            )
        else:
            q_view, k_view = fused_qk

        self.q_out = q_view.transpose(1, 2)
        self.k_out = k_view.transpose(1, 2)
        self.v_out = v.view(hidden_shape_kv).transpose(1, 2)

    def _run_attention(self, layer):
        attn_output, _ = layer._attention_interface(
            layer,
            self.q_out,
            self.k_out,
            self.v_out,
            None,
            dropout=0.0,
            scaling=layer.scaling,
        )
        attn_output = attn_output.reshape(*self.hidden_in_buf.shape[:-1], -1)
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()
        return attn_output

    def _run_post_segment(self):
        attn_output = F.linear(self.attn_out_buf, self.o_proj_weight_buf, self.o_proj_bias_buf)
        residual, hidden_states = _fused_add_rms_norm(
            self.hidden_in_buf,
            attn_output,
            self.post_attn_layernorm_weight_buf,
            self.template_layer.rms_norm_eps,
            prefer_small_triton=False,
        )
        gate_up = F.linear(hidden_states, self.gate_up_weight_buf)
        hidden_states = _fused_silu_mul_packed(gate_up, prefer_small_triton=False)
        if hidden_states is None:
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = _fused_silu_mul(gate, up, prefer_small_triton=False)
        down_proj = F.linear(hidden_states, self.down_proj_weight_buf, self.down_proj_bias_buf)
        self.hidden_out = _add(residual, down_proj, prefer_small_triton=False)

    def _capture_graph(self, run_fn, graph_name: str):
        side_stream = torch.cuda.Stream(device=self.hidden_in_buf.device)
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                run_fn()
        torch.cuda.current_stream(device=self.hidden_in_buf.device).wait_stream(side_stream)
        torch.cuda.synchronize(self.hidden_in_buf.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_fn()
        torch.cuda.synchronize(self.hidden_in_buf.device)
        setattr(self, graph_name, graph)

    def _capture(self, warmup_steps: int):
        self._capture_graph(self._run_pre_segment, "pre_graph")
        for _ in range(warmup_steps):
            self.pre_graph.replay()
            attn_output = self._run_attention(self.template_layer)
            self.attn_out_buf.copy_(attn_output)
        torch.cuda.synchronize(self.hidden_in_buf.device)
        self.pre_graph.replay()
        self.attn_out_buf.copy_(self._run_attention(self.template_layer))
        self._capture_graph(self._run_post_segment, "post_graph")

    def replay_layer(self, layer, hidden_states: torch.Tensor) -> torch.Tensor:
        self.hidden_in_buf.copy_(hidden_states)
        self._stage_pre_weights(layer)
        self.pre_graph.replay()
        self.attn_out_buf.copy_(self._run_attention(layer))
        self._stage_post_weights(layer)
        self.post_graph.replay()
        return self.hidden_out


def _build_runners(
    language_model,
    example_inputs: dict[int, torch.Tensor],
    position_embeddings,
    start_layer_idx: int,
    end_layer_idx: int,
    warmup_steps: int,
):
    runners = []
    for layer_idx in range(start_layer_idx, end_layer_idx + 1):
        runners.append(
            PrefillLayerSubgraphRunner(
                layer=language_model.layers[layer_idx],
                example_hidden=example_inputs[layer_idx],
                position_embeddings=position_embeddings,
                warmup_steps=warmup_steps,
            )
        )
    return runners


def _build_generic_runner(
    language_model,
    example_inputs: dict[int, torch.Tensor],
    position_embeddings,
    start_layer_idx: int,
    warmup_steps: int,
):
    return PrefillGenericSubgraphRunner(
        template_layer=language_model.layers[start_layer_idx],
        example_hidden=example_inputs[start_layer_idx],
        position_embeddings=position_embeddings,
        warmup_steps=warmup_steps,
    )


def _run_layers_graphed(
    language_model,
    runners: list[PrefillLayerSubgraphRunner],
    start_hidden: torch.Tensor,
):
    hidden_states = start_hidden
    with torch.no_grad():
        for runner in runners:
            hidden_states = runner.replay(hidden_states)
        hidden_states = language_model.norm(hidden_states)
    return hidden_states


def _run_layers_graphed_generic(
    language_model,
    generic_runner: PrefillGenericSubgraphRunner,
    start_hidden: torch.Tensor,
    start_layer_idx: int,
    end_layer_idx: int,
):
    hidden_states = start_hidden
    with torch.no_grad():
        for layer_idx in range(start_layer_idx, end_layer_idx + 1):
            hidden_states = generic_runner.replay_layer(language_model.layers[layer_idx], hidden_states)
        hidden_states = language_model.norm(hidden_states)
    return hidden_states


def _measure_variant(fn, warmup: int, iters: int, pre_replay):
    for _ in range(warmup):
        if pre_replay is not None:
            pre_replay()
        fn()
    torch.cuda.synchronize()

    host_values_ms = []
    gpu_values_ms = []
    last_out = None
    for _ in range(iters):
        if pre_replay is not None:
            pre_replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start = time.perf_counter()
        start.record()
        last_out = fn()
        end.record()
        torch.cuda.synchronize()
        host_values_ms.append((time.perf_counter() - host_start) * 1000.0)
        gpu_values_ms.append(start.elapsed_time(end))

    return {
        "host_ms": _event_stats_ms(host_values_ms),
        "gpu_ms": _event_stats_ms(gpu_values_ms),
        "last_out": last_out.detach().clone(),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark fixed-seq prefill text subgraphs with eager attention")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--start-layer-idx", type=int, default=3)
    parser.add_argument("--end-layer-idx", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--graph-warmup", type=int, default=2)
    parser.add_argument("--l2-flush-mb", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/prefill_text_subgraphs_sample{args.sample_idx}.json"

    print("[PrefillTextSubgraphs] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    language_model = vlm._model.model.language_model
    end_layer_idx = len(language_model.layers) - 1 if args.end_layer_idx < 0 else int(args.end_layer_idx)
    if args.start_layer_idx < 0 or end_layer_idx < args.start_layer_idx or end_layer_idx >= len(language_model.layers):
        raise ValueError(
            f"Invalid layer range start={args.start_layer_idx}, end={end_layer_idx}, "
            f"num_layers={len(language_model.layers)}"
        )

    print(
        "[PrefillTextSubgraphs] Capturing real prefill payload: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"prompt_length={inputs.input_ids.shape[1]}, layer_range={args.start_layer_idx}-{end_layer_idx}"
    )
    payload = _capture_prefill_text_payload(vlm, inputs, args.start_layer_idx)

    print("[PrefillTextSubgraphs] Running current eager baseline on the captured layer range...")
    baseline_out, example_inputs = _run_layers_current(
        language_model=language_model,
        start_hidden=payload["start_hidden"],
        position_embeddings=payload["position_embeddings"],
        start_layer_idx=args.start_layer_idx,
        end_layer_idx=end_layer_idx,
        position_ids=payload["position_ids"],
        attention_mask=payload["attention_mask"],
        prefill_static_cache_no_mask=bool(payload["prefill_static_cache_no_mask"]),
    )
    ref_diff = _max_abs_diff(baseline_out, payload["ref_norm_out"])
    print(f"[PrefillTextSubgraphs] Full-model ref diff (captured start -> current eager): {ref_diff:.6g}")

    print("[PrefillTextSubgraphs] Building per-layer pre/post CUDAGraph runners...")
    runners = _build_runners(
        language_model=language_model,
        example_inputs=example_inputs,
        position_embeddings=payload["position_embeddings"],
        start_layer_idx=args.start_layer_idx,
        end_layer_idx=end_layer_idx,
        warmup_steps=args.graph_warmup,
    )
    print("[PrefillTextSubgraphs] Building shared 2-graph runner with per-layer weight staging...")
    generic_runner = _build_generic_runner(
        language_model=language_model,
        example_inputs=example_inputs,
        position_embeddings=payload["position_embeddings"],
        start_layer_idx=args.start_layer_idx,
        warmup_steps=args.graph_warmup,
    )

    device = payload["start_hidden"].device
    pre_replay = make_l2_flush_runner(device, args.l2_flush_mb)

    print(
        "[PrefillTextSubgraphs] Benchmarking "
        f"(warmup={args.warmup}, iters={args.iters}, l2_flush_mb={args.l2_flush_mb})..."
    )
    eager_measured = _measure_variant(
        lambda: _run_layers_current(
            language_model=language_model,
            start_hidden=payload["start_hidden"],
            position_embeddings=payload["position_embeddings"],
            start_layer_idx=args.start_layer_idx,
            end_layer_idx=end_layer_idx,
            position_ids=payload["position_ids"],
            attention_mask=payload["attention_mask"],
            prefill_static_cache_no_mask=bool(payload["prefill_static_cache_no_mask"]),
        )[0],
        warmup=args.warmup,
        iters=args.iters,
        pre_replay=pre_replay,
    )
    graph_measured = _measure_variant(
        lambda: _run_layers_graphed(
            language_model=language_model,
            runners=runners,
            start_hidden=payload["start_hidden"],
        ),
        warmup=args.warmup,
        iters=args.iters,
        pre_replay=pre_replay,
    )
    generic_graph_measured = _measure_variant(
        lambda: _run_layers_graphed_generic(
            language_model=language_model,
            generic_runner=generic_runner,
            start_hidden=payload["start_hidden"],
            start_layer_idx=args.start_layer_idx,
            end_layer_idx=end_layer_idx,
        ),
        warmup=args.warmup,
        iters=args.iters,
        pre_replay=pre_replay,
    )

    graph_diff = _max_abs_diff(eager_measured["last_out"], graph_measured["last_out"])
    generic_graph_diff = _max_abs_diff(eager_measured["last_out"], generic_graph_measured["last_out"])
    speedup = eager_measured["gpu_ms"]["median_ms"] / graph_measured["gpu_ms"]["median_ms"]
    delta_ms = eager_measured["gpu_ms"]["median_ms"] - graph_measured["gpu_ms"]["median_ms"]
    generic_speedup = eager_measured["gpu_ms"]["median_ms"] / generic_graph_measured["gpu_ms"]["median_ms"]
    generic_delta_ms = eager_measured["gpu_ms"]["median_ms"] - generic_graph_measured["gpu_ms"]["median_ms"]

    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "prompt_length": int(inputs.input_ids.shape[1]),
        "layer_range": [int(args.start_layer_idx), int(end_layer_idx)],
        "num_layers_captured": int(end_layer_idx - args.start_layer_idx + 1),
        "capture_mode": "fixed_seq_no_bucket",
        "attention_mode": "eager",
        "graph_segments": ["pre_attn_nonattn", "post_attn_nonattn"],
        "graph_topologies": {
            "per_layer": int((end_layer_idx - args.start_layer_idx + 1) * 2),
            "generic_shared": 2,
        },
        "graph_warmup": int(args.graph_warmup),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "l2_flush_mb": int(args.l2_flush_mb),
        "start_hidden_shape": list(payload["start_hidden"].shape),
        "position_ids_shape": _shape_list(payload["position_ids"]),
        "position_embeddings_shape": _shape_list(payload["position_embeddings"]),
        "attention_mask_shape": _shape_list(payload["attention_mask"]),
        "ref_norm_out_shape": list(payload["ref_norm_out"].shape),
        "logits_shape": list(payload["logits"].shape),
        "rope_deltas_shape": _shape_list(payload["rope_deltas"]),
        "device": str(device),
        "dtype": str(payload["start_hidden"].dtype),
    }

    result = {
        "system_info": get_system_info(),
        "metadata": metadata,
        "correctness": {
            "full_model_ref_vs_current_eager_max_abs_diff": ref_diff,
            "current_eager_vs_per_layer_graph_max_abs_diff": graph_diff,
            "current_eager_vs_generic_shared_graph_max_abs_diff": generic_graph_diff,
        },
        "timings": {
            "current_eager": {
                "host_ms": eager_measured["host_ms"],
                "gpu_ms": eager_measured["gpu_ms"],
            },
            "graph_prepost_plus_eager_attention_per_layer": {
                "host_ms": graph_measured["host_ms"],
                "gpu_ms": graph_measured["gpu_ms"],
            },
            "graph_prepost_plus_eager_attention_generic_shared": {
                "host_ms": generic_graph_measured["host_ms"],
                "gpu_ms": generic_graph_measured["gpu_ms"],
            },
            "per_layer_delta_gpu_median_ms": float(delta_ms),
            "per_layer_speedup_vs_eager": float(speedup),
            "generic_shared_delta_gpu_median_ms": float(generic_delta_ms),
            "generic_shared_speedup_vs_eager": float(generic_speedup),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n[PrefillTextSubgraphs] Summary")
    print(
        "  eager gpu mean/median : "
        f"{eager_measured['gpu_ms']['mean_ms']:.3f} / {eager_measured['gpu_ms']['median_ms']:.3f} ms"
    )
    print(
        "  per-layer graph gpu   : "
        f"{graph_measured['gpu_ms']['mean_ms']:.3f} / {graph_measured['gpu_ms']['median_ms']:.3f} ms"
    )
    print(
        "  shared-2graph gpu     : "
        f"{generic_graph_measured['gpu_ms']['mean_ms']:.3f} / {generic_graph_measured['gpu_ms']['median_ms']:.3f} ms"
    )
    print(f"  per-layer delta       : {delta_ms:.3f} ms")
    print(f"  per-layer speedup     : {speedup:.4f}x")
    print(f"  shared-2graph delta   : {generic_delta_ms:.3f} ms")
    print(f"  shared-2graph speedup : {generic_speedup:.4f}x")
    print(f"  ref diff              : {ref_diff:.6g}")
    print(f"  per-layer diff        : {graph_diff:.6g}")
    print(f"  shared-2graph diff    : {generic_graph_diff:.6g}")
    print(f"  output                : {out_path}")


if __name__ == "__main__":
    main()
