#!/usr/bin/env python3
"""Capture real decode payloads, run full-layer softsync profiler, and emit HTML."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from statistics import median

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")
os.environ.setdefault("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC", "0")

import torch
import torch.nn.functional as F
from datasets import load_from_disk

import cuda_decode_gemv
from bench_decode_layer_chain import _prepare_inputs, _run_layer_chain_capture
from evaluation_wrapper import VLMModel
import optimized_layer

NUM_ITERS = 20
WARMUP = 10

S_KERNEL_START = 0
S_INPUT_RMSNORM_DONE = 1
S_QKV_DONE = 2
S_SYNC1_START = 3
S_SYNC1_DONE = 4
S_QKVCACHE_DONE = 5
S_SYNC2_START = 6
S_SYNC2_DONE = 7
S_ATTENTION_DONE = 8
S_SYNC3_START = 9
S_SYNC3_DONE = 10
S_OPROJ_DONE = 11
S_SYNC4_START = 12
S_SYNC4_DONE = 13
S_POST_RMSNORM_DONE = 14
S_GATEUP_SILU_DONE = 15
S_SYNC5_START = 16
S_SYNC5_DONE = 17
S_DOWNPROJ_DONE = 18
S_KERNEL_DONE = 19
S_GT_START = 20
S_GT_END = 21
S_SM_ID = 22
S_ATTN_PREFETCH_LINES = 23

PHASES = (
    ("input_rmsnorm", "Input RMSNorm", "#5b8def"),
    ("qkv", "QKV GEMV", "#2a9d8f"),
    ("sync1", "Sync 1", "#6c757d"),
    ("qkvcache", "Q/K Rope + KV Cache", "#9b5de5"),
    ("sync2", "Sync 2", "#6c757d"),
    ("attention", "Attention", "#e76f51"),
    ("sync3", "Sync 3", "#6c757d"),
    ("oproj", "O-Proj", "#f4a261"),
    ("sync4", "Sync 4", "#6c757d"),
    ("post_rmsnorm", "Post-Attn Add+RMSNorm", "#577590"),
    ("gateup_silu", "GateUp + SiLU*Up", "#43aa8b"),
    ("sync5", "Sync 5", "#6c757d"),
    ("downproj", "Down-Proj + Residual", "#f94144"),
)


def _equiv_bandwidth_gbps(data_mb: float, wall_us: float) -> float | None:
    if data_mb <= 0.0 or wall_us <= 0.0:
        return None
    return (data_mb / wall_us) * 1000.0


def _grid_blocks(device: torch.device, launch_blocks: int) -> int:
    props = torch.cuda.get_device_properties(device)
    return max(1, min(props.multi_processor_count, launch_blocks if launch_blocks > 0 else props.multi_processor_count))


def _make_profile_workspace(device: torch.device, dtype: torch.dtype, out_dim: int, grid_blocks: int) -> dict:
    attn_split_k = max(
        2,
        int(os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_SPLIT_K", "4")),
    )
    return {
        "type": "fused_decode_full_layer_softsync_profile",
        "barrier_count": torch.zeros((1,), dtype=torch.int32, device=device),
        "barrier_generation": torch.zeros((1,), dtype=torch.int32, device=device),
        "kv_ready_count": torch.zeros((1,), dtype=torch.int32, device=device),
        "attn_done_count": torch.zeros((1,), dtype=torch.int32, device=device),
        "attn_partial_acc": torch.empty((16, attn_split_k, 128), dtype=torch.float32, device=device),
        "attn_partial_stats": torch.empty((16, attn_split_k, 2), dtype=torch.float32, device=device),
        "attn_split_done_count": torch.zeros((16,), dtype=torch.int32, device=device),
        "qkv_out": torch.empty((4096,), dtype=dtype, device=device),
        "hidden_out": torch.empty((6144,), dtype=dtype, device=device),
        "out": torch.empty((out_dim,), dtype=dtype, device=device),
        "profile": torch.empty(
            (grid_blocks, len(cuda_decode_gemv.FULL_LAYER_PROFILE_SLOT_NAMES)),
            dtype=torch.int64,
            device=device,
        ),
    }


def _make_runtime_workspace(device: torch.device, dtype: torch.dtype, out_dim: int) -> dict:
    attn_split_k = max(
        2,
        int(os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_SPLIT_K", "4")),
    )
    return {
        "type": "fused_decode_full_layer_softsync",
        "barrier_count": torch.zeros((1,), dtype=torch.int32, device=device),
        "barrier_generation": torch.zeros((1,), dtype=torch.int32, device=device),
        "kv_ready_count": torch.zeros((1,), dtype=torch.int32, device=device),
        "attn_done_count": torch.zeros((1,), dtype=torch.int32, device=device),
        "attn_partial_acc": torch.empty((16, attn_split_k, 128), dtype=torch.float32, device=device),
        "attn_partial_stats": torch.empty((16, attn_split_k, 2), dtype=torch.float32, device=device),
        "attn_split_done_count": torch.zeros((16,), dtype=torch.int32, device=device),
        "qkv_out": torch.empty((4096,), dtype=dtype, device=device),
        "hidden_out": torch.empty((6144,), dtype=dtype, device=device),
        "out": torch.empty((out_dim,), dtype=dtype, device=device),
    }


def _capture_payload(args):
    print("[FullLayerProfile] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])
    print(
        "[FullLayerProfile] Capturing payloads: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"layer_idx={args.layer_idx}, decode_step={args.decode_step}, bucketed={args.bucketed}"
    )
    capture_flag_restore = optimized_layer.set_cuda_mlp_fusion_flags(
        add_rmsnorm_gateup=False,
        full_layer_softsync=False,
        post_attn_mlp_persistent=False,
        o_proj_post_attn_mlp_persistent=False,
        o_proj_post_attn_mlp_softsync=False,
        silu_downproj=False,
        gateup_silu_downproj=False,
        gateup_silu_downproj_mega=False,
    )
    try:
        payload, capture_meta = _run_layer_chain_capture(
            vlm,
            inputs,
            decode_step=args.decode_step,
            max_new_tokens=args.max_new_tokens,
            bucketed=args.bucketed,
            layer_idx=args.layer_idx,
        )
    finally:
        optimized_layer.set_cuda_mlp_fusion_flags(
            add_rmsnorm_gateup=capture_flag_restore[0],
            full_layer_softsync=capture_flag_restore[1],
            post_attn_mlp_persistent=capture_flag_restore[2],
            o_proj_post_attn_mlp_persistent=capture_flag_restore[3],
            o_proj_post_attn_mlp_softsync=capture_flag_restore[4],
            silu_downproj=capture_flag_restore[5],
            gateup_silu_downproj=capture_flag_restore[6],
            gateup_silu_downproj_mega=capture_flag_restore[7],
        )
    target_layer = vlm._model.model.language_model.layers[args.layer_idx]
    return vlm, sample, inputs, payload, capture_meta, target_layer


def run_profile(payload, target_layer, launch_blocks: int, warmup: int, iters: int):
    device = payload["layer_input"].device
    runtime_workspace = _make_runtime_workspace(
        device=device,
        dtype=payload["layer_input"].dtype,
        out_dim=payload["layer_input"].shape[-1],
    )
    grid_blocks = _grid_blocks(device, launch_blocks)
    profile_workspace = _make_profile_workspace(device, payload["layer_input"].dtype, payload["layer_input"].shape[-1], grid_blocks)

    call_kwargs = dict(
        hidden_states=payload["layer_input"],
        qkv_weight=target_layer.qkv_weight,
        input_layernorm_weight=target_layer.input_layernorm_weight,
        q_norm_weight=target_layer.q_norm_weight,
        k_norm_weight=target_layer.k_norm_weight,
        cos=payload["cos"],
        sin=payload["sin"],
        k_cache=payload["k_cache"],
        v_cache=payload["v_cache"],
        cache_position=payload["cache_position"],
        o_proj_weight=target_layer.o_proj_weight,
        o_proj_bias=target_layer.o_proj_bias,
        post_attn_norm_weight=target_layer.post_attn_layernorm_weight,
        gate_up_weight=target_layer.gate_up_weight,
        down_proj_weight=target_layer.down_proj_weight,
        down_proj_bias=target_layer.down_proj_bias,
        eps=target_layer.rms_norm_eps,
        attn_scale=target_layer.scaling,
        launch_blocks=launch_blocks,
    )

    print(f"[FullLayerProfile] Warmup ({warmup})...")
    ref_out = None
    for _ in range(warmup):
        ref_out = cuda_decode_gemv.fused_decode_full_layer_softsync_half(
            workspace=runtime_workspace,
            **call_kwargs,
        )
    torch.cuda.synchronize()

    print(f"[FullLayerProfile] Profiling ({iters})...")
    all_profiles = []
    prof_out = None
    meta = None
    prof_hidden = None
    for _ in range(iters):
        prof_hidden, prof_out, profile, meta = cuda_decode_gemv.fused_decode_full_layer_softsync_profile_half(
            workspace=profile_workspace,
            **call_kwargs,
        )
        torch.cuda.synchronize()
        all_profiles.append(profile.to(torch.int64).cpu())

    assert ref_out is not None and prof_out is not None and meta is not None and prof_hidden is not None
    diff = (ref_out - prof_out).abs().float()
    oproj_input_ref = payload["o_proj_input"].reshape(-1)
    downproj_input_ref = payload["down_proj_input"].reshape(-1)
    runtime_oproj_input = runtime_workspace["qkv_out"][:2048]
    runtime_hidden = runtime_workspace["hidden_out"]

    oproj = F.linear(
        oproj_input_ref.unsqueeze(0),
        target_layer.o_proj_weight,
        target_layer.o_proj_bias,
    ).reshape(-1)
    residual = payload["layer_input"].reshape(-1) + oproj
    inv_rms = torch.rsqrt(residual.float().pow(2).mean() + float(target_layer.rms_norm_eps))
    post = residual.float() * inv_rms
    post = (post * target_layer.post_attn_layernorm_weight.float()).to(payload["layer_input"].dtype)
    gate_up = F.linear(post.unsqueeze(0), target_layer.gate_up_weight, None).reshape(-1)
    gate, up = gate_up.split(6144)
    hidden_manual = (F.silu(gate.float()) * up.float()).to(payload["layer_input"].dtype)
    out_manual = (
        F.linear(hidden_manual.unsqueeze(0), target_layer.down_proj_weight, target_layer.down_proj_bias).reshape(-1)
        + residual
    ).to(payload["layer_input"].dtype)

    return {
        "profiles": all_profiles,
        "meta": meta,
        "validation": {
            "profile_vs_nonprofile_max_abs_diff": float(diff.max().item()),
            "profile_vs_nonprofile_mean_abs_diff": float(diff.mean().item()),
            "oproj_input_vs_capture_max_abs_diff": float((runtime_oproj_input - oproj_input_ref).abs().max().item()),
            "oproj_input_vs_capture_mean_abs_diff": float((runtime_oproj_input - oproj_input_ref).abs().float().mean().item()),
            "hidden_vs_capture_max_abs_diff": float((runtime_hidden - downproj_input_ref).abs().max().item()),
            "hidden_vs_capture_mean_abs_diff": float((runtime_hidden - downproj_input_ref).abs().float().mean().item()),
            "hidden_vs_manual_max_abs_diff": float((runtime_hidden - hidden_manual).abs().max().item()),
            "hidden_vs_manual_mean_abs_diff": float((runtime_hidden - hidden_manual).abs().float().mean().item()),
            "out_vs_manual_max_abs_diff": float((ref_out.reshape(-1) - out_manual).abs().max().item()),
            "out_vs_manual_mean_abs_diff": float((ref_out.reshape(-1) - out_manual).abs().float().mean().item()),
        },
    }


def _effective_ghz(ts: torch.Tensor) -> float:
    gt_span = ts[:, S_GT_END] - ts[:, S_GT_START]
    clk_span = ts[:, S_KERNEL_DONE] - ts[:, S_KERNEL_START]
    valid = gt_span > 0
    if valid.any():
        return float((clk_span[valid] / gt_span[valid]).median())
    return 1.41


def extract_block_data(profile_tensor: torch.Tensor):
    ts = profile_tensor.to(torch.float64)
    ghz = _effective_ghz(ts)

    def c2us(v):
        return float(v) / ghz / 1000.0

    blocks = []
    for block_idx in range(ts.shape[0]):
        t = ts[block_idx]
        phase_values = {
            "input_rmsnorm": c2us(t[S_INPUT_RMSNORM_DONE] - t[S_KERNEL_START]),
            "qkv": c2us(t[S_QKV_DONE] - t[S_INPUT_RMSNORM_DONE]),
            "sync1": c2us(t[S_SYNC1_DONE] - t[S_SYNC1_START]),
            "qkvcache": c2us(t[S_QKVCACHE_DONE] - t[S_SYNC1_DONE]),
            "sync2": c2us(t[S_SYNC2_DONE] - t[S_SYNC2_START]),
            "attention": c2us(t[S_ATTENTION_DONE] - t[S_SYNC2_DONE]),
            "sync3": c2us(t[S_SYNC3_DONE] - t[S_SYNC3_START]),
            "oproj": c2us(t[S_OPROJ_DONE] - t[S_SYNC3_DONE]),
            "sync4": c2us(t[S_SYNC4_DONE] - t[S_SYNC4_START]),
            "post_rmsnorm": c2us(t[S_POST_RMSNORM_DONE] - t[S_SYNC4_DONE]),
            "gateup_silu": c2us(t[S_GATEUP_SILU_DONE] - t[S_POST_RMSNORM_DONE]),
            "sync5": c2us(t[S_SYNC5_DONE] - t[S_SYNC5_START]),
            "downproj": c2us(t[S_DOWNPROJ_DONE] - t[S_SYNC5_DONE]),
        }
        blocks.append(
            {
                "block": block_idx,
                "sm_id": int(t[S_SM_ID].item()),
                "attn_prefetch_lines": int(t[S_ATTN_PREFETCH_LINES].item()),
                "attn_prefetch_mb": round(int(t[S_ATTN_PREFETCH_LINES].item()) * 128.0 / 1e6, 3),
                "total": c2us(t[S_KERNEL_DONE] - t[S_KERNEL_START]),
                **{k: round(v, 3) for k, v in phase_values.items()},
            }
        )
    return blocks, ghz


def compute_summary(
    all_profiles: list[torch.Tensor],
    valid_len: int,
    grid_blocks: int,
    attn_active_blocks: int | None = None,
    o_proj_active_blocks: int | None = None,
):
    attn_blocks = attn_active_blocks if attn_active_blocks is not None else min(grid_blocks, 16)
    oproj_blocks = o_proj_active_blocks if o_proj_active_blocks is not None else grid_blocks
    summaries = {
        "input_rmsnorm": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": 0.0},
        "qkv": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": (4096 * 2048 * 2) / 1e6},
        "sync1": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": 0.0},
        "qkvcache": {"wall_us": [], "active_blocks": 8, "data_mb": 0.0},
        "sync2": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": 0.0},
        "attention": {"wall_us": [], "active_blocks": attn_blocks, "data_mb": 0.0},
        "sync3": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": 0.0},
        "oproj": {"wall_us": [], "active_blocks": oproj_blocks, "data_mb": (2048 * 2048 * 2) / 1e6},
        "sync4": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": 0.0},
        "post_rmsnorm": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": 0.0},
        "gateup_silu": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": (12288 * 2048 * 2) / 1e6},
        "sync5": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": 0.0},
        "downproj": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": (2048 * 6144 * 2) / 1e6},
        "total": {"wall_us": [], "active_blocks": grid_blocks, "data_mb": ((4096 + 2048 + 12288) * 2048 * 2 + (2048 * 6144 * 2)) / 1e6},
    }

    for prof in all_profiles:
        ts = prof.to(torch.float64)
        ghz = _effective_ghz(ts)

        def max_c2us(delta):
            return float(delta.max()) / ghz / 1000.0

        summaries["input_rmsnorm"]["wall_us"].append(max_c2us(ts[:, S_INPUT_RMSNORM_DONE] - ts[:, S_KERNEL_START]))
        summaries["qkv"]["wall_us"].append(max_c2us(ts[:, S_QKV_DONE] - ts[:, S_INPUT_RMSNORM_DONE]))
        summaries["sync1"]["wall_us"].append(max_c2us(ts[:, S_SYNC1_DONE] - ts[:, S_SYNC1_START]))
        summaries["qkvcache"]["wall_us"].append(max_c2us(ts[:, S_QKVCACHE_DONE] - ts[:, S_SYNC1_DONE]))
        summaries["sync2"]["wall_us"].append(max_c2us(ts[:, S_SYNC2_DONE] - ts[:, S_SYNC2_START]))
        summaries["attention"]["wall_us"].append(max_c2us(ts[:, S_ATTENTION_DONE] - ts[:, S_SYNC2_DONE]))
        summaries["sync3"]["wall_us"].append(max_c2us(ts[:, S_SYNC3_DONE] - ts[:, S_SYNC3_START]))
        summaries["oproj"]["wall_us"].append(max_c2us(ts[:, S_OPROJ_DONE] - ts[:, S_SYNC3_DONE]))
        summaries["sync4"]["wall_us"].append(max_c2us(ts[:, S_SYNC4_DONE] - ts[:, S_SYNC4_START]))
        summaries["post_rmsnorm"]["wall_us"].append(max_c2us(ts[:, S_POST_RMSNORM_DONE] - ts[:, S_SYNC4_DONE]))
        summaries["gateup_silu"]["wall_us"].append(max_c2us(ts[:, S_GATEUP_SILU_DONE] - ts[:, S_POST_RMSNORM_DONE]))
        summaries["sync5"]["wall_us"].append(max_c2us(ts[:, S_SYNC5_DONE] - ts[:, S_SYNC5_START]))
        summaries["downproj"]["wall_us"].append(max_c2us(ts[:, S_DOWNPROJ_DONE] - ts[:, S_SYNC5_DONE]))
        summaries["total"]["wall_us"].append(max_c2us(ts[:, S_KERNEL_DONE] - ts[:, S_KERNEL_START]))

    prefetch_line_totals = [
        int(prof[:, S_ATTN_PREFETCH_LINES].sum().item())
        for prof in all_profiles
    ]
    prefetch_line_totals_sorted = sorted(prefetch_line_totals)
    median_prefetch_lines = prefetch_line_totals_sorted[len(prefetch_line_totals_sorted) // 2]
    median_prefetch_mb = (median_prefetch_lines * 128.0) / 1e6
    oproj_weight_mb = summaries["oproj"]["data_mb"]
    median_prefetch_pct = (
        (median_prefetch_mb / oproj_weight_mb) * 100.0
        if oproj_weight_mb > 0.0
        else None
    )

    phase_summary = []
    for key, _, _color in PHASES:
        walls = sorted(summaries[key]["wall_us"])
        med_wall = walls[len(walls) // 2]
        phase_summary.append(
            {
                "phase": key,
                "active_blocks": summaries[key]["active_blocks"],
                "wall_us": round(med_wall, 3),
                "data_mb": round(summaries[key]["data_mb"], 3),
                "equiv_bandwidth_gbps": (
                    round(_equiv_bandwidth_gbps(summaries[key]["data_mb"], med_wall), 3)
                    if _equiv_bandwidth_gbps(summaries[key]["data_mb"], med_wall) is not None
                    else None
                ),
            }
        )
    total_walls = sorted(summaries["total"]["wall_us"])
    phase_summary.append(
        {
            "phase": "total",
            "active_blocks": summaries["total"]["active_blocks"],
            "wall_us": round(total_walls[len(total_walls) // 2], 3),
            "data_mb": round(summaries["total"]["data_mb"], 3),
            "equiv_bandwidth_gbps": (
                round(
                    _equiv_bandwidth_gbps(
                        summaries["total"]["data_mb"],
                        total_walls[len(total_walls) // 2],
                    ),
                    3,
                )
                if _equiv_bandwidth_gbps(
                    summaries["total"]["data_mb"],
                    total_walls[len(total_walls) // 2],
                ) is not None
                else None
            ),
        }
    )
    return {
        "valid_len": int(valid_len),
        "phase_summary": phase_summary,
        "attn_prefetch_lines": int(median_prefetch_lines),
        "attn_prefetch_mb": round(median_prefetch_mb, 3),
        "attn_prefetch_vs_oproj_pct": (
            round(median_prefetch_pct, 3) if median_prefetch_pct is not None else None
        ),
    }


def _block_role(block_idx: int, metadata: dict) -> str:
    attn_active_blocks = int(metadata.get("attn_active_blocks", min(int(metadata["grid_blocks"]), 16)))
    num_q_heads = int(metadata.get("num_q_heads", 0))
    attn_qhead_splitkv = bool(metadata.get("attn_qhead_splitkv", False))
    attn_split_k = int(metadata.get("attn_split_k", 1))
    if block_idx < attn_active_blocks:
        if attn_qhead_splitkv and attn_split_k > 1 and num_q_heads > 0:
            q_head_idx = block_idx // attn_split_k
            split_idx = block_idx % attn_split_k
            return f"attn_q{q_head_idx:02d}_s{split_idx}"
        if num_q_heads > 0:
            return f"attn_q{block_idx:02d}"
        return "attn"
    return "idle"


def _block_role_group(block_idx: int, metadata: dict) -> str:
    attn_active_blocks = int(metadata.get("attn_active_blocks", min(int(metadata["grid_blocks"]), 16)))
    attn_qhead_splitkv = bool(metadata.get("attn_qhead_splitkv", False))
    if block_idx < attn_active_blocks:
        return "attn_split" if attn_qhead_splitkv else "attn"
    return "idle"


def _role_summary(blocks: list[dict], metadata: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for block in blocks:
        role = _block_role_group(int(block["block"]), metadata)
        grouped.setdefault(role, []).append(block)

    summary = []
    for role, items in grouped.items():
        totals = sorted(float(item["total"]) for item in items)
        summary.append(
            {
                "role": role,
                "count": len(items),
                "median_total_us": totals[len(totals) // 2],
                "max_total_us": max(totals),
                "median_attention_us": sorted(float(item["attention"]) for item in items)[len(items) // 2],
                "median_sync3_us": sorted(float(item["sync3"]) for item in items)[len(items) // 2],
                "median_oproj_us": sorted(float(item["oproj"]) for item in items)[len(items) // 2],
            }
        )
    summary.sort(key=lambda row: row["max_total_us"], reverse=True)
    return summary


def generate_html(phase_summary: dict, block_data_runs: list[list[dict]], effective_ghz_runs: list[float], validation: dict, metadata: dict) -> str:
    run_totals = [(max(block["total"] for block in blocks), idx) for idx, blocks in enumerate(block_data_runs)]
    run_totals.sort()
    representative_idx = run_totals[len(run_totals) // 2][1]
    representative_blocks = block_data_runs[representative_idx]
    representative_total = run_totals[len(run_totals) // 2][0]
    max_total = max(block["total"] for block in representative_blocks) if representative_blocks else 1.0
    effective_ghz = effective_ghz_runs[representative_idx]
    role_summary = _role_summary(representative_blocks, metadata)
    phase_color = {phase: color for phase, _label, color in PHASES}
    phase_label = {phase: label for phase, label, _color in PHASES}

    summary_rows = []
    for row in phase_summary["phase_summary"]:
        bandwidth = row.get("equiv_bandwidth_gbps")
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(row['phase'])}</td>"
            f"<td>{row['active_blocks']}</td>"
            f"<td>{row['wall_us']:.3f}</td>"
            f"<td>{row['data_mb']:.3f}</td>"
            f"<td>{f'{bandwidth:.3f}' if bandwidth is not None else '--'}</td>"
            "</tr>"
        )

    role_rows = []
    for row in role_summary:
        role_rows.append(
            "<tr>"
            f"<td>{html.escape(row['role'])}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{row['median_total_us']:.3f}</td>"
            f"<td>{row['max_total_us']:.3f}</td>"
            f"<td>{row['median_attention_us']:.3f}</td>"
            f"<td>{row['median_sync3_us']:.3f}</td>"
            f"<td>{row['median_oproj_us']:.3f}</td>"
            "</tr>"
        )

    timeline_rows = []
    for block in representative_blocks:
        role = _block_role(int(block["block"]), metadata)
        segments = []
        offset = 0.0
        for phase, _label, _color in PHASES:
            dur = float(block[phase])
            if dur <= 0:
                continue
            left = (offset / max_total) * 100.0
            width = max(0.15, (dur / max_total) * 100.0)
            segments.append(
                f'<div class="seg" title="{html.escape(phase_label[phase])}: {dur:.3f} us" '
                f'style="left:{left:.4f}%;width:{width:.4f}%;background:{phase_color[phase]};"></div>'
            )
            offset += dur
        timeline_rows.append(
            "<div class=\"row\">"
            f"<div class=\"label\">B{block['block']:02d} / SM{block['sm_id']:02d} / {html.escape(role)}</div>"
            f"<div class=\"bar\">{''.join(segments)}</div>"
            f"<div class=\"tail\">{block['total']:.3f} us</div>"
            "</div>"
        )

    detail_header = "".join(f"<th>{html.escape(phase)}</th>" for phase, _label, _color in PHASES)
    detail_rows = []
    for block in representative_blocks:
        role = _block_role(int(block["block"]), metadata)
        detail_rows.append(
            "<tr>"
            f"<td>{block['block']}</td>"
            f"<td>{block['sm_id']}</td>"
            f"<td>{html.escape(role)}</td>"
            + "".join(f"<td>{float(block[phase]):.3f}</td>" for phase, _label, _color in PHASES)
            + f"<td>{block['total']:.3f}</td>"
            "</tr>"
        )

    legend = "".join(
        f'<div class="legend-item"><span class="legend-color" style="background:{color};"></span>{html.escape(label)}</div>'
        for _phase, label, color in PHASES
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Full-Layer Softsync Profile</title>
<style>
body {{
    margin: 0;
    padding: 24px;
    font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
    background: #0f172a;
    color: #e2e8f0;
}}
h1 {{ margin: 0 0 8px 0; font-size: 20px; color: #f8fafc; }}
h2 {{ margin: 24px 0 12px 0; font-size: 15px; color: #cbd5e1; }}
.subtitle {{ color: #94a3b8; margin-bottom: 16px; }}
.meta {{ display: grid; grid-template-columns: repeat(3, max-content); gap: 8px 18px; color: #cbd5e1; margin-bottom: 18px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 10px 16px; margin: 10px 0 16px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; color: #cbd5e1; }}
.legend-color {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
table {{ border-collapse: collapse; width: 100%; max-width: 1100px; }}
th, td {{ border-bottom: 1px solid #243041; padding: 6px 10px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
.row {{ display: grid; grid-template-columns: 120px 1fr 90px; align-items: center; gap: 10px; margin-bottom: 3px; }}
.label {{ color: #94a3b8; font-size: 12px; text-align: right; }}
.bar {{ position: relative; height: 16px; background: #111827; border: 1px solid #243041; border-radius: 4px; }}
.seg {{ position: absolute; top: 0; bottom: 0; }}
.tail {{ color: #cbd5e1; font-size: 12px; }}
.note {{ color: #94a3b8; margin-top: 10px; }}
</style>
</head>
<body>
<h1>Full-Layer Softsync Decode Profile</h1>
<div class="subtitle">Representative run #{representative_idx} | clock64 calibrated by globaltimer</div>
<div class="meta">
  <div>sample={metadata["sample_idx"]}</div>
  <div>layer={metadata["layer_idx"]}</div>
  <div>decode_step={metadata["decode_step"]}</div>
  <div>question_id={metadata["question_id"]}</div>
  <div>valid_len={metadata["valid_len"]}</div>
  <div>launch_blocks={metadata["launch_blocks"]}</div>
  <div>grid_blocks={metadata["grid_blocks"]}</div>
  <div>num_q_heads={metadata.get("num_q_heads", "NA")}</div>
  <div>attn_qhead_splitkv={int(metadata.get("attn_qhead_splitkv", False))}</div>
  <div>attn_split_k={metadata.get("attn_split_k", 1)}</div>
  <div>attn_active_blocks={metadata.get("attn_active_blocks", "NA")}</div>
  <div>idle_blocks={metadata.get("idle_blocks", "NA")}</div>
  <div>partial_attn_flags={int(metadata.get("partial_attn_flags", False))}</div>
  <div>o_proj_idle_only={int(metadata.get("o_proj_idle_only", False))}</div>
  <div>o_proj_active_blocks={metadata.get("o_proj_active_blocks", metadata["grid_blocks"])}</div>
  <div>rows_per_wave={metadata["rows_per_wave"]}</div>
  <div>downproj_row_passes={metadata["downproj_row_passes"]}</div>
  <div>rep_total={representative_total:.3f} us</div>
  <div>rep_ghz={effective_ghz:.4f}</div>
  <div>attn_prefetch_lines={phase_summary.get("attn_prefetch_lines", 0)}</div>
  <div>attn_prefetch_mb={phase_summary.get("attn_prefetch_mb", 0.0):.3f}</div>
  <div>attn_prefetch_vs_oproj={phase_summary.get("attn_prefetch_vs_oproj_pct", 0.0):.2f}%</div>
  <div>profile_vs_nonprofile_max_abs_diff={validation["profile_vs_nonprofile_max_abs_diff"]:.6f}</div>
  <div>profile_vs_nonprofile_mean_abs_diff={validation["profile_vs_nonprofile_mean_abs_diff"]:.6f}</div>
  <div>oproj_input_vs_capture_mean_abs_diff={validation["oproj_input_vs_capture_mean_abs_diff"]:.6f}</div>
  <div>hidden_vs_capture_mean_abs_diff={validation["hidden_vs_capture_mean_abs_diff"]:.6f}</div>
  <div>out_vs_manual_mean_abs_diff={validation["out_vs_manual_mean_abs_diff"]:.6f}</div>
</div>

<h2>Phase Summary</h2>
<table>
  <thead>
    <tr><th>phase</th><th>active_blocks</th><th>median_wall_us</th><th>weight_data_mb</th><th>equiv_weight_GBps</th></tr>
  </thead>
  <tbody>
    {''.join(summary_rows)}
  </tbody>
</table>
<div class="note">`equiv_weight_GBps = weight_data_mb / median_wall_us * 1000`，按本表统计的权重字节量折算，仅作相对比较，不等同于真实 DRAM 带宽。</div>

<h2>Representative Role Summary</h2>
<table>
  <thead>
    <tr><th>role</th><th>count</th><th>median_total_us</th><th>max_total_us</th><th>median_attention_us</th><th>median_sync3_us</th><th>median_oproj_us</th></tr>
  </thead>
  <tbody>
    {''.join(role_rows)}
  </tbody>
</table>

<h2>Representative Timeline</h2>
<div class="legend">{legend}</div>
{''.join(timeline_rows)}
<div class="note">Timeline shows per-block phase durations for the median-total run. Sync bars are barrier wall time, not launch overhead.</div>

<h2>Representative Block Table</h2>
<table>
  <thead>
    <tr><th>block</th><th>sm_id</th><th>role</th>{detail_header}<th>total</th></tr>
  </thead>
  <tbody>
    {''.join(detail_rows)}
  </tbody>
</table>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Profile full-layer softsync decode kernel on real payloads")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=NUM_ITERS)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--launch-blocks", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-html", type=str, default=None)
    parser.add_argument("--output-pt", type=str, default=None)
    args = parser.parse_args()

    vlm, sample, _inputs, payload, capture_meta, target_layer = _capture_payload(args)
    launch_blocks = args.launch_blocks
    if launch_blocks is None:
        launch_blocks = int(os.environ.get("CUDA_FUSED_FULL_LAYER_SOFTSYNC_BLOCKS", "0"))

    result = run_profile(
        payload=payload,
        target_layer=target_layer,
        launch_blocks=launch_blocks,
        warmup=args.warmup,
        iters=args.iters,
    )

    all_profiles = result["profiles"]
    block_data_runs = []
    effective_ghz_runs = []
    for prof in all_profiles:
        blocks, ghz = extract_block_data(prof)
        block_data_runs.append(blocks)
        effective_ghz_runs.append(ghz)

    grid_blocks = result["meta"]["grid_blocks"]
    valid_len = int(payload["cache_position"].item()) + 1
    phase_summary = compute_summary(
        all_profiles,
        valid_len=valid_len,
        grid_blocks=grid_blocks,
        attn_active_blocks=result["meta"].get("attn_active_blocks"),
        o_proj_active_blocks=result["meta"].get("o_proj_active_blocks"),
    )

    stem = (
        f"full_layer_profile_sample{args.sample_idx}_layer{args.layer_idx}_"
        f"step{args.decode_step}_blocks{launch_blocks}"
    )
    output_json = Path(args.output_json) if args.output_json else Path("results") / f"{stem}.json"
    output_html = Path(args.output_html) if args.output_html else Path("results") / f"{stem}.html"
    output_pt = Path(args.output_pt) if args.output_pt else Path("results") / f"{stem}.pt"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "layer_idx": int(args.layer_idx),
        "decode_step": int(args.decode_step),
        "captured_decode_step": int(capture_meta["captured_decode_step"]),
        "prompt_length": int(capture_meta["prompt_length"]),
        "max_cache_len": int(capture_meta["max_cache_len"]),
        "valid_len": valid_len,
        "launch_blocks": int(launch_blocks),
        "grid_blocks": int(grid_blocks),
        "num_q_heads": int(result["meta"].get("num_q_heads", 0)),
        "attn_qhead_splitkv": bool(result["meta"].get("attn_qhead_splitkv", False)),
        "attn_split_k": int(result["meta"].get("attn_split_k", 1)),
        "attn_active_blocks": int(result["meta"].get("attn_active_blocks", min(grid_blocks, 16))),
        "idle_blocks": int(result["meta"].get("idle_blocks", 0)),
        "partial_attn_flags": bool(result["meta"].get("partial_attn_flags", False)),
        "o_proj_idle_only": bool(result["meta"].get("o_proj_idle_only", False)),
        "o_proj_active_blocks": int(result["meta"].get("o_proj_active_blocks", grid_blocks)),
        "rows_per_wave": int(result["meta"]["rows_per_wave"]),
        "downproj_row_passes": int(result["meta"]["downproj_row_passes"]),
    }
    payload_json = {
        "metadata": metadata,
        "validation": result["validation"],
        "phase_summary": phase_summary,
        "effective_ghz_runs": [round(v, 6) for v in effective_ghz_runs],
        "slot_names": result["meta"]["slot_names"],
    }
    output_json.write_text(json.dumps(payload_json, indent=2), encoding="utf-8")
    torch.save(torch.stack(all_profiles, dim=0), output_pt)
    output_html.write_text(
        generate_html(
            phase_summary=phase_summary,
            block_data_runs=block_data_runs,
            effective_ghz_runs=effective_ghz_runs,
            validation=result["validation"],
            metadata=metadata,
        ),
        encoding="utf-8",
    )

    print(f"[FullLayerProfile] JSON: {output_json}")
    print(f"[FullLayerProfile] PT:   {output_pt}")
    print(f"[FullLayerProfile] HTML: {output_html}")


if __name__ == "__main__":
    main()
