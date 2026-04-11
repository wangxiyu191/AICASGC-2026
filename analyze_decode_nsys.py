#!/usr/bin/env python3
"""Analyze decode-phase kernel timing from nsys SQLite export.

Reads an nsys-exported SQLite database and produces a detailed breakdown of
decode-phase GPU kernel timing, grouped by logical operation (input_rmsnorm,
qkv_proj, qk_norm_rope, attention, o_proj, post_attn_mlp, lm_head, etc.).

Also computes theoretical roofline limits for each operation on A800 80GB PCIe.

Usage:
    ./venv/bin/python analyze_decode_nsys.py traces/nsys_decode/report.sqlite
"""
import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# A800 80GB PCIe specs
A800_HBM_BW_GBS = 2039.0  # GB/s peak HBM bandwidth
A800_SM_COUNT = 108
A800_FP16_TFLOPS = 312.0  # FP16 Tensor Core TFLOPS

# Qwen3-VL-2B model dimensions
HIDDEN_SIZE = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 6144
VOCAB_SIZE = 151936
NUM_LAYERS = 28


@dataclass
class KernelInfo:
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    grid: tuple = (0, 0, 0)
    block: tuple = (0, 0, 0)


def load_kernels(db_path: str) -> list[KernelInfo]:
    """Load all GPU kernels from nsys SQLite export, sorted by start time."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute('''
            SELECT k.start, k.end, (k.end - k.start) as dur_ns,
                   COALESCE(dk.value, sk.value, 'unknown') as name,
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds dk ON k.demangledName = dk.id
            LEFT JOIN StringIds sk ON k.shortName = sk.id
            ORDER BY k.start
        ''').fetchall()
    finally:
        conn.close()

    kernels = []
    for row in rows:
        kernels.append(KernelInfo(
            name=row[3],
            start_ns=row[0],
            end_ns=row[1],
            duration_ns=row[2],
            grid=(row[4], row[5], row[6]),
            block=(row[7], row[8], row[9]),
        ))
    return kernels


def load_nvtx_ranges(db_path: str) -> list[dict]:
    """Load NVTX range events from nsys SQLite export."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute('''
            SELECT n.start, n.end, s.value as name
            FROM NVTX_EVENTS n
            JOIN StringIds s ON n.textId = s.id
            WHERE n.eventType = 59 OR n.eventType = 60
            ORDER BY n.start
        ''').fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    return [{"start": r[0], "end": r[1], "name": r[2]} for r in rows]


def classify_kernel(name: str) -> str:
    """Classify a kernel name into a logical decode operation category."""
    name_lower = name.lower()

    # CUDAGraph replay/launch overhead
    if 'cudagraphlaunch' in name_lower or 'graphkernellaunch' in name_lower:
        return 'graph_overhead'

    # lm_head (vocab projection)
    if 'gemv2t' in name_lower and '151936' in name:
        return 'lm_head'
    if 'ampere_fp16' in name_lower and '151936' in name:
        return 'lm_head'
    # Large grid gemv that looks like lm_head
    if ('gemv' in name_lower or 'gemm' in name_lower) and 'ampere' in name_lower:
        # Check if it's likely lm_head based on grid size
        pass

    # Fused post-attn MLP persistent kernels
    if 'fused_post_attn_mlp_persistent' in name_lower or 'fused_o_proj_post_attn_mlp_persistent' in name_lower:
        return 'fused_post_attn_mlp'
    if 'fused_gateup_silu_downproj' in name_lower:
        return 'fused_mlp'

    # CUDA nosplit GEMV
    if 'nosplit_matvec' in name_lower:
        # Distinguish by which weight matrix based on kernel variant name
        if 'k6144' in name_lower or 'down_proj' in name_lower:
            return 'down_proj_gemv'
        return 'qkv_or_oproj_gemv'

    # Triton kernels
    if '_rms_norm_fwd' in name:
        return 'rmsnorm'
    if '_fused_add_rms_norm' in name:
        return 'add_rmsnorm'
    if '_fused_rms_norm_rope' in name or '_fused_qk_rms_norm_rope' in name:
        return 'qk_norm_rope'
    if '_static_cache_single_token' in name:
        return 'kv_cache_update'
    if '_small_decode_attention' in name:
        return 'attention'
    if '_decode_gemv_fwd' in name:
        return 'triton_gemv'
    if '_fused_silu_mul' in name:
        return 'silu_mul'
    if 'fused_add_rmsnorm_gateup' in name_lower:
        return 'add_rmsnorm_gateup'

    # cuBLAS GEMM/GEMV
    if 'ampere' in name_lower and ('gemm' in name_lower or 'gemv' in name_lower or 'wmma' in name_lower):
        return 'cublas_gemm'
    if 'cutlass' in name_lower:
        return 'cublas_gemm'

    # Argmax
    if 'argmax' in name_lower or 'arg_max' in name_lower:
        return 'argmax'

    # Elementwise
    if 'elementwise' in name_lower or 'vectorized' in name_lower or 'Pointwise' in name:
        return 'elementwise'

    # Copy
    if 'copy' in name_lower or 'memcpy' in name_lower or 'memset' in name_lower:
        return 'memcpy'

    # Embedding
    if 'embedding' in name_lower or 'index_select' in name_lower:
        return 'embedding'

    return 'other'


def find_decode_region(kernels: list[KernelInfo]) -> tuple[int, int]:
    """Find the start and end indices of the decode phase in the kernel list.

    Heuristic: decode starts after the last large GEMM (prefill) and runs
    until the end. We look for a sequence of repeated small kernels (the
    CUDAGraph decode loop).
    """
    # Look for CUDAGraph replay pattern: many small kernels in quick succession
    # Or just take the entire kernel list if it's a decode-only trace
    return 0, len(kernels)


def compute_theoretical_times():
    """Compute theoretical minimum time for each decode operation on A800.

    All operations are memory-bandwidth-bound for batch=1 decode.
    Time = bytes_transferred / peak_bandwidth
    """
    bw = A800_HBM_BW_GBS * 1e9  # bytes/s
    fp16_bytes = 2

    ops = {}

    # Per-layer operations (× 28 layers)

    # 1. input_rmsnorm: read hidden[2048] + weight[2048], write output[2048]
    #    = (2048 + 2048 + 2048) * 2 bytes = 12288 bytes
    rmsnorm_bytes = (HIDDEN_SIZE * 3) * fp16_bytes
    ops['input_rmsnorm'] = {
        'bytes_per_layer': rmsnorm_bytes,
        'bytes_total': rmsnorm_bytes * NUM_LAYERS,
        'time_us_per_layer': rmsnorm_bytes / bw * 1e6,
        'time_us_total': rmsnorm_bytes * NUM_LAYERS / bw * 1e6,
        'note': f'read x[{HIDDEN_SIZE}] + w[{HIDDEN_SIZE}], write out[{HIDDEN_SIZE}]',
    }

    # 2. QKV projection: weight[4096, 2048] * x[2048] → out[4096]
    #    Fused QKV: (num_heads*head_dim + 2*num_kv_heads*head_dim) = 2048+1024+1024=4096
    qkv_out_size = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
    qkv_weight_bytes = qkv_out_size * HIDDEN_SIZE * fp16_bytes
    qkv_input_bytes = HIDDEN_SIZE * fp16_bytes
    qkv_output_bytes = qkv_out_size * fp16_bytes
    qkv_total = qkv_weight_bytes + qkv_input_bytes + qkv_output_bytes
    ops['qkv_proj'] = {
        'bytes_per_layer': qkv_total,
        'bytes_total': qkv_total * NUM_LAYERS,
        'time_us_per_layer': qkv_total / bw * 1e6,
        'time_us_total': qkv_total * NUM_LAYERS / bw * 1e6,
        'note': f'W[{qkv_out_size}×{HIDDEN_SIZE}] * x[{HIDDEN_SIZE}] → [{qkv_out_size}]',
    }

    # 3. QK-norm + RoPE: read q[2048]+k[1024]+w_q[128]+w_k[128]+cos[128]+sin[128],
    #    write q_out[2048]+k_out[1024]
    qk_norm_rope_read = (NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM + HEAD_DIM * 2 + HEAD_DIM * 2) * fp16_bytes
    qk_norm_rope_write = (NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM) * fp16_bytes
    qk_norm_rope_total = qk_norm_rope_read + qk_norm_rope_write
    ops['qk_norm_rope'] = {
        'bytes_per_layer': qk_norm_rope_total,
        'bytes_total': qk_norm_rope_total * NUM_LAYERS,
        'time_us_per_layer': qk_norm_rope_total / bw * 1e6,
        'time_us_total': qk_norm_rope_total * NUM_LAYERS / bw * 1e6,
        'note': f'q[{NUM_HEADS}×{HEAD_DIM}]+k[{NUM_KV_HEADS}×{HEAD_DIM}]+norms+rope',
    }

    # 4. KV cache update: write k[8×128]+v[8×128] to cache
    kv_write_bytes = 2 * NUM_KV_HEADS * HEAD_DIM * fp16_bytes
    ops['kv_cache_update'] = {
        'bytes_per_layer': kv_write_bytes,
        'bytes_total': kv_write_bytes * NUM_LAYERS,
        'time_us_per_layer': kv_write_bytes / bw * 1e6,
        'time_us_total': kv_write_bytes * NUM_LAYERS / bw * 1e6,
        'note': f'write k[{NUM_KV_HEADS}×{HEAD_DIM}]+v[{NUM_KV_HEADS}×{HEAD_DIM}] to cache',
    }

    # 5. Attention: for seq_len S, read q[16×128] + K[8×S×128] + V[8×S×128], write out[16×128]
    #    Typical S ~ 200-600 for this dataset
    for seq_len in [256, 512, 1024]:
        attn_q_bytes = NUM_HEADS * HEAD_DIM * fp16_bytes
        attn_kv_bytes = 2 * NUM_KV_HEADS * seq_len * HEAD_DIM * fp16_bytes
        attn_out_bytes = NUM_HEADS * HEAD_DIM * fp16_bytes
        attn_total = attn_q_bytes + attn_kv_bytes + attn_out_bytes
        ops[f'attention_S{seq_len}'] = {
            'bytes_per_layer': attn_total,
            'bytes_total': attn_total * NUM_LAYERS,
            'time_us_per_layer': attn_total / bw * 1e6,
            'time_us_total': attn_total * NUM_LAYERS / bw * 1e6,
            'note': f'q[{NUM_HEADS}×{HEAD_DIM}] + KV[{NUM_KV_HEADS}×{seq_len}×{HEAD_DIM}] → out[{NUM_HEADS}×{HEAD_DIM}]',
        }

    # 6. o_proj: weight[2048, 2048] * x[2048] → out[2048]
    oproj_weight_bytes = HIDDEN_SIZE * HIDDEN_SIZE * fp16_bytes
    oproj_input_bytes = HIDDEN_SIZE * fp16_bytes
    oproj_output_bytes = HIDDEN_SIZE * fp16_bytes
    oproj_total = oproj_weight_bytes + oproj_input_bytes + oproj_output_bytes
    ops['o_proj'] = {
        'bytes_per_layer': oproj_total,
        'bytes_total': oproj_total * NUM_LAYERS,
        'time_us_per_layer': oproj_total / bw * 1e6,
        'time_us_total': oproj_total * NUM_LAYERS / bw * 1e6,
        'note': f'W[{HIDDEN_SIZE}×{HIDDEN_SIZE}] * x[{HIDDEN_SIZE}] → [{HIDDEN_SIZE}]',
    }

    # 7. post_attn add+rmsnorm: same as input_rmsnorm but with residual add
    add_rmsnorm_bytes = (HIDDEN_SIZE * 4) * fp16_bytes  # read residual+x+w, write sum+norm
    ops['post_attn_add_rmsnorm'] = {
        'bytes_per_layer': add_rmsnorm_bytes,
        'bytes_total': add_rmsnorm_bytes * NUM_LAYERS,
        'time_us_per_layer': add_rmsnorm_bytes / bw * 1e6,
        'time_us_total': add_rmsnorm_bytes * NUM_LAYERS / bw * 1e6,
        'note': f'read res[{HIDDEN_SIZE}]+x[{HIDDEN_SIZE}]+w[{HIDDEN_SIZE}], write sum+norm',
    }

    # 8. gate_up projection: weight[12288, 2048] * x[2048] → out[12288]
    gateup_weight_bytes = INTERMEDIATE_SIZE * 2 * HIDDEN_SIZE * fp16_bytes
    gateup_input_bytes = HIDDEN_SIZE * fp16_bytes
    gateup_output_bytes = INTERMEDIATE_SIZE * 2 * fp16_bytes
    gateup_total = gateup_weight_bytes + gateup_input_bytes + gateup_output_bytes
    ops['gate_up_proj'] = {
        'bytes_per_layer': gateup_total,
        'bytes_total': gateup_total * NUM_LAYERS,
        'time_us_per_layer': gateup_total / bw * 1e6,
        'time_us_total': gateup_total * NUM_LAYERS / bw * 1e6,
        'note': f'W[{2*INTERMEDIATE_SIZE}×{HIDDEN_SIZE}] * x[{HIDDEN_SIZE}] → [{2*INTERMEDIATE_SIZE}]',
    }

    # 9. SiLU * mul: read gate_up[12288], write out[6144]
    silu_read = INTERMEDIATE_SIZE * 2 * fp16_bytes
    silu_write = INTERMEDIATE_SIZE * fp16_bytes
    silu_total = silu_read + silu_write
    ops['silu_mul'] = {
        'bytes_per_layer': silu_total,
        'bytes_total': silu_total * NUM_LAYERS,
        'time_us_per_layer': silu_total / bw * 1e6,
        'time_us_total': silu_total * NUM_LAYERS / bw * 1e6,
        'note': f'read gate_up[{2*INTERMEDIATE_SIZE}], write [{INTERMEDIATE_SIZE}]',
    }

    # 10. down_proj: weight[2048, 6144] * x[6144] → out[2048]
    downproj_weight_bytes = HIDDEN_SIZE * INTERMEDIATE_SIZE * fp16_bytes
    downproj_input_bytes = INTERMEDIATE_SIZE * fp16_bytes
    downproj_output_bytes = HIDDEN_SIZE * fp16_bytes
    downproj_total = downproj_weight_bytes + downproj_input_bytes + downproj_output_bytes
    ops['down_proj'] = {
        'bytes_per_layer': downproj_total,
        'bytes_total': downproj_total * NUM_LAYERS,
        'time_us_per_layer': downproj_total / bw * 1e6,
        'time_us_total': downproj_total * NUM_LAYERS / bw * 1e6,
        'note': f'W[{HIDDEN_SIZE}×{INTERMEDIATE_SIZE}] * x[{INTERMEDIATE_SIZE}] → [{HIDDEN_SIZE}]',
    }

    # 11. residual_add: read a[2048]+b[2048], write out[2048]
    resadd_bytes = HIDDEN_SIZE * 3 * fp16_bytes
    ops['residual_add'] = {
        'bytes_per_layer': resadd_bytes,
        'bytes_total': resadd_bytes * NUM_LAYERS,
        'time_us_per_layer': resadd_bytes / bw * 1e6,
        'time_us_total': resadd_bytes * NUM_LAYERS / bw * 1e6,
        'note': f'read a[{HIDDEN_SIZE}]+b[{HIDDEN_SIZE}], write [{HIDDEN_SIZE}]',
    }

    # Fused operations that combine multiple steps

    # Fused post_attn MLP persistent:
    # add_rmsnorm + gate_up + silu_mul + down_proj + residual_add
    # But weight reads dominate: gate_up_weight + down_proj_weight
    fused_mlp_weight_bytes = (INTERMEDIATE_SIZE * 2 * HIDDEN_SIZE + HIDDEN_SIZE * INTERMEDIATE_SIZE) * fp16_bytes
    fused_mlp_act_bytes = (HIDDEN_SIZE * 5 + INTERMEDIATE_SIZE * 2) * fp16_bytes  # rough
    fused_mlp_total = fused_mlp_weight_bytes + fused_mlp_act_bytes
    ops['fused_post_attn_mlp'] = {
        'bytes_per_layer': fused_mlp_total,
        'bytes_total': fused_mlp_total * NUM_LAYERS,
        'time_us_per_layer': fused_mlp_total / bw * 1e6,
        'time_us_total': fused_mlp_total * NUM_LAYERS / bw * 1e6,
        'note': 'add_rmsnorm + gate_up + silu + down_proj + residual (weight-dominant)',
    }

    # Fused o_proj + post_attn MLP persistent:
    fused_oproj_mlp_weight_bytes = fused_mlp_weight_bytes + HIDDEN_SIZE * HIDDEN_SIZE * fp16_bytes
    fused_oproj_mlp_act_bytes = (HIDDEN_SIZE * 6 + INTERMEDIATE_SIZE * 2) * fp16_bytes
    fused_oproj_mlp_total = fused_oproj_mlp_weight_bytes + fused_oproj_mlp_act_bytes
    ops['fused_o_proj_post_attn_mlp'] = {
        'bytes_per_layer': fused_oproj_mlp_total,
        'bytes_total': fused_oproj_mlp_total * NUM_LAYERS,
        'time_us_per_layer': fused_oproj_mlp_total / bw * 1e6,
        'time_us_total': fused_oproj_mlp_total * NUM_LAYERS / bw * 1e6,
        'note': 'o_proj + add_rmsnorm + gate_up + silu + down_proj + residual',
    }

    # Per-token operations (once per decode step)

    # final_rmsnorm
    final_norm_bytes = HIDDEN_SIZE * 3 * fp16_bytes
    ops['final_rmsnorm'] = {
        'bytes_per_layer': final_norm_bytes,
        'bytes_total': final_norm_bytes,
        'time_us_per_layer': final_norm_bytes / bw * 1e6,
        'time_us_total': final_norm_bytes / bw * 1e6,
        'note': f'final layer norm [{HIDDEN_SIZE}]',
    }

    # lm_head: weight[151936, 2048] * x[2048] → out[151936]
    lm_weight_bytes = VOCAB_SIZE * HIDDEN_SIZE * fp16_bytes
    lm_input_bytes = HIDDEN_SIZE * fp16_bytes
    lm_output_bytes = VOCAB_SIZE * fp16_bytes  # written as fp16
    lm_total = lm_weight_bytes + lm_input_bytes + lm_output_bytes
    ops['lm_head'] = {
        'bytes_per_layer': lm_total,
        'bytes_total': lm_total,
        'time_us_per_layer': lm_total / bw * 1e6,
        'time_us_total': lm_total / bw * 1e6,
        'note': f'W[{VOCAB_SIZE}×{HIDDEN_SIZE}] * x[{HIDDEN_SIZE}] → [{VOCAB_SIZE}]',
    }

    # argmax
    argmax_bytes = VOCAB_SIZE * fp16_bytes
    ops['argmax'] = {
        'bytes_per_layer': argmax_bytes,
        'bytes_total': argmax_bytes,
        'time_us_per_layer': argmax_bytes / bw * 1e6,
        'time_us_total': argmax_bytes / bw * 1e6,
        'note': f'read logits[{VOCAB_SIZE}]',
    }

    return ops


def analyze_decode_kernels(kernels: list[KernelInfo], skip_first_n: int = 0):
    """Analyze decode-phase kernels and group by operation type."""
    categories = defaultdict(lambda: {'count': 0, 'total_ns': 0, 'kernels': []})

    for k in kernels[skip_first_n:]:
        cat = classify_kernel(k.name)
        categories[cat]['count'] += 1
        categories[cat]['total_ns'] += k.duration_ns
        categories[cat]['kernels'].append(k)

    return dict(categories)


def find_decode_steps(kernels: list[KernelInfo]) -> list[list[KernelInfo]]:
    """Try to identify individual decode steps (one per token generation).

    Heuristic: look for repeating patterns of kernel groups, separated by
    short gaps or graph replay boundaries.
    """
    steps = []
    current_step = []
    last_end = None

    for k in kernels:
        # Large gap (> 50us) or graph-related kernel suggests step boundary
        if last_end is not None:
            gap = k.start_ns - last_end
            if gap > 50000:  # 50us gap
                if current_step:
                    steps.append(current_step)
                    current_step = []

        current_step.append(k)
        last_end = k.end_ns

    if current_step:
        steps.append(current_step)

    return steps


def print_report(categories: dict, theoretical: dict, total_decode_kernels: int):
    """Print analysis report."""
    print("=" * 100)
    print("DECODE PHASE KERNEL TIMING BREAKDOWN")
    print("=" * 100)

    total_ns = sum(c['total_ns'] for c in categories.values())

    # Sort by total time descending
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1]['total_ns'])

    print(f"\n{'Category':<35} {'Count':>6} {'Total(us)':>12} {'Avg(us)':>10} {'%Total':>8}")
    print("-" * 75)

    for cat, info in sorted_cats:
        avg_us = info['total_ns'] / info['count'] / 1000 if info['count'] else 0
        pct = info['total_ns'] / total_ns * 100 if total_ns else 0
        total_us = info['total_ns'] / 1000
        print(f"{cat:<35} {info['count']:>6} {total_us:>12.1f} {avg_us:>10.2f} {pct:>7.1f}%")

    print("-" * 75)
    print(f"{'TOTAL':<35} {total_decode_kernels:>6} {total_ns/1000:>12.1f} {'':>10} {'100.0':>7}%")
    print(f"\nTotal decode GPU time: {total_ns/1e6:.3f} ms")

    # GPU gaps
    all_kernels = []
    for c in categories.values():
        all_kernels.extend(c['kernels'])
    all_kernels.sort(key=lambda k: k.start_ns)

    if len(all_kernels) > 1:
        total_gap_ns = 0
        gap_count = 0
        for i in range(1, len(all_kernels)):
            gap = all_kernels[i].start_ns - all_kernels[i-1].end_ns
            if gap > 0:
                total_gap_ns += gap
                gap_count += 1
        wall_ns = all_kernels[-1].end_ns - all_kernels[0].start_ns
        busy_ns = sum(k.duration_ns for k in all_kernels)
        print(f"\nWall time: {wall_ns/1e6:.3f} ms")
        print(f"GPU busy: {busy_ns/1e6:.3f} ms")
        print(f"GPU gaps: {total_gap_ns/1e6:.3f} ms ({gap_count} gaps)")
        print(f"GPU utilization: {busy_ns/wall_ns*100:.1f}%")

    # Theoretical roofline comparison
    print("\n" + "=" * 100)
    print("THEORETICAL ROOFLINE (A800 80GB PCIe, HBM BW = 2039 GB/s)")
    print("=" * 100)
    print(f"\n{'Operation':<35} {'Theory(us)':>12} {'×28 layers':>12} {'Bytes':>14} {'Note'}")
    print("-" * 100)

    # Per-layer ops
    per_layer_ops = [
        'input_rmsnorm', 'qkv_proj', 'qk_norm_rope', 'kv_cache_update',
        'attention_S256', 'attention_S512', 'attention_S1024',
        'o_proj', 'post_attn_add_rmsnorm', 'gate_up_proj', 'silu_mul',
        'down_proj', 'residual_add',
    ]

    total_theory_per_step = 0
    for op in per_layer_ops:
        if op not in theoretical:
            continue
        t = theoretical[op]
        per_layer = t['time_us_per_layer']
        total = t['time_us_total']
        bytes_total = t['bytes_total']
        note = t['note']
        print(f"{op:<35} {per_layer:>12.3f} {total:>12.1f} {bytes_total:>14,} {note}")
        if not op.startswith('attention_'):
            total_theory_per_step += total

    # Fused operations
    print(f"\n{'--- Fused variants ---':<35}")
    fused_ops = ['fused_post_attn_mlp', 'fused_o_proj_post_attn_mlp']
    for op in fused_ops:
        if op not in theoretical:
            continue
        t = theoretical[op]
        per_layer = t['time_us_per_layer']
        total = t['time_us_total']
        bytes_total = t['bytes_total']
        note = t['note']
        print(f"{op:<35} {per_layer:>12.3f} {total:>12.1f} {bytes_total:>14,} {note}")

    # Per-token ops
    print(f"\n{'--- Per-token ops ---':<35}")
    per_token_ops = ['final_rmsnorm', 'lm_head', 'argmax']
    total_per_token = 0
    for op in per_token_ops:
        if op not in theoretical:
            continue
        t = theoretical[op]
        print(f"{op:<35} {t['time_us_total']:>12.3f} {'':>12} {t['bytes_total']:>14,} {t['note']}")
        total_per_token += t['time_us_total']

    # Summary
    print("\n" + "=" * 100)
    print("THEORETICAL DECODE STEP SUMMARY (per token)")
    print("=" * 100)

    # Non-fused per-layer total
    non_fused_total = sum(theoretical[op]['time_us_total'] for op in
                         ['input_rmsnorm', 'qkv_proj', 'qk_norm_rope', 'kv_cache_update',
                          'o_proj', 'post_attn_add_rmsnorm', 'gate_up_proj', 'silu_mul',
                          'down_proj', 'residual_add'])

    # Best fused scenario (o_proj + post_attn_mlp persistent)
    fused_block = theoretical['fused_o_proj_post_attn_mlp']['time_us_total']
    fused_remaining = sum(theoretical[op]['time_us_total'] for op in
                         ['input_rmsnorm', 'qkv_proj', 'qk_norm_rope', 'kv_cache_update'])

    for seq_len in [256, 512, 1024]:
        attn_total = theoretical[f'attention_S{seq_len}']['time_us_total']

        non_fused = non_fused_total + attn_total + total_per_token
        best_fused = fused_remaining + fused_block + attn_total + total_per_token

        tok_s_nonfused = 1e6 / non_fused if non_fused > 0 else 0
        tok_s_fused = 1e6 / best_fused if best_fused > 0 else 0

        print(f"\n  seq_len={seq_len}:")
        print(f"    Non-fused roofline:  {non_fused:.1f} us/token → {tok_s_nonfused:.0f} tok/s")
        print(f"    Best-fused roofline: {best_fused:.1f} us/token → {tok_s_fused:.0f} tok/s")

    print(f"\n  Current actual: ~{1e6/255.65:.0f} us/token → 255.65 tok/s (5000-sample official)")
    print(f"  Current actual: ~{1e6/247.95:.0f} us/token → 247.95 tok/s (100-sample local)")


def main():
    parser = argparse.ArgumentParser(description="Analyze decode-phase nsys trace")
    parser.add_argument("sqlite_path", nargs='?', help="Path to nsys-exported .sqlite file")
    parser.add_argument("--theory-only", action="store_true",
                        help="Only print theoretical roofline, no trace analysis")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="Skip first N kernels (e.g. to skip prefill)")
    args = parser.parse_args()

    theoretical = compute_theoretical_times()

    if args.theory_only:
        print_report({}, theoretical, 0)
        return

    if not args.sqlite_path:
        print("No SQLite path provided. Showing theoretical analysis only.\n")
        print_report({}, theoretical, 0)
        return

    print(f"Loading kernels from {args.sqlite_path}...")
    kernels = load_kernels(args.sqlite_path)
    print(f"Loaded {len(kernels)} GPU kernels")

    if not kernels:
        print("No kernels found!")
        return

    categories = analyze_decode_kernels(kernels, skip_first_n=args.skip_first)
    total_count = sum(c['count'] for c in categories.values())
    print_report(categories, theoretical, total_count)

    # Print top kernels by name
    print("\n" + "=" * 100)
    print("TOP 20 KERNEL TYPES BY TOTAL TIME")
    print("=" * 100)
    kernel_by_name = defaultdict(lambda: {'count': 0, 'total_ns': 0})
    for k in kernels[args.skip_first:]:
        short_name = k.name[:80]
        kernel_by_name[short_name]['count'] += 1
        kernel_by_name[short_name]['total_ns'] += k.duration_ns

    sorted_kernels = sorted(kernel_by_name.items(), key=lambda x: -x[1]['total_ns'])
    print(f"\n{'Kernel Name':<82} {'Count':>6} {'Total(us)':>12} {'Avg(us)':>10}")
    print("-" * 115)
    for name, info in sorted_kernels[:20]:
        avg_us = info['total_ns'] / info['count'] / 1000
        print(f"{name:<82} {info['count']:>6} {info['total_ns']/1000:>12.1f} {avg_us:>10.2f}")


if __name__ == "__main__":
    main()
