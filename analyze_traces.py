#!/usr/bin/env python3
"""Analyze PyTorch profiler trace files to identify performance bottlenecks.

Processes large trace JSON files one at a time to manage memory.
Outputs summary statistics for GPU kernels, CPU ops, memory ops, etc.
Includes TTFT (prefill) vs Decode phase separation analysis.

Usage:
    python analyze_traces.py [trace_dir]
    # Default: ./traces
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Qwen3-VL-2B has 28 decoder layers
NUM_DECODER_LAYERS = 28


def simplify_kernel_name(name):
    """Shorten verbose CUDA kernel names for readability."""
    # Flash attention kernels
    if 'flash' in name.lower() or 'fmha' in name.lower():
        if 'fwd' in name.lower():
            return f"[FlashAttn FWD] {name[:80]}"
        elif 'bwd' in name.lower():
            return f"[FlashAttn BWD] {name[:80]}"
        return f"[FlashAttn] {name[:80]}"

    # GEMV kernels (decode-phase vector-matrix multiply)
    if 'gemv' in name.lower():
        return f"[GEMV] {name[:80]}"

    # GEMM kernels
    if 'gemm' in name.lower() or 'xmma' in name.lower():
        if 'fprop' in name.lower() or 'conv' in name.lower():
            return f"[Conv GEMM] {name[:80]}"
        if 'wmma_tensorop' in name:
            return f"[GEMM decode] {name[:80]}"
        return f"[GEMM prefill] {name[:80]}"

    # Elementwise kernels
    if 'elementwise' in name.lower():
        if 'MulFunctor' in name:
            return "[Elementwise] Mul"
        elif 'AddFunctor' in name:
            return "[Elementwise] Add"
        elif 'CopyFunctor' in name or 'copy' in name.lower():
            return "[Elementwise] Copy"
        elif 'DivFunctor' in name:
            return "[Elementwise] Div"
        elif 'SiluFunctor' in name or 'silu' in name.lower():
            return "[Elementwise] SiLU"
        elif 'CompareEqFunctor' in name:
            return "[Elementwise] CompareEq"
        elif 'Fill' in name:
            return "[Elementwise] Fill"
        elif 'cast' in name.lower() or 'Copy' in name:
            return "[Elementwise] Cast/Copy"
        return f"[Elementwise] {name[:60]}"

    # LayerNorm / RMSNorm
    if 'layer_norm' in name.lower() or 'layernorm' in name.lower():
        return f"[LayerNorm] {name[:60]}"
    if 'rms_norm' in name.lower() or 'rmsnorm' in name.lower():
        return f"[RMSNorm] {name[:60]}"

    # Softmax
    if 'softmax' in name.lower():
        return f"[Softmax] {name[:60]}"

    # Reduction
    if 'reduce' in name.lower() or 'reduction' in name.lower():
        return f"[Reduction] {name[:60]}"

    # Index/scatter/gather
    if 'index' in name.lower() or 'scatter' in name.lower() or 'gather' in name.lower():
        return f"[Index/Scatter] {name[:60]}"

    # Cat operations
    if 'CatArray' in name:
        return f"[Cat] {name[:60]}"

    # Memset/fill
    if 'Memset' in name.lower() or 'memset' in name.lower():
        return f"[Memset] {name[:60]}"

    if len(name) > 100:
        return name[:100] + "..."
    return name


def categorize_gpu_kernel(name):
    """Categorize a GPU kernel into a high-level group."""
    nl = name.lower()
    if 'flash' in nl or 'fmha' in nl:
        return 'FlashAttn'
    if 'gemv' in nl:
        return 'GEMV'
    if 'wmma_tensorop' in nl:
        return 'GEMM (decode small)'
    if 'xmma_gemm' in nl or 'gemm' in nl:
        if 'fprop' in nl or 'conv' in nl:
            return 'Conv (Vision)'
        return 'GEMM (prefill large)'
    if 'layer_norm' in nl or 'rms' in nl:
        return 'Norm'
    if 'CatArray' in name:
        return 'Cat/Concat'
    if 'elementwise' in nl:
        return 'Elementwise'
    if 'reduce' in nl or 'reduction' in nl:
        return 'Reduction'
    return 'Other'


def categorize_cpu_op(name):
    """Categorize a CPU op into a high-level group."""
    if name in ('aten::linear', 'aten::matmul', 'aten::mm', 'aten::addmm'):
        return 'Linear/MatMul dispatch'
    if name in ('aten::to', 'aten::_to_copy', 'aten::copy_'):
        return 'Type conv/copy'
    if name in ('FlashAttnFunc', 'FlashAttnVarlenFunc',
                'flash_attn::_flash_attn_forward', 'flash_attn::_flash_attn_varlen_forward'):
        return 'FlashAttn dispatch'
    if name in ('aten::item', 'aten::_local_scalar_dense'):
        return 'item() sync'
    if name in ('aten::mul', 'aten::add', 'aten::pow', 'aten::mean',
                'aten::rsqrt', 'aten::neg', 'aten::abs'):
        return 'Math/RMSNorm ops'
    if name in ('aten::cat', 'aten::arange', 'aten::sum'):
        return 'Cat/Arange/Sum'
    if name == 'aten::is_nonzero':
        return 'is_nonzero sync'
    return 'Other'


def analyze_single_trace(filepath):
    """Load and analyze a single trace file, return summary dict."""
    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Loading {filepath.name} ({filesize_mb:.1f} MB)...", flush=True)

    with open(filepath, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', [])
    print(f"    {len(events)} events", flush=True)

    stats = {
        'file': filepath.name,
        'filesize_mb': filesize_mb,
        'num_events': len(events),
        'kernel_ops': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'cpu_ops': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'cuda_runtime': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'gpu_memcpy': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'gpu_memset': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'cat_summary': defaultdict(lambda: {'count': 0, 'total_dur': 0.0}),
        # High-level categories for phase analysis
        'gpu_groups': defaultdict(lambda: {'count': 0, 'total_dur': 0.0}),
        'cpu_groups': defaultdict(lambda: {'count': 0, 'total_dur': 0.0}),
        # Decode step estimation
        'fa_count': 0,
        'fa_varlen_count': 0,
        'item_count': 0,
    }

    for ev in events:
        cat = ev.get('cat', '')
        name = ev.get('name', '')
        dur = ev.get('dur', 0)
        ph = ev.get('ph', '')

        # Only process complete events and duration events
        if ph not in ('X', 'B', 'E', 'i'):
            if cat:
                stats['cat_summary'][cat]['count'] += 1
            continue

        if cat:
            stats['cat_summary'][cat]['count'] += 1
            stats['cat_summary'][cat]['total_dur'] += dur

        if cat == 'kernel':
            short_name = simplify_kernel_name(name)
            stats['kernel_ops'][short_name]['count'] += 1
            stats['kernel_ops'][short_name]['total_dur'] += dur
            stats['kernel_ops'][short_name]['max_dur'] = max(
                stats['kernel_ops'][short_name]['max_dur'], dur)
            grp = categorize_gpu_kernel(name)
            stats['gpu_groups'][grp]['count'] += 1
            stats['gpu_groups'][grp]['total_dur'] += dur
        elif cat == 'cpu_op':
            stats['cpu_ops'][name]['count'] += 1
            stats['cpu_ops'][name]['total_dur'] += dur
            stats['cpu_ops'][name]['max_dur'] = max(
                stats['cpu_ops'][name]['max_dur'], dur)
            grp = categorize_cpu_op(name)
            stats['cpu_groups'][grp]['count'] += 1
            stats['cpu_groups'][grp]['total_dur'] += dur
            if name == 'FlashAttnFunc':
                stats['fa_count'] += 1
            elif name == 'FlashAttnVarlenFunc':
                stats['fa_varlen_count'] += 1
            elif name == 'aten::item':
                stats['item_count'] += 1
        elif cat == 'cuda_runtime':
            stats['cuda_runtime'][name]['count'] += 1
            stats['cuda_runtime'][name]['total_dur'] += dur
            stats['cuda_runtime'][name]['max_dur'] = max(
                stats['cuda_runtime'][name]['max_dur'], dur)
        elif cat == 'gpu_memcpy':
            stats['gpu_memcpy'][name]['count'] += 1
            stats['gpu_memcpy'][name]['total_dur'] += dur
            stats['gpu_memcpy'][name]['max_dur'] = max(
                stats['gpu_memcpy'][name]['max_dur'], dur)
        elif cat == 'gpu_memset':
            stats['gpu_memset'][name]['count'] += 1
            stats['gpu_memset'][name]['total_dur'] += dur
            stats['gpu_memset'][name]['max_dur'] = max(
                stats['gpu_memset'][name]['max_dur'], dur)

    stats['decode_steps'] = stats['fa_count'] // NUM_DECODER_LAYERS \
        if stats['fa_count'] >= NUM_DECODER_LAYERS else 0

    return stats


def print_top_ops(ops_dict, title, top_n=20):
    """Print top-N operations sorted by total duration."""
    if not ops_dict:
        return

    sorted_ops = sorted(ops_dict.items(), key=lambda x: x[1]['total_dur'], reverse=True)
    total_dur = sum(v['total_dur'] for v in ops_dict.values())

    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"  Total: {total_dur/1000:.2f} ms across {sum(v['count'] for v in ops_dict.values())} calls")
    print(f"{'='*100}")
    print(f"  {'Rank':<5} {'Total(ms)':<12} {'%':>6} {'Count':<8} {'Avg(us)':<12} {'Max(us)':<12} {'Name'}")
    print(f"  {'-'*5} {'-'*12} {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*40}")

    for i, (name, info) in enumerate(sorted_ops[:top_n]):
        pct = (info['total_dur'] / total_dur * 100) if total_dur > 0 else 0
        avg = info['total_dur'] / info['count'] if info['count'] > 0 else 0
        print(f"  {i+1:<5} {info['total_dur']/1000:<12.3f} {pct:>5.1f}% "
              f"{info['count']:<8} {avg:<12.1f} {info['max_dur']:<12.1f} {name}")


def print_group_table(groups, title, total_dur, divisor=1, divisor_label=""):
    """Print a grouped breakdown table."""
    print(f"\n  {title}:")
    hdr_time = f"Time(ms){divisor_label}" if not divisor_label else f"Per{divisor_label}(ms)"
    print("  {:<30} {:>12} {:>10} {:>8}".format('Category', hdr_time, 'Count', '%'))
    print("  " + "-" * 62)
    for name, info in sorted(groups.items(), key=lambda x: -x[1]['total_dur']):
        dur = info['total_dur'] / divisor / 1000
        cnt = info['count'] / divisor
        pct = info['total_dur'] / total_dur * 100 if total_dur > 0 else 0
        print("  {:<30} {:>12.2f} {:>10.0f} {:>7.1f}%".format(name, dur, cnt, pct))


def merge_stats(all_stats):
    """Merge stats from multiple trace files."""
    merged = {
        'kernel_ops': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'cpu_ops': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'cuda_runtime': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'gpu_memcpy': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'gpu_memset': defaultdict(lambda: {'count': 0, 'total_dur': 0.0, 'max_dur': 0.0}),
        'cat_summary': defaultdict(lambda: {'count': 0, 'total_dur': 0.0}),
    }

    for stats in all_stats:
        for key in ['kernel_ops', 'cpu_ops', 'cuda_runtime', 'gpu_memcpy', 'gpu_memset']:
            for name, info in stats[key].items():
                merged[key][name]['count'] += info['count']
                merged[key][name]['total_dur'] += info['total_dur']
                merged[key][name]['max_dur'] = max(merged[key][name]['max_dur'], info['max_dur'])
        for cat, info in stats['cat_summary'].items():
            merged['cat_summary'][cat]['count'] += info['count']
            merged['cat_summary'][cat]['total_dur'] += info['total_dur']

    return merged


def print_category_summary(cat_summary):
    """Print overall category summary."""
    print(f"\n{'='*100}")
    print(f"  CATEGORY SUMMARY (All Files)")
    print(f"{'='*100}")
    print(f"  {'Category':<20} {'Count':<12} {'Total(ms)':<15} {'%':>6}")
    print(f"  {'-'*20} {'-'*12} {'-'*15} {'-'*6}")

    total_dur = sum(v['total_dur'] for v in cat_summary.values())
    sorted_cats = sorted(cat_summary.items(), key=lambda x: x[1]['total_dur'], reverse=True)

    for cat, info in sorted_cats:
        pct = (info['total_dur'] / total_dur * 100) if total_dur > 0 else 0
        print(f"  {cat:<20} {info['count']:<12} {info['total_dur']/1000:<15.3f} {pct:>5.1f}%")


def print_per_file_summary(all_stats):
    """Print per-file summary with decode step estimation."""
    print(f"\n{'='*110}")
    print(f"  PER-FILE SUMMARY")
    print(f"{'='*110}")
    print(f"  {'File':<22} {'Size(MB)':<10} {'Events':<10} {'GPU(ms)':<10} "
          f"{'CPU(ms)':<10} {'MemCpy(ms)':<11} {'FA/FAvar':>10} {'DecSteps':>10} {'Type':<8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*10} {'-'*10} {'-'*8}")

    for stats in sorted(all_stats, key=lambda x: int(x['file'].split('_')[1].split('.')[0])):
        gpu_dur = sum(v['total_dur'] for v in stats['kernel_ops'].values()) / 1000
        cpu_dur = sum(v['total_dur'] for v in stats['cpu_ops'].values()) / 1000
        mem_dur = sum(v['total_dur'] for v in stats['gpu_memcpy'].values()) / 1000
        fa_str = f"{stats['fa_count']}/{stats['fa_varlen_count']}"
        phase = "TTFT" if stats['decode_steps'] <= 1 else "Throughput"
        print(f"  {stats['file']:<22} {stats['filesize_mb']:<10.1f} {stats['num_events']:<10} "
              f"{gpu_dur:<10.1f} {cpu_dur:<10.1f} {mem_dur:<11.2f} {fa_str:>10} "
              f"{stats['decode_steps']:>10} {phase:<8}")


def analyze_phase_breakdown(all_stats):
    """Separate TTFT (prefill) and decode phase analysis."""
    ttft_stats = [s for s in all_stats if s['decode_steps'] <= 1]
    throughput_stats = [s for s in all_stats if s['decode_steps'] > 1]

    # --- TTFT ---
    if ttft_stats:
        print(f"\n{'='*100}")
        print(f"  TTFT / PREFILL PHASE ANALYSIS ({len(ttft_stats)} samples)")
        print(f"{'='*100}")

        ttft_gpu_groups = defaultdict(lambda: {'count': 0, 'total_dur': 0.0})
        ttft_cpu_groups = defaultdict(lambda: {'count': 0, 'total_dur': 0.0})
        ttft_total_gpu = 0
        ttft_total_cpu = 0

        for s in ttft_stats:
            for grp, info in s['gpu_groups'].items():
                ttft_gpu_groups[grp]['count'] += info['count']
                ttft_gpu_groups[grp]['total_dur'] += info['total_dur']
                ttft_total_gpu += info['total_dur']
            for grp, info in s['cpu_groups'].items():
                ttft_cpu_groups[grp]['count'] += info['count']
                ttft_cpu_groups[grp]['total_dur'] += info['total_dur']
                ttft_total_cpu += info['total_dur']

        n = len(ttft_stats)
        per_gpu = ttft_total_gpu / 1000 / n
        per_cpu = ttft_total_cpu / 1000 / n
        print(f"\n  Avg GPU kernel time per sample: {per_gpu:.1f} ms")
        print(f"  Avg CPU op time per sample:     {per_cpu:.1f} ms")
        print(f"  CPU/GPU ratio:                  {per_cpu/per_gpu:.1f}x")

        print_group_table(ttft_gpu_groups, "GPU Kernel Breakdown (avg per sample)",
                          ttft_total_gpu, divisor=n)
        print_group_table(ttft_cpu_groups, "CPU Op Breakdown (avg per sample)",
                          ttft_total_cpu, divisor=n)

    # --- Throughput ---
    if throughput_stats:
        total_decode_steps = sum(s['decode_steps'] for s in throughput_stats)

        print(f"\n{'='*100}")
        print(f"  DECODE / THROUGHPUT PHASE ANALYSIS "
              f"({len(throughput_stats)} samples, {total_decode_steps} total decode steps)")
        print(f"{'='*100}")

        for s in throughput_stats:
            print(f"  {s['file']}: {s['decode_steps']} decode steps")

        dec_gpu_groups = defaultdict(lambda: {'count': 0, 'total_dur': 0.0})
        dec_cpu_groups = defaultdict(lambda: {'count': 0, 'total_dur': 0.0})
        dec_total_gpu = 0
        dec_total_cpu = 0
        dec_launch_dur = 0
        dec_launch_count = 0
        dec_sync_dur = 0
        dec_sync_count = 0

        for s in throughput_stats:
            for grp, info in s['gpu_groups'].items():
                dec_gpu_groups[grp]['count'] += info['count']
                dec_gpu_groups[grp]['total_dur'] += info['total_dur']
                dec_total_gpu += info['total_dur']
            for grp, info in s['cpu_groups'].items():
                dec_cpu_groups[grp]['count'] += info['count']
                dec_cpu_groups[grp]['total_dur'] += info['total_dur']
                dec_total_cpu += info['total_dur']
            for name, info in s['cuda_runtime'].items():
                if name == 'cudaLaunchKernel':
                    dec_launch_dur += info['total_dur']
                    dec_launch_count += info['count']
                elif name == 'cudaStreamSynchronize':
                    dec_sync_dur += info['total_dur']
                    dec_sync_count += info['count']

        d = total_decode_steps
        per_gpu = dec_total_gpu / 1000 / d
        per_cpu = dec_total_cpu / 1000 / d
        print(f"\n  Avg GPU kernel time per decode step: {per_gpu:.2f} ms")
        print(f"  Avg CPU op time per decode step:     {per_cpu:.2f} ms")
        print(f"  CPU/GPU ratio:                       {per_cpu/per_gpu:.1f}x")
        print(f"  Theoretical max tok/s (GPU only):    {1000/per_gpu:.0f}")
        print(f"  Actual estimated tok/s (CPU bound):  {1000/per_cpu:.0f}")
        print(f"  cudaLaunchKernel per step:           {dec_launch_count/d:.0f} calls, "
              f"{dec_launch_dur/d/1000:.2f} ms")
        print(f"  cudaStreamSync per step:             {dec_sync_count/d:.0f} calls, "
              f"{dec_sync_dur/d/1000:.2f} ms")

        print_group_table(dec_gpu_groups, "GPU Kernel per Decode Step",
                          dec_total_gpu, divisor=d)
        print_group_table(dec_cpu_groups, "CPU Op per Decode Step",
                          dec_total_cpu, divisor=d)


def analyze_flash_attention(merged):
    """Analyze FlashAttention usage patterns."""
    print(f"\n{'='*100}")
    print(f"  FLASH ATTENTION ANALYSIS")
    print(f"{'='*100}")

    fa_kernels = {k: v for k, v in merged['kernel_ops'].items()
                  if 'FlashAttn' in k or 'flash' in k.lower() or 'fmha' in k.lower()}
    fa_cpu_ops = {k: v for k, v in merged['cpu_ops'].items()
                  if 'FlashAttn' in k or 'flash' in k.lower()}

    if fa_kernels:
        total_fa_gpu = sum(v['total_dur'] for v in fa_kernels.values())
        total_gpu = sum(v['total_dur'] for v in merged['kernel_ops'].values())
        print(f"  GPU Flash Attention: {total_fa_gpu/1000:.2f} ms "
              f"({total_fa_gpu/total_gpu*100:.1f}% of total GPU kernel time)")
        for name, info in sorted(fa_kernels.items(), key=lambda x: x[1]['total_dur'], reverse=True):
            print(f"    {info['total_dur']/1000:.2f} ms  (x{info['count']})  {name}")
    else:
        print("  No Flash Attention GPU kernels found!")

    if fa_cpu_ops:
        print(f"\n  CPU Flash Attention ops:")
        for name, info in sorted(fa_cpu_ops.items(), key=lambda x: x[1]['total_dur'], reverse=True):
            print(f"    {info['total_dur']/1000:.2f} ms  (x{info['count']})  {name}")


def analyze_gemm(merged):
    """Analyze GEMM (matrix multiplication) patterns."""
    print(f"\n{'='*100}")
    print(f"  GEMM / MATMUL ANALYSIS")
    print(f"{'='*100}")

    gemm_kernels = {k: v for k, v in merged['kernel_ops'].items()
                    if 'GEMM' in k or 'GEMV' in k}
    if gemm_kernels:
        total_gemm = sum(v['total_dur'] for v in gemm_kernels.values())
        total_gpu = sum(v['total_dur'] for v in merged['kernel_ops'].values())
        print(f"  GEMM/GEMV kernels: {total_gemm/1000:.2f} ms "
              f"({total_gemm/total_gpu*100:.1f}% of total GPU kernel time)")
        for name, info in sorted(gemm_kernels.items(), key=lambda x: x[1]['total_dur'],
                                 reverse=True)[:15]:
            avg = info['total_dur'] / info['count']
            print(f"    {info['total_dur']/1000:.2f} ms  (x{info['count']}, avg {avg:.1f} us)  {name}")


def analyze_memory_ops(merged):
    """Analyze memory operations."""
    print(f"\n{'='*100}")
    print(f"  MEMORY OPERATIONS ANALYSIS")
    print(f"{'='*100}")

    if merged['gpu_memcpy']:
        total_memcpy = sum(v['total_dur'] for v in merged['gpu_memcpy'].values())
        print(f"\n  GPU Memory Copies: {total_memcpy/1000:.2f} ms total")
        for name, info in sorted(merged['gpu_memcpy'].items(),
                                 key=lambda x: x[1]['total_dur'], reverse=True):
            print(f"    {info['total_dur']/1000:.2f} ms  (x{info['count']})  {name}")

    if merged['gpu_memset']:
        total_memset = sum(v['total_dur'] for v in merged['gpu_memset'].values())
        print(f"\n  GPU Memset: {total_memset/1000:.2f} ms total")
        for name, info in sorted(merged['gpu_memset'].items(),
                                 key=lambda x: x[1]['total_dur'], reverse=True):
            print(f"    {info['total_dur']/1000:.2f} ms  (x{info['count']})  {name}")

    malloc_ops = {k: v for k, v in merged['cuda_runtime'].items()
                  if 'malloc' in k.lower() or 'alloc' in k.lower()}
    if malloc_ops:
        total_malloc = sum(v['total_dur'] for v in malloc_ops.values())
        print(f"\n  CUDA Malloc: {total_malloc/1000:.2f} ms total")
        for name, info in sorted(malloc_ops.items(),
                                 key=lambda x: x[1]['total_dur'], reverse=True):
            print(f"    {info['total_dur']/1000:.2f} ms  (x{info['count']})  {name}")


def main():
    parser = argparse.ArgumentParser(description='Analyze PyTorch profiler trace files')
    parser.add_argument('trace_dir', nargs='?', default='./traces',
                        help='Directory containing generate_*.json trace files (default: ./traces)')
    parser.add_argument('--top', type=int, default=30,
                        help='Number of top ops to display (default: 30)')
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Error: trace directory '{trace_dir}' does not exist")
        return 1

    trace_files = sorted(trace_dir.glob('generate_*.json'),
                         key=lambda p: int(p.stem.split('_')[1]))

    if not trace_files:
        print(f"No generate_*.json files found in {trace_dir}")
        return 1

    print(f"Found {len(trace_files)} trace files in {trace_dir}")
    print(f"Total size: {sum(f.stat().st_size for f in trace_files) / (1024**3):.2f} GB\n")

    all_stats = []
    for tf in trace_files:
        try:
            stats = analyze_single_trace(tf)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ERROR processing {tf.name}: {e}")

    if not all_stats:
        print("No trace files processed successfully!")
        return 1

    # Per-file summary
    print_per_file_summary(all_stats)

    # Phase breakdown (TTFT vs Decode)
    analyze_phase_breakdown(all_stats)

    # Merge all stats for global analysis
    merged = merge_stats(all_stats)

    # Category summary
    print_category_summary(merged['cat_summary'])

    # Top GPU kernels
    print_top_ops(merged['kernel_ops'], 'TOP GPU KERNELS (by total time)', top_n=args.top)

    # Top CPU ops
    print_top_ops(merged['cpu_ops'], 'TOP CPU OPERATIONS (by total time)', top_n=args.top)

    # Detailed analyses
    analyze_flash_attention(merged)
    analyze_gemm(merged)
    analyze_memory_ops(merged)
    print_top_ops(merged['cuda_runtime'], 'CUDA RUNTIME CALLS', top_n=15)

    print(f"\n{'='*100}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*100}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
