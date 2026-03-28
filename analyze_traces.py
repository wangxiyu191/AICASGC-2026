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


SYNC_RUNTIME_NAMES = {
    'cudaStreamSynchronize',
    'cudaEventSynchronize',
    'cudaDeviceSynchronize',
}


def _build_fee_stack(fee_events, ts, tid):
    """Reconstruct the enclosing Python stack for a timestamp on one thread."""
    containing = []
    for ev in fee_events:
        if ev['tid'] != tid:
            continue
        start = ev['ts']
        end = start + ev['dur']
        if start <= ts <= end:
            containing.append((start, end, ev['name']))

    stack = []
    current_end = float('inf')
    for start, end, name in sorted(containing, key=lambda x: (x[0], -(x[1] - x[0]))):
        if end <= current_end:
            stack.append(name)
            current_end = end
    return stack


def _recent_cpu_ops(events_by_tid, sync_event, lookback_us=120.0, max_ops=8):
    """Collect CPU ops immediately preceding one sync event on the same thread."""
    tid_events = events_by_tid.get(sync_event['tid'], [])
    ts = sync_event['ts']
    recent = []
    for ev in reversed(tid_events):
        if ev['ts'] >= ts:
            continue
        if ts - ev['ts'] > lookback_us:
            break
        if ev['cat'] == 'cpu_op':
            recent.append(ev['name'])
            if len(recent) >= max_ops:
                break
    recent.reverse()
    return recent


def _sync_hint(sync_name, stack, prev_cpu_ops):
    """Heuristic explanation for a sync site."""
    stack_text = ' | '.join(stack)
    ops = set(prev_cpu_ops)

    if 'DecodeCudaGraphRunner._write_inputs' in stack_text and 'aten::copy_' in ops and 'aten::select' in ops:
        return 'Python scalar write into CUDA buffer slice triggers H2D memcpy and stream sync'
    if 'Event.synchronize' in stack_text:
        return 'Explicit host wait on CUDA event'
    if 'Tensor.item' in stack_text or 'aten::item' in ops or 'aten::_local_scalar_dense' in ops:
        return 'Host scalar materialization from device tensor'
    if sync_name == 'cudaEventSynchronize':
        return 'Host waits for recorded CUDA event'
    if sync_name == 'cudaStreamSynchronize':
        return 'Host waits for queued work on CUDA stream'
    return 'Needs manual inspection'


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
        'graph_launch_count': 0,
        'graph_launch_dur': 0.0,
        'graph_replay_count': 0,
        'graph_capture_count': 0,
        'decode_source': 'unknown',
        'sync_sites': [],
    }

    fee_events = []
    thread_events = defaultdict(list)

    for ev in events:
        cat = ev.get('cat', '')
        name = ev.get('name', '')
        dur = ev.get('dur', 0)
        ph = ev.get('ph', '')
        tid = ev.get('tid')

        # Only process complete events and duration events
        if ph not in ('X', 'B', 'E', 'i'):
            if cat:
                stats['cat_summary'][cat]['count'] += 1
            continue

        if cat:
            stats['cat_summary'][cat]['count'] += 1
            stats['cat_summary'][cat]['total_dur'] += dur

        if tid is not None and cat in ('FEE', 'cpu_op', 'cuda_runtime'):
            thread_events[tid].append({
                'ts': ev.get('ts', 0),
                'dur': dur,
                'cat': cat,
                'name': name,
                'tid': tid,
            })

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
            if name == 'cudaGraphLaunch':
                stats['graph_launch_count'] += 1
                stats['graph_launch_dur'] += dur
            elif name in ('cudaGraphInstantiateWithFlags', 'cudaGraphGetNodes'):
                stats['graph_capture_count'] += 1
        elif cat == 'FEE':
            fee_events.append({
                'ts': ev.get('ts', 0),
                'dur': dur,
                'tid': tid,
                'name': name,
            })
            if 'replay (' in name and 'evaluation_wrapper.py' in name:
                stats['graph_replay_count'] += 1
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

    if stats['graph_launch_count'] > 0:
        stats['decode_steps'] = stats['graph_launch_count']
        stats['decode_source'] = 'cudaGraph'
    elif stats['fa_count'] >= NUM_DECODER_LAYERS:
        stats['decode_steps'] = stats['fa_count'] // NUM_DECODER_LAYERS
        stats['decode_source'] = 'flash_attn'
    else:
        stats['decode_steps'] = 0
        stats['decode_source'] = 'none'

    for tid in thread_events:
        thread_events[tid].sort(key=lambda x: x['ts'])

    sync_aggregate = defaultdict(lambda: {
        'count': 0,
        'total_dur': 0.0,
        'max_dur': 0.0,
        'sync_name': '',
        'stack': (),
        'prev_cpu_ops': (),
        'hint': '',
    })
    for ev in events:
        if ev.get('cat') != 'cuda_runtime' or ev.get('name') not in SYNC_RUNTIME_NAMES:
            continue
        stack = tuple(_build_fee_stack(fee_events, ev.get('ts', 0), ev.get('tid')))
        prev_cpu_ops = tuple(_recent_cpu_ops(thread_events, ev))
        hint = _sync_hint(ev.get('name', ''), stack, prev_cpu_ops)
        key = (ev.get('name'), stack, prev_cpu_ops, hint)
        sync_aggregate[key]['count'] += 1
        sync_aggregate[key]['total_dur'] += ev.get('dur', 0)
        sync_aggregate[key]['max_dur'] = max(sync_aggregate[key]['max_dur'], ev.get('dur', 0))
        sync_aggregate[key]['sync_name'] = ev.get('name', '')
        sync_aggregate[key]['stack'] = stack
        sync_aggregate[key]['prev_cpu_ops'] = prev_cpu_ops
        sync_aggregate[key]['hint'] = hint

    stats['sync_sites'] = sorted(
        sync_aggregate.values(),
        key=lambda x: x['total_dur'],
        reverse=True,
    )

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
        'sync_sites': defaultdict(lambda: {
            'count': 0,
            'total_dur': 0.0,
            'max_dur': 0.0,
            'sync_name': '',
            'stack': (),
            'prev_cpu_ops': (),
            'hint': '',
        }),
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
        for site in stats.get('sync_sites', []):
            key = (site['sync_name'], site['stack'], site['prev_cpu_ops'], site['hint'])
            merged['sync_sites'][key]['count'] += site['count']
            merged['sync_sites'][key]['total_dur'] += site['total_dur']
            merged['sync_sites'][key]['max_dur'] = max(
                merged['sync_sites'][key]['max_dur'], site['max_dur'])
            merged['sync_sites'][key]['sync_name'] = site['sync_name']
            merged['sync_sites'][key]['stack'] = site['stack']
            merged['sync_sites'][key]['prev_cpu_ops'] = site['prev_cpu_ops']
            merged['sync_sites'][key]['hint'] = site['hint']

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
          f"{'CPU(ms)':<10} {'MemCpy(ms)':<11} {'FA/FAvar':>10} {'DecSteps':>10} {'Type':<10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*10} {'-'*10} {'-'*8}")

    for stats in sorted(all_stats, key=lambda x: int(x['file'].split('_')[1].split('.')[0])):
        gpu_dur = sum(v['total_dur'] for v in stats['kernel_ops'].values()) / 1000
        cpu_dur = sum(v['total_dur'] for v in stats['cpu_ops'].values()) / 1000
        mem_dur = sum(v['total_dur'] for v in stats['gpu_memcpy'].values()) / 1000
        fa_str = f"{stats['fa_count']}/{stats['fa_varlen_count']}"
        if stats['graph_launch_count'] > 0:
            phase = "Throughput"
        else:
            phase = "TTFT" if stats['decode_steps'] <= 1 else "Throughput"
        print(f"  {stats['file']:<22} {stats['filesize_mb']:<10.1f} {stats['num_events']:<10} "
              f"{gpu_dur:<10.1f} {cpu_dur:<10.1f} {mem_dur:<11.2f} {fa_str:>10} "
              f"{stats['decode_steps']:>10} {phase:<10}")


def analyze_phase_breakdown(all_stats):
    """Separate TTFT (prefill) and decode phase analysis."""
    ttft_stats = [s for s in all_stats if s['decode_steps'] <= 1 and s['graph_launch_count'] == 0]
    throughput_stats = [s for s in all_stats if s['decode_steps'] > 1 or s['graph_launch_count'] > 0]

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
        dec_graph_launch_dur = 0
        dec_graph_launch_count = 0
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
                elif name == 'cudaGraphLaunch':
                    dec_graph_launch_dur += info['total_dur']
                    dec_graph_launch_count += info['count']
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
        if dec_graph_launch_count > 0:
            print(f"  cudaGraphLaunch per step:            {dec_graph_launch_count/d:.0f} calls, "
                  f"{dec_graph_launch_dur/d/1000:.2f} ms")
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


def print_sync_stack_analysis(merged, top_n=10):
    """Print aggregated CUDA sync sites with Python stack context."""
    if not merged['sync_sites']:
        return

    sites = sorted(merged['sync_sites'].values(), key=lambda x: x['total_dur'], reverse=True)
    total_sync = sum(site['total_dur'] for site in sites)

    print(f"\n{'='*100}")
    print("  CUDA SYNC STACK ANALYSIS")
    print(f"{'='*100}")
    print(f"  Total sync time: {total_sync/1000:.2f} ms across {sum(site['count'] for site in sites)} calls")

    for idx, site in enumerate(sites[:top_n], start=1):
        avg_us = site['total_dur'] / site['count'] if site['count'] else 0.0
        print(f"\n  [{idx}] {site['sync_name']}  total={site['total_dur']/1000:.2f} ms  "
              f"count={site['count']}  avg={avg_us:.1f} us  max={site['max_dur']:.1f} us")
        print(f"      Hint: {site['hint']}")
        if site['prev_cpu_ops']:
            print(f"      Prev CPU ops: {' -> '.join(site['prev_cpu_ops'])}")
        if site['stack']:
            print("      Python stack:")
            for frame in site['stack'][-8:]:
                print(f"        {frame}")


def _diff_ops_dict(curr, base):
    diff = {}
    for name in set(curr) | set(base):
        count = curr.get(name, {}).get('count', 0) - base.get(name, {}).get('count', 0)
        total_dur = curr.get(name, {}).get('total_dur', 0.0) - base.get(name, {}).get('total_dur', 0.0)
        max_dur = max(curr.get(name, {}).get('max_dur', 0.0), base.get(name, {}).get('max_dur', 0.0))
        if count > 0 and total_dur > 0:
            diff[name] = {'count': count, 'total_dur': total_dur, 'max_dur': max_dur}
    return diff


def _pick_op(delta_ops, include_substrings, target_calls_per_step, steady_steps, exclude_substrings=()):
    candidates = []
    include_substrings = tuple(include_substrings)
    exclude_substrings = tuple(exclude_substrings)
    for name, info in delta_ops.items():
        if any(token not in name for token in include_substrings):
            continue
        if any(token in name for token in exclude_substrings):
            continue
        calls_per_step = info['count'] / steady_steps
        score = abs(calls_per_step - target_calls_per_step)
        candidates.append((score, -info['total_dur'], name, info))
    if not candidates:
        return None
    _, _, name, info = min(candidates)
    return name, info


def analyze_steady_state_decode(all_stats):
    """Approximate steady-state decode by subtracting generate_1 from generate_2."""
    by_file = {stats['file']: stats for stats in all_stats}
    ttft = by_file.get('generate_1.json')
    throughput = by_file.get('generate_2.json')
    if ttft is None or throughput is None or throughput['decode_steps'] <= 1:
        return

    steady_steps = throughput['decode_steps'] - 1
    delta_gpu_groups = _diff_ops_dict(throughput['gpu_groups'], ttft['gpu_groups'])
    delta_kernel_ops = _diff_ops_dict(throughput['kernel_ops'], ttft['kernel_ops'])
    total_gpu_us = sum(info['total_dur'] for info in delta_gpu_groups.values())

    print(f"\n{'='*100}")
    print("  STEADY-STATE DECODE ESTIMATE (generate_2 - generate_1)")
    print(f"{'='*100}")
    print(f"  Approx steady decode steps:         {steady_steps}")
    print(f"  Avg GPU kernel time per steady step: {total_gpu_us / steady_steps / 1000:.2f} ms")
    print("  Note: subtracts TTFT trace to remove prompt prefill + first token from generate_2.")

    print_group_table(
        delta_gpu_groups,
        "GPU Kernel per Steady Decode Step",
        total_gpu_us,
        divisor=steady_steps,
    )

    print(f"\n  Top GPU kernel deltas per steady step:")
    print(f"  {'Time(ms)':<10} {'Calls/step':<12} {'Kernel'}")
    print(f"  {'-'*10} {'-'*12} {'-'*60}")
    for name, info in sorted(delta_kernel_ops.items(), key=lambda x: x[1]['total_dur'], reverse=True)[:18]:
        print(f"  {info['total_dur']/steady_steps/1000:<10.4f} {info['count']/steady_steps:<12.2f} {name}")

    flash = _pick_op(delta_kernel_ops, ('[FlashAttn]',), NUM_DECODER_LAYERS, steady_steps)
    fused_post = _pick_op(delta_kernel_ops, ('_fused_add_rms_norm_fwd_kernel',), NUM_DECODER_LAYERS, steady_steps)
    gemm_big = _pick_op(
        delta_kernel_ops,
        ('[GEMM prefill]',),
        NUM_DECODER_LAYERS,
        steady_steps,
        exclude_substrings=('sliced1x2', '128x', '256x', 'cutlass', 'sm80_xmma'),
    )
    gemm_small = _pick_op(
        delta_kernel_ops,
        ('[GEMM prefill]', 'sliced1x2'),
        NUM_DECODER_LAYERS,
        steady_steps,
    )
    lm_head = _pick_op(
        delta_kernel_ops,
        ('[GEMM prefill]', '64x64_sliced1x2_ldg8_f2f_stages_64x5_tn'),
        1.0,
        steady_steps,
    )
    gemv_both = _pick_op(delta_kernel_ops, ('[GEMV]',), 2 * NUM_DECODER_LAYERS, steady_steps)
    silu = _pick_op(delta_kernel_ops, ('[Elementwise] SiLU',), NUM_DECODER_LAYERS, steady_steps)

    print(f"\n  Decode structure estimate per steady step:")
    print("  These rows combine measured kernel deltas with model-structure inference.")
    print(f"  {'Stage':<28} {'Time(ms)':<10} {'Per-layer(us)':<14} {'Basis'}")
    print(f"  {'-'*28} {'-'*10} {'-'*14} {'-'*40}")

    def _print_stage(stage, total_ms, basis, per_layer=True):
        per_layer_us = total_ms * 1000 / NUM_DECODER_LAYERS if per_layer else 0.0
        per_layer_text = f"{per_layer_us:.1f}" if per_layer else "-"
        print(f"  {stage:<28} {total_ms:<10.4f} {per_layer_text:<14} {basis}")

    if gemm_small:
        _print_stage("qkv_proj", gemm_small[1]['total_dur'] / steady_steps / 1000, "count≈28/layer, likely smaller fused linear")
    if gemm_big:
        _print_stage("gate_up_proj", gemm_big[1]['total_dur'] / steady_steps / 1000, "count≈28/layer, likely larger fused linear")
    if flash:
        _print_stage("attention", flash[1]['total_dur'] / steady_steps / 1000, "FlashAttn kernel count≈28/layer")
    if gemv_both:
        total_ms = gemv_both[1]['total_dur'] / steady_steps / 1000
        _print_stage("o_proj+down_proj", total_ms, "GEMV count≈56/step = 2 per layer")
        _print_stage("o_proj (inferred)", total_ms * 0.25, "split by weight bytes (2048x2048 vs 2048x6144)")
        _print_stage("down_proj (inferred)", total_ms * 0.75, "split by weight bytes (2048x2048 vs 2048x6144)")
    if fused_post:
        _print_stage("post_attn_add_rmsnorm", fused_post[1]['total_dur'] / steady_steps / 1000, "direct Triton kernel count≈28/layer")
    reduction_ms = delta_gpu_groups.get('Reduction', {}).get('total_dur', 0.0) / steady_steps / 1000
    cat_ms = delta_gpu_groups.get('Cat/Concat', {}).get('total_dur', 0.0) / steady_steps / 1000
    if reduction_ms > 0 or cat_ms > 0:
        _print_stage("input/qk norm + rope", reduction_ms + cat_ms, "Reduction + Cat/Concat; Mul/Copy counted below")
    if silu:
        _print_stage("SiLU activation", silu[1]['total_dur'] / steady_steps / 1000, "native SiLU kernel count≈28/layer")
    if lm_head:
        _print_stage("lm_head", lm_head[1]['total_dur'] / steady_steps / 1000, "count≈1/step large vocab projection", per_layer=False)

    known_ms = 0.0
    for picked in (gemm_small, gemm_big, flash, gemv_both, fused_post, silu, lm_head):
        if picked:
            known_ms += picked[1]['total_dur'] / steady_steps / 1000
    known_ms += reduction_ms + cat_ms
    residual_ms = total_gpu_us / steady_steps / 1000 - known_ms
    if residual_ms > 0:
        _print_stage(
            "fragmented Copy/Mul/misc",
            residual_ms,
            "remaining elementwise/copy/fill/cast/index work; main next target",
        )


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
    analyze_steady_state_decode(all_stats)

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
    print_sync_stack_analysis(merged, top_n=12)
    print_top_ops(merged['cuda_runtime'], 'CUDA RUNTIME CALLS', top_n=15)

    print(f"\n{'='*100}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*100}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
