#!/usr/bin/env python3
"""Run o_proj+post_attn+MLP persistent kernel profiler and generate an interactive HTML visualization.

Outputs: results/oproj_mlp_profile_viz.html
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import cuda_decode_gemv

PEAK_BW_GBS = 2039.0
NUM_ITERS = 20
WARMUP = 10

# Slot indices (must match OPROJ_MLP_PROFILE_SLOT_NAMES / OProjMlpProfileSlot enum)
# Slots 0-12: clock64() cycle counts (per-SM cycle counter)
# Slots 13-14: globaltimer nanoseconds (for calibration)
S_KERNEL_START       = 0
S_X_LOAD_DONE        = 1
S_OPROJ_DONE         = 2
S_GRIDSYNC1_START    = 3
S_GRIDSYNC1_DONE     = 4
S_RESNORM_DONE       = 5
S_GATEUP_ITER0_DONE  = 6
S_GATEUP_ITER1_DONE  = 7
S_GRIDSYNC2_START    = 8
S_GRIDSYNC2_DONE     = 9
S_DOWN_HIDDEN_LOADED = 10
S_DOWN_COMPUTE_DONE  = 11
S_KERNEL_DONE        = 12
S_GT_START           = 13
S_GT_END             = 14
NUM_SLOTS            = 15


def run_profile(device):
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    # Model dimensions for Qwen2.5-3B-Instruct
    residual = torch.randn((1, 1, 2048), dtype=torch.float16, device=device)
    attn_input = torch.randn((1, 1, 2048), dtype=torch.float16, device=device)
    o_proj_w = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    norm_w = torch.randn((2048,), dtype=torch.float16, device=device)
    gate_up_w = torch.randn((12288, 2048), dtype=torch.float16, device=device)
    down_proj_w = torch.randn((2048, 6144), dtype=torch.float16, device=device)
    eps = 1e-6

    # Warmup with the non-profile variant
    for _ in range(WARMUP):
        cuda_decode_gemv.fused_o_proj_post_attn_mlp_persistent_half(
            residual, attn_input, o_proj_w, None, norm_w, gate_up_w, down_proj_w, None, eps
        )
    torch.cuda.synchronize()

    # Collect profiles
    all_profiles = []
    for _ in range(NUM_ITERS):
        _, _, profile, meta = cuda_decode_gemv.fused_o_proj_post_attn_mlp_persistent_profile_half(
            residual, attn_input, o_proj_w, None, norm_w, gate_up_w, down_proj_w, None, eps
        )
        torch.cuda.synchronize()
        all_profiles.append(profile.to(torch.int64).cpu())

    return all_profiles, num_sms


def extract_block_data(profile_tensor, num_sms):
    """Extract per-block phase data from a single run's profile tensor [num_sms, 15].

    Slots 0-12 are clock64 cycle counts; slots 13-14 are globaltimer (ns) for calibration.
    We compute per-block effective GHz = (kernel_done - kernel_start) cycles / (gt_end - gt_start) ns,
    then convert each cycle delta to microseconds: delta_cycles / effective_ghz / 1000.
    """
    ts = profile_tensor.to(torch.float64)
    n = ts.shape[0]

    # Compute per-block effective frequency (GHz = cycles/ns)
    gt_span_ns = ts[:, S_GT_END] - ts[:, S_GT_START]     # globaltimer ns span
    clk_span   = ts[:, S_KERNEL_DONE] - ts[:, S_KERNEL_START]  # clock64 cycle span

    # Use median across all blocks as robust effective_ghz (all SMs run same frequency)
    valid = gt_span_ns > 0
    if valid.any():
        per_block_ghz = clk_span[valid] / gt_span_ns[valid]
        effective_ghz = float(per_block_ghz.median())
    else:
        effective_ghz = 1.41  # fallback to nominal A800 frequency

    blocks = []
    for b in range(n):
        t = ts[b]
        # Convert cycle delta to microseconds: cycles / ghz / 1000
        def c2us(cycles):
            return float(cycles) / effective_ghz / 1000.0

        x_load = c2us(t[S_X_LOAD_DONE] - t[S_KERNEL_START])
        o_proj = c2us(t[S_OPROJ_DONE] - t[S_X_LOAD_DONE])
        gridsync1_wait = c2us(t[S_GRIDSYNC1_START] - t[S_OPROJ_DONE])
        gridsync1 = c2us(t[S_GRIDSYNC1_DONE] - t[S_GRIDSYNC1_START])
        resnorm = c2us(t[S_RESNORM_DONE] - t[S_GRIDSYNC1_DONE])

        has_iter1 = t[S_GATEUP_ITER1_DONE].item() > 0
        iter0 = c2us(t[S_GATEUP_ITER0_DONE] - t[S_RESNORM_DONE])
        iter1 = c2us(t[S_GATEUP_ITER1_DONE] - t[S_GATEUP_ITER0_DONE]) if has_iter1 else 0.0

        gridsync2_wait = c2us(t[S_GRIDSYNC2_START] - (t[S_GATEUP_ITER1_DONE] if has_iter1 else t[S_GATEUP_ITER0_DONE]))
        gridsync2 = c2us(t[S_GRIDSYNC2_DONE] - t[S_GRIDSYNC2_START])
        down_ld = c2us(t[S_DOWN_HIDDEN_LOADED] - t[S_GRIDSYNC2_DONE])
        down_active = b < 64
        down_cp = c2us(t[S_DOWN_COMPUTE_DONE] - t[S_DOWN_HIDDEN_LOADED])
        total = c2us(t[S_KERNEL_DONE] - t[S_KERNEL_START])
        blocks.append({
            "block": b,
            "x_load": round(x_load, 3),
            "o_proj": round(o_proj, 3),
            "gridsync1_wait": round(gridsync1_wait, 3),
            "gridsync1": round(gridsync1, 3),
            "resnorm": round(resnorm, 3),
            "iter0": round(iter0, 3),
            "has_iter1": has_iter1,
            "iter1": round(iter1, 3),
            "gridsync2_wait": round(gridsync2_wait, 3),
            "gridsync2": round(gridsync2, 3),
            "down_ld": round(down_ld, 3),
            "down_active": down_active,
            "down_cp": round(down_cp, 3),
            "total": round(total, 3),
        })
    return blocks, effective_ghz


def compute_summary(all_profiles, num_sms):
    """Compute aggregated summary across runs.

    Profile data uses clock64 (cycles). We calibrate per-run using globaltimer slots.
    """
    warps_per_block = 32
    stride = num_sms * warps_per_block  # 3456

    data_bytes = {
        "o_proj": 2048 * 2048 * 2,
        "res+norm": 0,  # compute-bound, no HBM weight read
        "gate_up iter0": min(stride, 6144) * 2 * 2048 * 2,
        "gate_up iter1": max(0, 6144 - stride) * 2 * 2048 * 2,
        "down_proj": 2048 * 6144 * 2,
        "total": 2048 * 2048 * 2 + 12288 * 2048 * 2 + 2048 * 6144 * 2,
    }

    phase_stats = {
        "o_proj": {"sms": num_sms, "wall_us": [], "data_mb": data_bytes["o_proj"] / 1e6},
        "res+norm": {"sms": num_sms, "wall_us": [], "data_mb": 0},
        "grid_sync_1": {"sms": num_sms, "wall_us": [], "data_mb": 0},
        "gate_up iter0": {"sms": num_sms, "wall_us": [], "data_mb": data_bytes["gate_up iter0"] / 1e6},
        "gate_up iter1": {"sms": 84, "wall_us": [], "data_mb": data_bytes["gate_up iter1"] / 1e6},
        "grid_sync_2": {"sms": num_sms, "wall_us": [], "data_mb": 0},
        "down_proj": {"sms": 64, "wall_us": [], "data_mb": data_bytes["down_proj"] / 1e6},
        "total": {"sms": num_sms, "wall_us": [], "data_mb": data_bytes["total"] / 1e6},
    }

    for prof in all_profiles:
        ts = prof.to(torch.float64)

        # Per-run calibration: compute effective GHz from globaltimer span
        gt_span = ts[:, S_GT_END] - ts[:, S_GT_START]
        clk_span = ts[:, S_KERNEL_DONE] - ts[:, S_KERNEL_START]
        valid = gt_span > 0
        if valid.any():
            ghz = float((clk_span[valid] / gt_span[valid]).median())
        else:
            ghz = 1.41

        # Helper: max cycle delta across blocks → microseconds
        def max_c2us(cycle_deltas):
            return float(cycle_deltas.max()) / ghz / 1000.0

        wall_oproj = max_c2us(ts[:, S_OPROJ_DONE] - ts[:, S_X_LOAD_DONE])
        phase_stats["o_proj"]["wall_us"].append(wall_oproj)

        wall_sync1 = max_c2us(ts[:, S_GRIDSYNC1_DONE] - ts[:, S_GRIDSYNC1_START])
        phase_stats["grid_sync_1"]["wall_us"].append(wall_sync1)

        wall_resnorm = max_c2us(ts[:, S_RESNORM_DONE] - ts[:, S_GRIDSYNC1_DONE])
        phase_stats["res+norm"]["wall_us"].append(wall_resnorm)

        wall_iter0 = max_c2us(ts[:, S_GATEUP_ITER0_DONE] - ts[:, S_RESNORM_DONE])
        phase_stats["gate_up iter0"]["wall_us"].append(wall_iter0)

        mask = ts[:, S_GATEUP_ITER1_DONE] > 0
        if mask.any():
            wall_iter1 = max_c2us(ts[mask, S_GATEUP_ITER1_DONE] - ts[mask, S_GATEUP_ITER0_DONE])
        else:
            wall_iter1 = 0.0
        phase_stats["gate_up iter1"]["wall_us"].append(wall_iter1)

        wall_sync2 = max_c2us(ts[:, S_GRIDSYNC2_DONE] - ts[:, S_GRIDSYNC2_START])
        phase_stats["grid_sync_2"]["wall_us"].append(wall_sync2)

        wall_down = max_c2us(ts[:64, S_DOWN_COMPUTE_DONE] - ts[:64, S_DOWN_HIDDEN_LOADED])
        phase_stats["down_proj"]["wall_us"].append(wall_down)

        wall_total = max_c2us(ts[:, S_KERNEL_DONE] - ts[:, S_KERNEL_START])
        phase_stats["total"]["wall_us"].append(wall_total)

    summary = []
    for name, s in phase_stats.items():
        sorted_wall = sorted(s["wall_us"])
        med_wall = sorted_wall[len(sorted_wall) // 2]
        db = data_bytes.get(name, 0)
        bw = (db / (med_wall * 1000.0)) if med_wall > 0 and db > 0 else 0.0
        pct = (bw / PEAK_BW_GBS * 100) if bw > 0 else 0.0
        summary.append({
            "phase": name,
            "sms": s["sms"],
            "data_mb": round(s["data_mb"], 2),
            "wall_us": round(med_wall, 2),
            "bw_gbs": round(bw, 1),
            "pct_peak": round(pct, 1),
        })
    return summary


def generate_html(block_data_runs, summary, num_sms, effective_ghz=1.41):
    """Generate self-contained HTML visualization."""
    run_totals = []
    for i, blocks in enumerate(block_data_runs):
        max_total = max(b["total"] for b in blocks)
        run_totals.append((max_total, i))
    run_totals.sort()
    median_run_idx = run_totals[len(run_totals) // 2][1]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>O-Proj + Post-Attn MLP Persistent Kernel Per-SM Profile (clock64 calibrated)</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace;
    background: #0d1117;
    color: #c9d1d9;
    padding: 20px;
    font-size: 13px;
}}
h1 {{
    color: #58a6ff;
    font-size: 18px;
    margin-bottom: 6px;
}}
h2 {{
    color: #79c0ff;
    font-size: 15px;
    margin: 20px 0 10px 0;
    border-bottom: 1px solid #21262d;
    padding-bottom: 4px;
}}
.subtitle {{
    color: #8b949e;
    font-size: 12px;
    margin-bottom: 16px;
}}
/* Summary Table */
.summary-table {{
    border-collapse: collapse;
    width: 100%;
    max-width: 900px;
    margin-bottom: 20px;
}}
.summary-table th, .summary-table td {{
    padding: 6px 12px;
    text-align: right;
    border-bottom: 1px solid #21262d;
    white-space: nowrap;
}}
.summary-table th {{
    color: #8b949e;
    font-weight: 600;
    border-bottom: 2px solid #30363d;
    text-align: right;
}}
.summary-table th:first-child, .summary-table td:first-child {{
    text-align: left;
}}
.summary-table tr:hover {{
    background: #161b22;
}}
.summary-table .total-row {{
    border-top: 2px solid #30363d;
    font-weight: bold;
    color: #58a6ff;
}}
.bw-bar {{
    display: inline-block;
    height: 14px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 6px;
}}
/* Timeline chart */
.timeline-container {{
    width: 100%;
    overflow-x: auto;
    margin-bottom: 20px;
}}
.timeline-row {{
    display: flex;
    align-items: center;
    height: 16px;
    margin-bottom: 1px;
}}
.timeline-label {{
    width: 50px;
    min-width: 50px;
    text-align: right;
    padding-right: 8px;
    color: #8b949e;
    font-size: 11px;
}}
.timeline-bar-area {{
    flex: 1;
    position: relative;
    height: 14px;
}}
.timeline-seg {{
    position: absolute;
    height: 14px;
    border-radius: 1px;
    cursor: pointer;
    opacity: 0.85;
    transition: opacity 0.15s;
}}
.timeline-seg:hover {{
    opacity: 1.0;
    outline: 1px solid #fff;
}}
/* Legend */
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 10px 0 16px 0;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
}}
.legend-swatch {{
    width: 14px;
    height: 14px;
    border-radius: 2px;
    display: inline-block;
}}
/* Tooltip */
.tooltip {{
    display: none;
    position: fixed;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    z-index: 1000;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    max-width: 320px;
}}
.tooltip .tt-phase {{ color: #58a6ff; font-weight: 600; }}
.tooltip .tt-val {{ color: #e6edf3; }}
/* Heatmap */
.heatmap-container {{
    overflow-x: auto;
    margin-bottom: 20px;
}}
.heatmap-table {{
    border-collapse: collapse;
    font-size: 11px;
}}
.heatmap-table th {{
    padding: 4px 6px;
    color: #8b949e;
    font-weight: 600;
    border-bottom: 2px solid #30363d;
    position: sticky;
    top: 0;
    background: #0d1117;
    z-index: 1;
}}
.heatmap-table td {{
    padding: 3px 6px;
    text-align: right;
    border-bottom: 1px solid #161b22;
    white-space: nowrap;
}}
.heatmap-table td:first-child {{
    text-align: center;
    color: #8b949e;
    font-weight: 600;
}}
.heatmap-table tr:hover {{
    outline: 1px solid #30363d;
}}
/* Tabs */
.tabs {{
    display: flex;
    gap: 2px;
    margin-bottom: 12px;
}}
.tab {{
    padding: 6px 16px;
    background: #161b22;
    color: #8b949e;
    border-radius: 6px 6px 0 0;
    cursor: pointer;
    border: 1px solid #21262d;
    border-bottom: none;
    font-size: 12px;
}}
.tab.active {{
    background: #0d1117;
    color: #58a6ff;
    border-color: #30363d;
}}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
/* Run selector */
.run-selector {{
    margin: 8px 0;
    font-size: 12px;
    color: #8b949e;
}}
.run-selector select {{
    background: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: inherit;
    font-size: 12px;
}}
</style>
</head>
<body>

<h1>O-Proj + Post-Attn MLP Persistent Kernel Per-SM Profile (clock64)</h1>
<div class="subtitle">
    NVIDIA A800 80GB PCIe &middot; {num_sms} SMs &middot;
    Peak HBM BW: {PEAK_BW_GBS:.0f} GB/s &middot;
    {NUM_ITERS} runs (showing median) &middot;
    Timer: clock64 @ {effective_ghz:.4f} GHz (globaltimer calibrated)
</div>

<h2>1. Bandwidth Summary</h2>
<table class="summary-table" id="summaryTable"></table>

<h2>2. Per-SM Timeline</h2>
<div class="legend" id="legend"></div>
<div class="run-selector">
    Run: <select id="runSelect"></select>
</div>
<div class="timeline-container" id="timelineContainer"></div>

<h2>3. Per-SM Heatmap (us)</h2>
<div class="tabs">
    <div class="tab active" data-tab="heatmap">Duration Heatmap</div>
    <div class="tab" data-tab="table">Raw Table</div>
</div>
<div class="heatmap-container">
    <div class="tab-content active" id="tab-heatmap"></div>
    <div class="tab-content" id="tab-table"></div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const NUM_SMS = {num_sms};
const PEAK_BW = {PEAK_BW_GBS};
const summary = {json.dumps(summary)};
const allRuns = {json.dumps(block_data_runs)};
const medianRunIdx = {median_run_idx};

const PHASE_COLORS = {{
    x_load:         '#484f58',
    o_proj:         '#bc8cff',
    gridsync1_wait: '#6e7681',
    gridsync1:      '#da3633',
    resnorm:        '#d29922',
    iter0:          '#3fb950',
    iter1:          '#2ea043',
    gridsync2_wait: '#6e7681',
    gridsync2:      '#f85149',
    down_ld:        '#8b949e',
    down_cp:        '#58a6ff',
    down_idle:      '#21262d',
}};
const PHASE_LABELS = {{
    x_load:         'x load',
    o_proj:         'o_proj GEMV',
    gridsync1_wait: 'wait sync 1',
    gridsync1:      'grid sync 1',
    resnorm:        'res+RMSNorm',
    iter0:          'gate_up iter0',
    iter1:          'gate_up iter1',
    gridsync2_wait: 'wait sync 2',
    gridsync2:      'grid sync 2',
    down_ld:        'down hidden load',
    down_cp:        'down_proj compute',
    down_idle:      'down_proj (idle)',
}};

// Summary table
function renderSummary() {{
    const tbl = document.getElementById('summaryTable');
    let h = '<tr><th>Phase</th><th>Active SMs</th><th>Data (MB)</th><th>Wall (us)</th><th>BW (GB/s)</th><th>% Peak</th><th></th></tr>';
    summary.forEach((r, i) => {{
        const isTotal = r.phase === 'total';
        const cls = isTotal ? ' class="total-row"' : '';
        const barW = Math.max(0, Math.min(200, r.pct_peak * 2));
        const barColor = r.pct_peak > 80 ? '#3fb950' : r.pct_peak > 60 ? '#d29922' : r.pct_peak > 0 ? '#da3633' : '#21262d';
        const bwStr = r.bw_gbs > 0 ? r.bw_gbs.toFixed(1) : '--';
        const pctStr = r.pct_peak > 0 ? r.pct_peak.toFixed(1) + '%' : '--';
        const barHtml = r.pct_peak > 0 ? `<span class="bw-bar" style="width:${{barW}}px;background:${{barColor}}"></span>` : '';
        h += `<tr${{cls}}><td>${{r.phase}}</td><td>${{r.sms}}</td><td>${{r.data_mb.toFixed(2)}}</td><td>${{r.wall_us.toFixed(2)}}</td><td>${{bwStr}}</td><td>${{pctStr}}</td><td>${{barHtml}}</td></tr>`;
    }});
    tbl.innerHTML = h;
}}

// Legend
function renderLegend() {{
    const el = document.getElementById('legend');
    let h = '';
    for (const [key, color] of Object.entries(PHASE_COLORS)) {{
        h += `<div class="legend-item"><span class="legend-swatch" style="background:${{color}}"></span>${{PHASE_LABELS[key]}}</div>`;
    }}
    el.innerHTML = h;
}}

// Timeline
function renderTimeline(runIdx) {{
    const blocks = allRuns[runIdx];
    const container = document.getElementById('timelineContainer');
    const maxTotal = Math.max(...blocks.map(b => b.total));
    const scale = maxTotal > 0 ? 1.0 / maxTotal : 1.0;

    let h = '';
    for (const b of blocks) {{
        h += `<div class="timeline-row">`;
        h += `<div class="timeline-label">${{b.block}}</div>`;
        h += `<div class="timeline-bar-area">`;

        const segs = [];
        let offset = 0;
        const addSeg = (dur, phase) => {{
            if (dur > 0.001) {{
                segs.push({{ left: offset * scale * 100, width: dur * scale * 100, phase, dur }});
            }}
            offset += dur;
        }};

        addSeg(b.x_load, 'x_load');
        addSeg(b.o_proj, 'o_proj');
        addSeg(b.gridsync1_wait, 'gridsync1_wait');
        addSeg(b.gridsync1, 'gridsync1');
        addSeg(b.resnorm, 'resnorm');
        addSeg(b.iter0, 'iter0');
        if (b.has_iter1) {{
            addSeg(b.iter1, 'iter1');
        }}
        addSeg(b.gridsync2_wait, 'gridsync2_wait');
        addSeg(b.gridsync2, 'gridsync2');
        addSeg(b.down_ld, 'down_ld');
        if (b.down_active) {{
            addSeg(b.down_cp, 'down_cp');
        }} else {{
            addSeg(b.down_cp, 'down_idle');
        }}

        for (const s of segs) {{
            h += `<div class="timeline-seg" style="left:${{s.left.toFixed(2)}}%;width:${{Math.max(s.width, 0.15).toFixed(2)}}%;background:${{PHASE_COLORS[s.phase]}}" data-block="${{b.block}}" data-phase="${{PHASE_LABELS[s.phase]}}" data-dur="${{s.dur.toFixed(2)}}"></div>`;
        }}

        h += `</div></div>`;
    }}
    container.innerHTML = h;

    container.querySelectorAll('.timeline-seg').forEach(el => {{
        el.addEventListener('mouseenter', e => {{
            const tt = document.getElementById('tooltip');
            const block = e.target.dataset.block;
            const phase = e.target.dataset.phase;
            const dur = e.target.dataset.dur;
            tt.innerHTML = `<span class="tt-phase">Block ${{block}}</span><br><span class="tt-val">${{phase}}: ${{dur}} us</span>`;
            tt.style.display = 'block';
        }});
        el.addEventListener('mousemove', e => {{
            const tt = document.getElementById('tooltip');
            tt.style.left = (e.clientX + 12) + 'px';
            tt.style.top = (e.clientY - 10) + 'px';
        }});
        el.addEventListener('mouseleave', () => {{
            document.getElementById('tooltip').style.display = 'none';
        }});
    }});
}}

// Heatmap
function renderHeatmap(runIdx) {{
    const blocks = allRuns[runIdx];
    const phases = ['x_load', 'o_proj', 'gridsync1', 'resnorm', 'iter0', 'iter1', 'gridsync2', 'down_ld', 'down_cp', 'total'];
    const labels = ['x_load', 'o_proj', 'sync1', 'resnorm', 'iter0', 'iter1', 'sync2', 'down_ld', 'down_cp', 'total'];

    const maxPerPhase = {{}};
    for (const p of phases) {{
        maxPerPhase[p] = Math.max(...blocks.map(b => b[p] || 0), 0.001);
    }}

    let h = '<table class="heatmap-table"><tr><th>Block</th>';
    labels.forEach(l => {{ h += `<th>${{l}}</th>`; }});
    h += '<th>Status</th></tr>';

    for (const b of blocks) {{
        h += '<tr>';
        h += `<td>${{b.block}}</td>`;
        for (const p of phases) {{
            let val = b[p] || 0;
            const maxV = maxPerPhase[p];
            let intensity = val / maxV;

            if (p === 'iter1' && !b.has_iter1) {{
                h += `<td style="background:rgba(99,110,123,0.15);color:#484f58">--</td>`;
                continue;
            }}
            if (p === 'down_cp' && !b.down_active) {{
                h += `<td style="background:rgba(99,110,123,0.15);color:#484f58">${{val.toFixed(1)}}<small>*</small></td>`;
                continue;
            }}

            let r, g, bl;
            if (intensity < 0.5) {{
                const t = intensity * 2;
                r = Math.round(63 + t * (210 - 63));
                g = Math.round(185 - t * (185 - 153));
                bl = Math.round(80 - t * (80 - 34));
            }} else {{
                const t = (intensity - 0.5) * 2;
                r = Math.round(210 + t * (218 - 210));
                g = Math.round(153 - t * (153 - 54));
                bl = Math.round(34 - t * (34 - 51));
            }}
            const alpha = 0.3 + intensity * 0.5;
            h += `<td style="background:rgba(${{r}},${{g}},${{bl}},${{alpha}})">${{val.toFixed(1)}}</td>`;
        }}

        let status = '';
        if (b.has_iter1 && b.down_active) status = '2iter + down';
        else if (b.has_iter1 && !b.down_active) status = '2iter';
        else if (!b.has_iter1 && b.down_active) status = '1iter + down';
        else status = '1iter only';
        h += `<td style="color:#8b949e">${{status}}</td>`;
        h += '</tr>';
    }}
    h += '</table>';
    document.getElementById('tab-heatmap').innerHTML = h;

    // Raw table
    let t = '<table class="heatmap-table"><tr><th>Block</th><th>x_load</th><th>o_proj</th><th>sync1</th><th>resnorm</th><th>iter0</th><th>iter1</th><th>sync2</th><th>down_ld</th><th>down_cp</th><th>total</th><th>has_iter1</th><th>down_active</th></tr>';
    for (const b of blocks) {{
        t += `<tr><td>${{b.block}}</td>`;
        t += `<td>${{b.x_load.toFixed(2)}}</td>`;
        t += `<td>${{b.o_proj.toFixed(2)}}</td>`;
        t += `<td>${{b.gridsync1.toFixed(2)}}</td>`;
        t += `<td>${{b.resnorm.toFixed(2)}}</td>`;
        t += `<td>${{b.iter0.toFixed(2)}}</td>`;
        t += `<td>${{b.has_iter1 ? b.iter1.toFixed(2) : '--'}}</td>`;
        t += `<td>${{b.gridsync2.toFixed(2)}}</td>`;
        t += `<td>${{b.down_ld.toFixed(2)}}</td>`;
        t += `<td>${{b.down_cp.toFixed(2)}}</td>`;
        t += `<td>${{b.total.toFixed(2)}}</td>`;
        t += `<td>${{b.has_iter1 ? 'Y' : 'N'}}</td>`;
        t += `<td>${{b.down_active ? 'Y' : 'N'}}</td>`;
        t += '</tr>';
    }}
    t += '</table>';
    document.getElementById('tab-table').innerHTML = t;
}}

// Run selector
function initRunSelector() {{
    const sel = document.getElementById('runSelect');
    for (let i = 0; i < allRuns.length; i++) {{
        const opt = document.createElement('option');
        opt.value = i;
        const maxT = Math.max(...allRuns[i].map(b => b.total)).toFixed(1);
        opt.textContent = `#${{i}} (${{maxT}} us)` + (i === medianRunIdx ? ' [median]' : '');
        if (i === medianRunIdx) opt.selected = true;
        sel.appendChild(opt);
    }}
    sel.addEventListener('change', () => {{
        const idx = parseInt(sel.value);
        renderTimeline(idx);
        renderHeatmap(idx);
    }});
}}

// Tabs
document.querySelectorAll('.tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    }});
}});

// Init
renderSummary();
renderLegend();
initRunSelector();
renderTimeline(medianRunIdx);
renderHeatmap(medianRunIdx);
</script>
</body>
</html>"""
    return html


def main():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print("Running o_proj+post_attn+MLP profiler...")

    all_profiles, num_sms = run_profile(device)

    print("Extracting per-block data...")
    block_data_runs = []
    effective_ghz_list = []
    for prof in all_profiles:
        blocks, effective_ghz = extract_block_data(prof, num_sms)
        block_data_runs.append(blocks)
        effective_ghz_list.append(effective_ghz)

    median_ghz = sorted(effective_ghz_list)[len(effective_ghz_list) // 2]
    print(f"Calibrated SM frequency: {median_ghz:.4f} GHz (median across {len(effective_ghz_list)} runs)")

    print("Computing summary...")
    summary = compute_summary(all_profiles, num_sms)

    print("Generating HTML...")
    html = generate_html(block_data_runs, summary, num_sms, median_ghz)

    out_path = Path("results/oproj_mlp_profile_viz.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
