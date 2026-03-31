#!/usr/bin/env python3
"""Render an SM-level CTA timeline for the 2-CTA-per-SM xwarp async variants."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import torch.nn.functional as F
from datasets import load_from_disk

import cuda_decode_gemv
from bench_decode_attention import _prepare_inputs
from bench_decode_linears import _run_capture
from evaluation_wrapper import VLMModel


ITER_COLORS = {
    "0": "#4e79a7",
    "1": "#f28e2b",
    "2": "#59a14f",
    "3": "#e15759",
}


def _round_us(value: float) -> float:
    return round(float(value), 3)


def _variant_trace_name(variant: str) -> str:
    if variant == "sms2":
        return "sms2"
    if variant == "sms2_balanced":
        return "sms2_balanced"
    raise ValueError(f"unknown variant: {variant}")


def _collect_trace_payload(args: argparse.Namespace) -> tuple[dict, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    print(
        f"[Sms2Timeline] variant={args.variant} sample={args.sample_idx} layer={args.layer_idx} "
        f"step={args.decode_step} target={args.target} repeats={args.repeats}"
    )
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])
    payloads, capture_meta = _run_capture(
        vlm,
        inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        layer_idx=args.layer_idx,
    )

    payload = payloads[args.target]
    x = payload["x"]
    weight = payload["weight"]
    bias = payload["bias"]
    if weight.shape[1] != 6144:
        raise RuntimeError(f"{args.target} weight shape {tuple(weight.shape)} is not k=6144")
    device_props = torch.cuda.get_device_properties(x.device)

    ref = F.linear(x, weight, bias)
    trace_reps = []
    smid_reps = []
    row_base_reps = []
    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    meta = None

    with torch.no_grad():
        cuda_decode_gemv.nosplit_matvec_half(x, weight, bias)
        torch.cuda.synchronize()
        for rep in range(args.repeats):
            if args.variant == "sms2":
                out, trace, smid, row_base, meta = cuda_decode_gemv.nosplit_matvec_half_profile_xwarp_sms2_cta_trace(
                    x, weight, bias
                )
            elif args.variant == "sms2_balanced":
                out, trace, smid, row_base, meta = cuda_decode_gemv.nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace(
                    x, weight, bias
                )
            else:
                raise ValueError(f"unknown variant: {args.variant}")
            torch.cuda.synchronize()
            if rep == 0:
                diff = (out - ref).abs()
                max_abs_diff = float(diff.max().item())
                mean_abs_diff = float(diff.float().mean().item())
            trace_reps.append(trace.cpu())
            smid_reps.append(smid.cpu())
            row_base_reps.append(row_base.cpu())

    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "layer_idx": int(args.layer_idx),
        "decode_step": int(args.decode_step),
        "bucketed": bool(args.bucketed),
        "repeats": int(args.repeats),
        "prompt_length": int(capture_meta["prompt_length"]),
        "max_cache_len": int(capture_meta["max_cache_len"]),
        "target": args.target,
        "variant": args.variant,
        "x_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "has_bias": bias is not None,
        "kernel_name": meta["kernel_name"],
        "launch_grid_blocks": int(meta["launch_grid_blocks"]),
        "full_grid_blocks": int(meta["full_grid_blocks"]),
        "max_iters": int(meta["max_iters"]),
        "sms": int(meta["sms"]),
        "device_name": device_props.name,
        "device_cc": f"{device_props.major}.{device_props.minor}",
        "slot_names": list(meta["slot_names"]),
        "max_abs_diff_vs_f_linear": max_abs_diff,
        "mean_abs_diff_vs_f_linear": mean_abs_diff,
    }
    return metadata, trace_reps, smid_reps, row_base_reps


def _assign_lanes(blocks: list[dict[str, object]]) -> int:
    lane_ends: list[float] = []
    for block in sorted(blocks, key=lambda item: (float(item["start_us"]), int(item["block"]))):
        start_us = float(block["start_us"])
        lane = 0
        while lane < len(lane_ends) and start_us < lane_ends[lane]:
            lane += 1
        if lane == len(lane_ends):
            lane_ends.append(float(block["end_us"]))
        else:
            lane_ends[lane] = float(block["end_us"])
        block["lane"] = lane
    return max(1, len(lane_ends))


def _build_processed_timeline(
    metadata: dict,
    trace_reps: list[torch.Tensor],
    smid_reps: list[torch.Tensor],
    row_base_reps: list[torch.Tensor],
) -> dict:
    all_durations = []
    repeat_entries = []
    active_smids = set()

    for rep_idx, (trace_rep, smid_rep, row_base_rep) in enumerate(zip(trace_reps, smid_reps, row_base_reps)):
        starts = trace_rep[..., 0]
        valid_starts = starts[starts > 0]
        if valid_starts.numel() == 0:
            continue
        kernel_base = int(valid_starts.min().item())
        block_rows = []

        for block in range(trace_rep.shape[0]):
            smid = int(smid_rep[block].item())
            if smid < 0:
                continue
            segments = []
            for iter_idx in range(trace_rep.shape[1]):
                seg_start = int(trace_rep[block, iter_idx, 0].item())
                seg_end = int(trace_rep[block, iter_idx, 1].item())
                base_row = int(row_base_rep[block, iter_idx].item())
                if base_row < 0 or seg_start <= 0 or seg_end <= seg_start:
                    continue
                segments.append(
                    {
                        "iter_idx": int(iter_idx),
                        "row_base": int(base_row),
                        "start_us": _round_us((seg_start - kernel_base) / 1000.0),
                        "end_us": _round_us((seg_end - kernel_base) / 1000.0),
                        "dur_us": _round_us((seg_end - seg_start) / 1000.0),
                    }
                )
            if not segments:
                continue

            start_us = min(seg["start_us"] for seg in segments)
            end_us = max(seg["end_us"] for seg in segments)
            block_rows.append(
                {
                    "block": int(block),
                    "smid": int(smid),
                    "start_us": _round_us(start_us),
                    "end_us": _round_us(end_us),
                    "dur_us": _round_us(end_us - start_us),
                    "iter_segments": segments,
                }
            )
            all_durations.append(float(end_us - start_us))
            active_smids.add(smid)

        block_rows.sort(key=lambda item: (int(item["smid"]), float(item["start_us"]), int(item["block"])))
        sm_groups: dict[int, list[dict[str, object]]] = {}
        for block in block_rows:
            sm_groups.setdefault(int(block["smid"]), []).append(block)
        sm_lane_counts: dict[str, int] = {}
        for smid, blocks in sm_groups.items():
            sm_lane_counts[str(smid)] = _assign_lanes(blocks)

        durations = torch.tensor([float(item["dur_us"]) for item in block_rows], dtype=torch.float64)
        second_iter_counter = Counter(
            _round_us(seg["start_us"])
            for block in block_rows
            for seg in block["iter_segments"]
            if int(seg["iter_idx"]) > 0
        )
        repeat_entries.append(
            {
                "rep": int(rep_idx),
                "kernel_end_us": _round_us(max(float(item["end_us"]) for item in block_rows)) if block_rows else 0.0,
                "active_smid_count": int(len(sm_groups)),
                "max_lanes_per_smid": int(max((int(item["lane"]) + 1) for item in block_rows)) if block_rows else 0,
                "cta_duration_summary": {
                    "count": int(len(block_rows)),
                    "mean_us": _round_us(float(durations.mean().item())) if durations.numel() else 0.0,
                    "median_us": _round_us(float(durations.median().item())) if durations.numel() else 0.0,
                    "p95_us": _round_us(float(torch.quantile(durations, 0.95).item())) if durations.numel() else 0.0,
                    "max_us": _round_us(float(durations.max().item())) if durations.numel() else 0.0,
                },
                "second_iter_starts": {str(k): int(v) for k, v in sorted(second_iter_counter.items())},
                "block_rows": block_rows,
                "smid_order": sorted(int(smid) for smid in sm_groups.keys()),
                "sm_lane_counts": sm_lane_counts,
            }
        )

    duration_tensor = torch.tensor(all_durations, dtype=torch.float64)
    global_thresholds = {
        "mean_us": _round_us(float(duration_tensor.mean().item())) if duration_tensor.numel() else 0.0,
        "p95_us": _round_us(float(torch.quantile(duration_tensor, 0.95).item())) if duration_tensor.numel() else 0.0,
        "p99_us": _round_us(float(torch.quantile(duration_tensor, 0.99).item())) if duration_tensor.numel() else 0.0,
        "max_us": _round_us(float(duration_tensor.max().item())) if duration_tensor.numel() else 0.0,
    }
    default_repeat_index = 0
    if repeat_entries:
        default_repeat_index = max(range(len(repeat_entries)), key=lambda idx: repeat_entries[idx]["kernel_end_us"])

    return {
        "metadata": metadata,
        "iter_colors": ITER_COLORS,
        "global_thresholds": global_thresholds,
        "active_smids": sorted(active_smids),
        "default_repeat_index": int(default_repeat_index),
        "repeats": repeat_entries,
    }


def _build_html(processed: dict) -> str:
    data_json = json.dumps(processed, ensure_ascii=False, separators=(",", ":"))
    title = html.escape("nosplit xwarp sms2 timeline")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "DejaVu Sans Mono", "SFMono-Regular", Consolas, monospace;
      background: #f5f1e8;
      color: #1f1f1f;
    }}
    .page {{ padding: 18px 22px 28px 22px; }}
    .title {{ font-size: 20px; font-weight: 700; margin-bottom: 8px; }}
    .meta {{ font-size: 12px; margin-bottom: 14px; line-height: 1.5; }}
    .controls {{ display: flex; gap: 14px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }}
    .controls label {{ font-size: 13px; font-weight: 600; }}
    .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
    .card {{
      border: 1px solid #b9b2a3;
      background: #fffaf0;
      padding: 10px 12px;
      min-width: 180px;
      box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04);
    }}
    .card .k {{ font-size: 11px; color: #6b665b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .card .v {{ font-size: 18px; font-weight: 700; }}
    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; font-size: 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 12px; height: 12px; border: 1px solid rgba(0,0,0,0.25); display: inline-block; }}
    .notes {{ font-size: 12px; color: #534f47; margin-bottom: 10px; line-height: 1.45; }}
    .svg-wrap {{ overflow-x: auto; border: 1px solid #c7c0b1; background: #fffdf8; }}
    svg {{ display: block; }}
    .axis {{ stroke: #7c776c; stroke-width: 1; }}
    .tick {{ stroke: #d8d1c2; stroke-width: 1; }}
    .sm-label {{ font-size: 11px; fill: #3f3a32; dominant-baseline: middle; text-anchor: end; }}
    .axis-label {{ font-size: 12px; fill: #3f3a32; }}
    .tick-label {{ font-size: 10px; fill: #6f695f; text-anchor: middle; }}
    .row-bg {{ fill: #fcf7eb; }}
    .row-sep {{ stroke: #eee8d7; stroke-width: 1; }}
    .bar-outline {{ fill: none; stroke-width: 0.9; }}
    .footer {{ margin-top: 8px; font-size: 11px; color: #6f695f; line-height: 1.45; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="title" id="page-title"></div>
    <div class="meta" id="meta"></div>
    <div class="controls">
      <label for="rep-select">Repeat</label>
      <select id="rep-select"></select>
      <button id="prev-btn" type="button">Prev</button>
      <button id="next-btn" type="button">Next</button>
      <span id="repeat-headline"></span>
    </div>
    <div class="summary" id="summary"></div>
    <div class="legend" id="legend"></div>
    <div class="notes">
      <span id="notes-text"></span>
    </div>
    <div class="svg-wrap">
      <svg id="timeline"></svg>
    </div>
    <div class="footer" id="footer"></div>
  </div>
  <script>
    const payload = {data_json};
    const selectEl = document.getElementById("rep-select");
    const summaryEl = document.getElementById("summary");
    const legendEl = document.getElementById("legend");
    const metaEl = document.getElementById("meta");
    const footerEl = document.getElementById("footer");
    const headlineEl = document.getElementById("repeat-headline");
    const titleEl = document.getElementById("page-title");
    const notesEl = document.getElementById("notes-text");
    const svg = document.getElementById("timeline");

    function fmt(value) {{
      return Number(value).toFixed(3).replace(/\\.000$/, "");
    }}

    function niceStep(maxValue) {{
      const rough = maxValue / 10.0;
      const power = Math.pow(10, Math.floor(Math.log10(Math.max(rough, 1e-6))));
      const scaled = rough / power;
      let step = 1;
      if (scaled > 5) step = 10;
      else if (scaled > 2) step = 5;
      else if (scaled > 1) step = 2;
      return step * power;
    }}

    function buildLegend() {{
      const items = Object.entries(payload.iter_colors).map(([iterIdx, color]) =>
        `<span><i class="swatch" style="background:${{color}}"></i>iter${{iterIdx}}</span>`
      );
      items.push(`<span><i class="swatch" style="background:transparent;border:1.6px solid #222"></i>&ge; global p95 duration</span>`);
      items.push(`<span><i class="swatch" style="background:transparent;border:1.8px solid #d81b60"></i>&ge; global p99 duration</span>`);
      legendEl.innerHTML = items.join("");
    }}

    function buildMeta() {{
      const m = payload.metadata;
      const variantLabel = m.variant === "sms2_balanced" ? "sms2_balanced" : "sms2";
      titleEl.textContent = `SM-level CTA timeline for k6144 xwarp async ${{variantLabel}}`;
      notesEl.textContent =
        m.variant === "sms2_balanced"
          ? `Each horizontal bar is one CTA. The bar is segmented by balanced row batches: iter0 is the first 8-row chunk, iter1 is the remaining 6/7-row chunk for that CTA. This trace was captured on ${{m.device_name}} (sm_${{String(m.device_cc).replace(".", "")}}) with launch_grid=${{m.launch_grid_blocks}} = min(full_grid=${{m.full_grid_blocks}}, 2 * sms=${{m.sms}}), so there is no warp-level coloring noise here.`
          : `Each horizontal bar is one CTA. The bar is segmented by grid-stride iterations: iter0 is the first row-tile batch, iter1 is the second. This trace was captured on ${{m.device_name}} (sm_${{String(m.device_cc).replace(".", "")}}) with launch_grid=${{m.launch_grid_blocks}} = min(full_grid=${{m.full_grid_blocks}}, 2 * sms=${{m.sms}}), so there is no warp-level coloring noise here.`;
      metaEl.textContent =
        `device=${{m.device_name}} cc=${{m.device_cc}} variant=${{m.variant}} target=${{m.target}} kernel=${{m.kernel_name}} sample=${{m.sample_idx}} layer=${{m.layer_idx}} decode_step=${{m.decode_step}} ` +
        `x=${{m.x_shape.join("x")}} weight=${{m.weight_shape.join("x")}} launch_grid=${{m.launch_grid_blocks}} full_grid=${{m.full_grid_blocks}} ` +
        `max_iters=${{m.max_iters}} sms=${{m.sms}} repeats=${{m.repeats}} ` +
        `max_abs_diff=${{m.max_abs_diff_vs_f_linear.toExponential(3)}} mean_abs_diff=${{m.mean_abs_diff_vs_f_linear.toExponential(3)}}`;
      footerEl.textContent =
        `CTA duration mean=${{fmt(payload.global_thresholds.mean_us)}}us, p95=${{fmt(payload.global_thresholds.p95_us)}}us, ` +
        `p99=${{fmt(payload.global_thresholds.p99_us)}}us, max=${{fmt(payload.global_thresholds.max_us)}}us.`;
    }}

    function buildSummary(rep) {{
      const slowBlocks = rep.block_rows
        .slice()
        .sort((a, b) => b.dur_us - a.dur_us)
        .slice(0, 5)
        .map((row) => `b${{row.block}}@sm${{row.smid}}=${{fmt(row.dur_us)}}us`)
        .join("  ");
      const iter1Starts = Object.entries(rep.second_iter_starts)
        .map(([start, count]) => `@${{start}}us:${{count}}`)
        .join("  ");

      summaryEl.innerHTML = [
        ["kernel span", `${{fmt(rep.kernel_end_us)}} us`],
        ["active sms", `${{rep.active_smid_count}}`],
        ["max lanes / sm", `${{rep.max_lanes_per_smid}}`],
        ["cta median", `${{fmt(rep.cta_duration_summary.median_us)}} us`],
        ["cta p95", `${{fmt(rep.cta_duration_summary.p95_us)}} us`],
        ["cta max", `${{fmt(rep.cta_duration_summary.max_us)}} us`],
        ["iter1 starts", iter1Starts || "none"],
        ["top slow blocks", slowBlocks || "none"],
      ].map(([k, v]) => `<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");
    }}

    function render(repIndex) {{
      const rep = payload.repeats[repIndex];
      headlineEl.textContent = `default=${{payload.default_repeat_index}}, showing repeat #${{rep.rep}}`;
      buildSummary(rep);

      const left = 78;
      const right = 22;
      const top = 34;
      const bottom = 34;
      const smGap = 6;
      const laneHeight = 9;
      const laneGap = 2;
      const width = 1800;
      const innerWidth = width - left - right;
      const sms = rep.smid_order.slice().sort((a, b) => a - b);
      const rowLayouts = [];
      let cursorY = top;
      for (const sm of sms) {{
        const lanes = rep.sm_lane_counts[String(sm)] || 1;
        const bodyH = lanes * laneHeight + (lanes - 1) * laneGap;
        rowLayouts.push({{ sm, lanes, y: cursorY, height: bodyH }});
        cursorY += bodyH + smGap;
      }}
      const height = cursorY + bottom;
      const scale = innerWidth / Math.max(rep.kernel_end_us, 1e-9);
      const tickStep = niceStep(rep.kernel_end_us);
      const tickCount = Math.ceil(rep.kernel_end_us / tickStep);
      const rowMap = Object.fromEntries(rowLayouts.map((row) => [String(row.sm), row]));

      const parts = [];
      parts.push(`<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#fffdf8"></rect>`);
      parts.push(`<line class="axis" x1="${{left}}" y1="${{top - 10}}" x2="${{left}}" y2="${{height - bottom + 4}}"></line>`);
      parts.push(`<line class="axis" x1="${{left}}" y1="${{height - bottom + 4}}" x2="${{width - right}}" y2="${{height - bottom + 4}}"></line>`);
      for (let t = 0; t <= tickCount; t++) {{
        const value = Math.min(rep.kernel_end_us, t * tickStep);
        const x = left + value * scale;
        parts.push(`<line class="tick" x1="${{x}}" y1="${{top - 10}}" x2="${{x}}" y2="${{height - bottom + 4}}"></line>`);
        parts.push(`<text class="tick-label" x="${{x}}" y="${{height - 10}}">${{fmt(value)}}</text>`);
      }}
      parts.push(`<text class="axis-label" x="${{width / 2}}" y="${{height - 2}}">time (us)</text>`);
      parts.push(`<text class="axis-label" x="14" y="18">sm id</text>`);

      for (const row of rowLayouts) {{
        parts.push(`<rect class="row-bg" x="${{left}}" y="${{row.y - 1}}" width="${{innerWidth}}" height="${{row.height + 2}}" rx="2"></rect>`);
        parts.push(`<text class="sm-label" x="${{left - 8}}" y="${{row.y + row.height / 2}}">sm${{row.sm}}</text>`);
        parts.push(`<line class="row-sep" x1="${{left}}" y1="${{row.y + row.height + smGap / 2}}" x2="${{width - right}}" y2="${{row.y + row.height + smGap / 2}}"></line>`);
      }}

      for (const block of rep.block_rows) {{
        const row = rowMap[String(block.smid)];
        if (!row) continue;
        const lane = block.lane || 0;
        const y = row.y + lane * (laneHeight + laneGap);
        const x0 = left + block.start_us * scale;
        const totalW = Math.max((block.end_us - block.start_us) * scale, 0.8);
        const outlineColor =
          block.dur_us >= payload.global_thresholds.p99_us ? "#d81b60" :
          block.dur_us >= payload.global_thresholds.p95_us ? "#222222" : "#6d675d";
        const outlineWidth =
          block.dur_us >= payload.global_thresholds.p99_us ? 1.8 :
          block.dur_us >= payload.global_thresholds.p95_us ? 1.2 : 0.8;
        const segParts = [];
        const tipParts = [
          `block=${{block.block}}`,
          `smid=${{block.smid}}`,
          `lane=${{lane}}`,
          `cta_start=${{fmt(block.start_us)}}us`,
          `cta_end=${{fmt(block.end_us)}}us`,
          `cta_dur=${{fmt(block.dur_us)}}us`,
        ];
        for (const seg of block.iter_segments) {{
          const segX = left + seg.start_us * scale;
          const segW = Math.max((seg.end_us - seg.start_us) * scale, 0.6);
          const fill = payload.iter_colors[String(seg.iter_idx)] || "#999999";
          segParts.push(`<rect x="${{segX}}" y="${{y}}" width="${{segW}}" height="${{laneHeight}}" fill="${{fill}}" opacity="0.96"></rect>`);
          tipParts.push(`iter${{seg.iter_idx}} row_base=${{seg.row_base}} start=${{fmt(seg.start_us)}}us end=${{fmt(seg.end_us)}}us dur=${{fmt(seg.dur_us)}}us`);
        }}
        parts.push(`<g><title>${{tipParts.join(" | ")}}</title>${{segParts.join("")}}<rect class="bar-outline" x="${{x0}}" y="${{y}}" width="${{totalW}}" height="${{laneHeight}}" stroke="${{outlineColor}}" stroke-width="${{outlineWidth}}"></rect></g>`);
      }}

      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      svg.innerHTML = parts.join("");
    }}

    for (const rep of payload.repeats) {{
      const opt = document.createElement("option");
      opt.value = rep.rep;
      opt.textContent = `repeat #${{rep.rep}}`;
      selectEl.appendChild(opt);
    }}
    selectEl.value = String(payload.repeats[payload.default_repeat_index].rep);
    buildLegend();
    buildMeta();
    render(payload.default_repeat_index);

    function selectByOffset(delta) {{
      const current = payload.repeats.findIndex((rep) => String(rep.rep) === selectEl.value);
      const next = Math.min(payload.repeats.length - 1, Math.max(0, current + delta));
      selectEl.value = String(payload.repeats[next].rep);
      render(next);
    }}

    selectEl.addEventListener("change", () => {{
      const idx = payload.repeats.findIndex((rep) => String(rep.rep) === selectEl.value);
      if (idx >= 0) render(idx);
    }});
    document.getElementById("prev-btn").addEventListener("click", () => selectByOffset(-1));
    document.getElementById("next-btn").addEventListener("click", () => selectByOffset(1));
  </script>
</body>
</html>
"""


def _default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    variant = _variant_trace_name(args.variant)
    stem = f"nosplit_xwarp_{variant}_cta_timeline_{args.target}_k6144_layer{args.layer_idx}_step{args.decode_step}_r{args.repeats}"
    return Path(f"results/{stem}.json"), Path(f"traces/nosplit_xwarp/{stem}.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CTA-level 2-CTA/SM timeline for k6144 xwarp async GEMV")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--target", type=str, default="down_proj", choices=["down_proj"])
    parser.add_argument("--variant", type=str, default="sms2", choices=["sms2", "sms2_balanced"])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-html", type=str, default=None)
    args = parser.parse_args()

    if args.input_json is not None:
        processed = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        metadata, trace_reps, smid_reps, row_base_reps = _collect_trace_payload(args)
        processed = _build_processed_timeline(metadata, trace_reps, smid_reps, row_base_reps)

    default_json, default_html = _default_output_paths(args)
    output_json = Path(args.output_json) if args.output_json is not None else default_json
    output_html = Path(args.output_html) if args.output_html is not None else default_html
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8")
    output_html.write_text(_build_html(processed), encoding="utf-8")

    rep = processed["repeats"][processed["default_repeat_index"]]
    print(f"[Sms2Timeline] processed JSON saved to {output_json}")
    print(f"[Sms2Timeline] HTML saved to {output_html}")
    print(
        "[Sms2Timeline] default repeat "
        f"#{rep['rep']}: kernel_span={rep['kernel_end_us']:.3f}us "
        f"cta_median={rep['cta_duration_summary']['median_us']:.3f}us "
        f"cta_max={rep['cta_duration_summary']['max_us']:.3f}us "
        f"active_sm={rep['active_smid_count']} "
        f"max_lanes_per_sm={rep['max_lanes_per_smid']}"
    )


if __name__ == "__main__":
    main()
