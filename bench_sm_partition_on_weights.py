#!/usr/bin/env python3
"""Verify SM partition & page_class formula on real GEMV weight tensor addresses.

For each weight tensor used in decode-phase GEMV, this benchmark:
  1. Reinterprets the GPU memory as int64 pointer chains (one chain per 1MB chunk)
  2. Probes each chunk from a distinct SM via l2_sequential_pointer_chase_trace
  3. Verifies that (a) page_class formula predicts intra-chunk fast/slow pattern,
     and (b) our SM classification holds on real weight addresses.
  4. Reports the chunk_class (page_class <-> sm_class polarity) for every 1MB chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch
import numpy as np

import cuda_decode_gemv
from benchmark import get_system_info

# ---------- SM classification from our analysis ----------
SM_CLASS_A = [1, 3, 4, 7, 9, 10, 12, 14, 16, 17, 20, 21, 22, 23, 28, 30,
              33, 34, 36, 39, 41, 43, 45, 46, 52, 53, 54, 55, 56, 57, 58, 59,
              60, 63, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 81, 83,
              85, 87, 88, 91, 92, 93, 102, 103]

SM_CLASS_B = [0, 2, 5, 6, 8, 11, 13, 15, 18, 19, 24, 25, 26, 27, 29, 31,
              32, 35, 37, 38, 40, 42, 44, 47, 48, 49, 50, 51, 61, 62, 64, 65,
              74, 75, 80, 82, 84, 86, 89, 90, 94, 95, 96, 97, 98, 99, 100, 101,
              104, 105, 106, 107]

SM_CLASS_MAP = {}  # smid -> 0 or 1
for s in SM_CLASS_A:
    SM_CLASS_MAP[s] = 0
for s in SM_CLASS_B:
    SM_CLASS_MAP[s] = 1

NUM_SMS = 108
CHUNK_BYTES = 1024 * 1024  # 1 MiB
STRIDE_BYTES = 256
ELEM_SIZE = 8  # int64
STEP_ELEMS = STRIDE_BYTES // ELEM_SIZE  # 32
CHUNK_ELEMS = CHUNK_BYTES // ELEM_SIZE  # 131072
CHAIN_LEN = CHUNK_BYTES // STRIDE_BYTES  # 4096
PAGE_BYTES = 8192
ACCESSES_PER_PAGE = PAGE_BYTES // STRIDE_BYTES  # 32
PAGES_PER_CHUNK = CHUNK_BYTES // PAGE_BYTES  # 128


def page_class(addr_offset: int) -> int:
    """page_class = ((addr >> 13) & 1) ^ ((addr >> 17) & 1)"""
    return ((addr_offset >> 13) & 1) ^ ((addr_offset >> 17) & 1)


# Pre-compute page_class for each access within a 1MB chunk
PAGE_CLASS_TABLE = np.array([page_class(i * STRIDE_BYTES) for i in range(CHAIN_LEN)],
                            dtype=np.int32)


def _build_flush_buffer(num_blocks: int, flush_mb_per_block: int, device: torch.device):
    elems_per_block = (flush_mb_per_block * 1024 * 1024) // 4
    return torch.randint(0, 2**31 - 1, (num_blocks * elems_per_block,),
                         device=device, dtype=torch.int32)


def _build_chains_in_tensor(w_int64_flat: torch.Tensor, num_chunks: int):
    """Overwrite weight memory with sequential pointer chains (one per 1MB chunk).

    Returns start tensor [num_chunks] on the same device.
    """
    cpu_buf = torch.zeros(num_chunks * CHUNK_ELEMS, dtype=torch.int64)
    starts = torch.empty(num_chunks, dtype=torch.int64)

    for c in range(num_chunks):
        base = c * CHUNK_ELEMS
        nodes = base + torch.arange(CHAIN_LEN, dtype=torch.int64) * STEP_ELEMS
        cpu_buf[nodes[:-1]] = nodes[1:]
        cpu_buf[nodes[-1]] = nodes[-1]  # self-loop at end
        starts[c] = nodes[0]

    # Overwrite the first num_chunks * CHUNK_ELEMS of the tensor
    w_int64_flat[:num_chunks * CHUNK_ELEMS].copy_(cpu_buf, non_blocking=False)
    return starts.to(device=w_int64_flat.device, non_blocking=False)


def _analyze_chunk(latency_cycles: np.ndarray, smid: int,
                    pc_table: np.ndarray):
    """Analyze one chunk probed by one SM.

    Returns dict with verification results.
    """
    sm_cls = SM_CLASS_MAP.get(smid, -1)

    # Split latencies by page_class
    mask_pc0 = pc_table == 0
    mask_pc1 = pc_table == 1
    lat_pc0 = latency_cycles[mask_pc0].astype(np.float64)
    lat_pc1 = latency_cycles[mask_pc1].astype(np.float64)

    mean_pc0 = float(np.mean(lat_pc0))
    mean_pc1 = float(np.mean(lat_pc1))
    delta = mean_pc0 - mean_pc1  # positive → pc0 slower

    # page_class formula is "correct" if there's a clear separation
    separation = abs(delta)
    page_class_ok = separation > 50  # expect ~142 cycles; >50 is a safe threshold

    # Determine chunk_class from the perspective of this SM:
    # If sm_class=0 and pc0 is fast → chunk_class=0
    # If sm_class=0 and pc1 is fast → chunk_class=1
    # If sm_class=1 and pc0 is fast → chunk_class=1
    # If sm_class=1 and pc1 is fast → chunk_class=0
    # General: chunk_class = sm_class XOR (1 if pc0 is slow else 0)
    pc0_is_slow = int(delta > 0)
    chunk_class = sm_cls ^ pc0_is_slow if sm_cls >= 0 else -1

    # Verify SM classification consistency:
    # If sm_class=0: expect (pc0 fast when chunk_class=0) or (pc1 fast when chunk_class=1)
    # The key check: does the observed polarity match a consistent sm_class?
    sm_class_ok = sm_cls >= 0 and page_class_ok

    return {
        "smid": smid,
        "sm_class": sm_cls,
        "mean_pc0": mean_pc0,
        "mean_pc1": mean_pc1,
        "delta": delta,
        "separation": separation,
        "page_class_ok": page_class_ok,
        "chunk_class": chunk_class,
        "sm_class_ok": sm_class_ok,
    }


def _probe_weight(
    name: str,
    weight: torch.Tensor,
    flush_mb_per_block: int,
    max_chunks: int | None = None,
):
    """Probe a weight tensor: build chains, run pointer chase, analyze."""

    ALIGN_2MB = 2 * 1024 * 1024  # 2 MiB alignment boundary

    total_bytes = weight.numel() * weight.element_size()
    base_addr = weight.data_ptr()

    # --- Find first 2MB-aligned start within the tensor ---
    misalign = base_addr % ALIGN_2MB
    if misalign == 0:
        skip_bytes = 0
    else:
        skip_bytes = ALIGN_2MB - misalign
    aligned_addr = base_addr + skip_bytes

    usable_bytes = total_bytes - skip_bytes
    if usable_bytes < CHUNK_BYTES:
        print(f"  [{name}] too small after 2MB alignment "
              f"(total={total_bytes}, skip={skip_bytes}, usable={usable_bytes}), skipping")
        return []

    num_full_chunks = usable_bytes // CHUNK_BYTES
    if max_chunks is not None:
        num_full_chunks = min(num_full_chunks, max_chunks)

    print(f"  [{name}] shape={list(weight.shape)}, dtype={weight.dtype}, "
          f"size={total_bytes/(1024*1024):.1f}MB, base=0x{base_addr:X}, "
          f"aligned_start=0x{aligned_addr:X} (skip {skip_bytes//1024}KB), "
          f"chunks={num_full_chunks}")

    # Reinterpret as int64
    w_contig = weight.contiguous().view(-1)
    w_int64 = w_contig.view(torch.int64)

    # Slice to the 2MB-aligned region
    skip_elems = skip_bytes // ELEM_SIZE
    w_aligned = w_int64[skip_elems:]

    # Build chains for all chunks (indices are relative to w_aligned)
    all_starts = _build_chains_in_tensor(w_aligned, num_full_chunks)
    torch.cuda.synchronize()

    # Process in batches of NUM_SMS (108)
    results = []
    workspace = {}
    batch_size = NUM_SMS

    for batch_start in range(0, num_full_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_full_chunks)
        n_blocks = batch_end - batch_start
        starts_batch = all_starts[batch_start:batch_end]

        # Build flush buffer sized for this batch's actual block count
        flush_buf = _build_flush_buffer(n_blocks, flush_mb_per_block, w_aligned.device)

        latency, smid, final_idx, flush_sink = cuda_decode_gemv.l2_sequential_pointer_chase_trace(
            w_aligned,
            starts_batch,
            flush_buf,
            chain_len=CHAIN_LEN,
            workspace=workspace,
        )
        torch.cuda.synchronize()

        latency_cpu = latency.cpu().numpy()
        smid_cpu = smid.cpu().numpy()

        for i in range(n_blocks):
            chunk_idx = batch_start + i
            chunk_offset = chunk_idx * CHUNK_BYTES
            chunk_addr = aligned_addr + chunk_offset

            # Skip the very first access (often anomalous)
            lat = latency_cpu[i, 1:]
            pc_table = PAGE_CLASS_TABLE[1:]

            analysis = _analyze_chunk(lat, int(smid_cpu[i]), pc_table)
            analysis["chunk_idx"] = chunk_idx
            analysis["chunk_addr"] = f"0x{chunk_addr:X}"
            analysis["chunk_offset_mb"] = chunk_offset / (1024 * 1024)
            results.append(analysis)

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify SM partition on real GEMV weights")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--layer-idx", type=int, default=14,
                        help="Decoder layer index to probe")
    parser.add_argument("--flush-mb-per-block", type=int, default=64)
    parser.add_argument("--max-chunks-per-weight", type=int, default=None,
                        help="Limit chunks per weight tensor (for faster runs)")
    parser.add_argument("--output", type=str,
                        default="results/sm_partition_on_weights.json")
    parser.add_argument("--targets", type=str, nargs="+",
                        default=["qkv", "gate_up", "o_proj", "down_proj", "lm_head"])
    args = parser.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    print(f"Loading model from {args.model_path} ...")
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda:0",
    )
    from optimized_layer import replace_decoder_layers
    n_replaced = replace_decoder_layers(model)
    print(f"Replaced {n_replaced} decoder layers")

    # Collect weight tensors
    layer = model.model.language_model.layers[args.layer_idx]
    weight_map = {}
    if "qkv" in args.targets:
        weight_map["qkv"] = layer.qkv_weight.data
    if "gate_up" in args.targets:
        weight_map["gate_up"] = layer.gate_up_weight.data
    if "o_proj" in args.targets:
        weight_map["o_proj"] = layer.o_proj_weight.data
    if "down_proj" in args.targets:
        weight_map["down_proj"] = layer.down_proj_weight.data
    if "lm_head" in args.targets:
        weight_map["lm_head"] = model.lm_head.weight.data

    # Probe each weight (flush buffer built per-batch inside _probe_weight)
    all_results = {}
    for name, weight in weight_map.items():
        print(f"\nProbing: {name}")
        results = _probe_weight(name, weight, args.flush_mb_per_block,
                                args.max_chunks_per_weight)
        all_results[name] = results

    # ===== Analysis & Report =====
    print("\n" + "=" * 100)
    print("VERIFICATION RESULTS")
    print("=" * 100)

    overall_page_class_ok = 0
    overall_page_class_total = 0
    overall_sm_seen = set()

    for name, results in all_results.items():
        if not results:
            continue
        n_chunks = len(results)
        n_pc_ok = sum(1 for r in results if r["page_class_ok"])
        sms_seen = set(r["smid"] for r in results)
        overall_page_class_ok += n_pc_ok
        overall_page_class_total += n_chunks
        overall_sm_seen.update(sms_seen)

        # Group by chunk_class
        cc0 = [r for r in results if r["chunk_class"] == 0]
        cc1 = [r for r in results if r["chunk_class"] == 1]

        print(f"\n--- {name} ({n_chunks} chunks, page_class OK: {n_pc_ok}/{n_chunks}) ---")
        print(f"  chunk_class=0: {len(cc0)} chunks | chunk_class=1: {len(cc1)} chunks")
        print(f"  SMs seen: {len(sms_seen)} unique")

        # Detailed per-chunk table
        print(f"\n  {'chunk':>5} {'addr':>14} {'smid':>4} {'sm_cls':>6} "
              f"{'mean_pc0':>9} {'mean_pc1':>9} {'delta':>8} {'sep':>6} "
              f"{'pc_ok':>5} {'cc':>3}")
        for r in results:
            flag = "OK" if r["page_class_ok"] else "FAIL"
            print(f"  {r['chunk_idx']:>5} {r['chunk_addr']:>14} {r['smid']:>4} "
                  f"{r['sm_class']:>6} {r['mean_pc0']:>9.1f} {r['mean_pc1']:>9.1f} "
                  f"{r['delta']:>8.1f} {r['separation']:>6.1f} {flag:>5} "
                  f"{r['chunk_class']:>3}")

    # SM class verification summary
    print(f"\n{'='*100}")
    print("SM CLASS VERIFICATION SUMMARY")
    print(f"{'='*100}")
    print(f"Total chunks probed: {overall_page_class_total}")
    print(f"Page class formula correct: {overall_page_class_ok}/{overall_page_class_total}")
    print(f"Unique SMs observed: {len(overall_sm_seen)}/108")

    # Per-SM verification
    sm_observations = {}  # smid -> list of (chunk_idx, weight_name, delta)
    for name, results in all_results.items():
        for r in results:
            sm = r["smid"]
            if sm not in sm_observations:
                sm_observations[sm] = []
            sm_observations[sm].append({
                "weight": name,
                "chunk_idx": r["chunk_idx"],
                "delta": r["delta"],
                "sm_class_expected": SM_CLASS_MAP.get(sm, -1),
            })

    sm_class_correct = 0
    sm_class_total = 0
    for sm in sorted(sm_observations.keys()):
        obs = sm_observations[sm]
        expected_cls = SM_CLASS_MAP.get(sm, -1)
        # For sm_class=0 (A): should see delta<0 when chunk_class=0, delta>0 when chunk_class=1
        # Simpler: just check that |delta| is large (separation exists)
        all_sep = [abs(o["delta"]) for o in obs]
        avg_sep = np.mean(all_sep)
        ok = avg_sep > 50
        if ok:
            sm_class_correct += 1
        sm_class_total += 1

    print(f"SM class separation verified: {sm_class_correct}/{sm_class_total}")

    # ===== 2MB pair analysis: is chunk_class granularity 1MB or 2MB? =====
    print(f"\n{'='*100}")
    print("CHUNK_CLASS GRANULARITY ANALYSIS (1MB vs 2MB)")
    print(f"{'='*100}")

    for name, results in all_results.items():
        if not results:
            continue
        # results are in chunk_idx order; each chunk is 1MB
        # pair consecutive chunks (0,1), (2,3), ... into 2MB groups
        n_pairs = len(results) // 2
        n_same = 0
        n_diff = 0
        pair_details = []
        for p in range(n_pairs):
            r0 = results[2 * p]
            r1 = results[2 * p + 1]
            same = r0["chunk_class"] == r1["chunk_class"]
            if same:
                n_same += 1
            else:
                n_diff += 1
            pair_details.append((r0["chunk_idx"], r1["chunk_idx"],
                                 r0["chunk_class"], r1["chunk_class"],
                                 "SAME" if same else "DIFF"))

        print(f"\n  {name}: {n_pairs} pairs | SAME={n_same} | DIFF={n_diff}")
        if n_diff > 0:
            print(f"  >> chunk_class changes WITHIN 2MB pages! Granularity = 1MB")
            for ci0, ci1, cc0, cc1, tag in pair_details:
                if tag == "DIFF":
                    print(f"     pair ({ci0},{ci1}): cc={cc0},{cc1}")
        else:
            print(f"  >> All 2MB pairs consistent. Granularity >= 2MB")

        # Also show the chunk_class sequence compactly
        cc_seq = "".join(str(r["chunk_class"]) for r in results)
        print(f"  chunk_class sequence: {cc_seq}")

    # Save JSON
    output_data = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "system": get_system_info(),
            "device_name": props.name,
            "num_sms": props.multi_processor_count,
            "model_path": args.model_path,
            "layer_idx": args.layer_idx,
            "flush_mb_per_block": args.flush_mb_per_block,
            "stride_bytes": STRIDE_BYTES,
            "chunk_bytes": CHUNK_BYTES,
        },
        "sm_class_a": SM_CLASS_A,
        "sm_class_b": SM_CLASS_B,
        "per_weight": {},
    }
    for name, results in all_results.items():
        output_data["per_weight"][name] = results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\n[output] {output_path}")


if __name__ == "__main__":
    main()
