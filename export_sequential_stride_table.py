#!/usr/bin/env python3
"""Export sequential-stride latency JSON into an address x smid CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export sequential-stride latency JSON to CSV")
    parser.add_argument("input", type=str, help="Path to bench_sequential_stride_latency JSON")
    parser.add_argument(
        "--metric",
        type=str,
        default="cycles",
        choices=("cycles", "ns_approx"),
        help="Which metric to export",
    )
    parser.add_argument("--repeat", type=int, default=0, help="Repeat index to export")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path; defaults to alongside input with metric suffix",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text())
    repeats = data["repeats"]
    if args.repeat < 0 or args.repeat >= len(repeats):
        raise ValueError(f"repeat {args.repeat} out of range [0, {len(repeats)})")

    repeat_entry = repeats[args.repeat]
    blocks = sorted(repeat_entry["per_block"], key=lambda block: block["smid"])
    if not blocks:
        raise ValueError("no per_block entries found")

    chain_meta = data["metadata"]["chain_meta"]
    stride_bytes = int(chain_meta["stride_bytes"])
    base_offset_bytes = int(chain_meta.get("base_offset_bytes", 0))
    chain_len = int(chain_meta["chain_len"])
    page_bytes = int(data["metadata"]["page_bytes"])

    metric_key = "latency_cycles" if args.metric == "cycles" else "latency_ns_approx"
    for block in blocks:
        if len(block[metric_key]) != chain_len:
            raise ValueError(f"smid {block['smid']} has unexpected {metric_key} length")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_repeat{args.repeat}_{args.metric}_table.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    smids = [int(block["smid"]) for block in blocks]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "access_idx",
                "addr_offset_bytes",
                "page_idx_8kb",
                "offset_in_8kb_bytes",
                *[f"smid_{smid}" for smid in smids],
            ]
        )
        for access_idx in range(chain_len):
            addr_offset_bytes = base_offset_bytes + access_idx * stride_bytes
            page_idx = addr_offset_bytes // page_bytes
            offset_in_page_bytes = addr_offset_bytes % page_bytes
            writer.writerow(
                [
                    access_idx,
                    addr_offset_bytes,
                    page_idx,
                    offset_in_page_bytes,
                    *[block[metric_key][access_idx] for block in blocks],
                ]
            )

    print(f"[metric] {args.metric}")
    print(f"[repeat] {args.repeat}")
    print(f"[rows] {chain_len}")
    print(f"[smids] {len(smids)}")
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
