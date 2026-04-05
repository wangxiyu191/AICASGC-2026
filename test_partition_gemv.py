#!/usr/bin/env python3
"""Test correctness of partition-aware k=2048 GEMV kernel."""

import torch
import sys
sys.path.insert(0, ".")
import cuda_decode_gemv
from hbm_latency_partition import (
    probe_sm_classes, probe_all_chunk_classes,
    compute_interleaved_size, pack_interleaved_weight,
    _PC0_PAGES, _PC1_PAGES, _PAIRS_PER_CHUNK, _PAGES_PER_CHUNK,
    _HALVES_PER_PAGE, _HALVES_PER_ROW,
)


def test_partition_gemv(n: int, k: int = 2048, num_trials: int = 5):
    """Test partition GEMV against standard GEMV for shape [n, k]."""
    device = torch.device("cuda")
    print(f"\n{'='*60}")
    print(f"Testing partition GEMV: [{n}, {k}]")
    print(f"{'='*60}")

    # 1. Create random weight and input
    weight = torch.randn(n, k, dtype=torch.float16, device=device)
    x = torch.randn(1, k, dtype=torch.float16, device=device)

    # 2. Standard GEMV result (ground truth)
    ref_out = cuda_decode_gemv.nosplit_matvec_half(x, weight, None)
    print(f"  Standard GEMV output: mean={ref_out.float().mean():.6f}, std={ref_out.float().std():.6f}")

    # 3. Probe SM classes
    sm_result = probe_sm_classes(device=device)
    sm_class_lo = 0
    sm_class_hi = 0
    for smid, cls in sm_result.sm_class_map.items():
        if smid < 64:
            sm_class_lo |= (cls << smid)
        else:
            sm_class_hi |= (cls << (smid - 64))
    print(f"  SM classes: {len(sm_result.sm_class_0)} class-0, {len(sm_result.sm_class_1)} class-1")

    # 4. Allocate interleaved buffer
    buf_bytes = compute_interleaved_size(n, k)
    buf_halves = buf_bytes // 2
    buf = torch.zeros(buf_halves, dtype=torch.float16, device=device)
    print(f"  Buffer: {buf_bytes / (1024*1024):.1f} MB ({buf_halves} halves)")

    # 5. Probe chunk classes
    chunk_classes_results = probe_all_chunk_classes(buf, sm_result.sm_class_map)
    num_chunks = buf_bytes // (2 * 1024 * 1024)
    cc_tensor = torch.tensor(
        [chunk_classes_results[i].chunk_class for i in range(num_chunks)],
        dtype=torch.int32,
    )
    print(f"  Chunks: {num_chunks}, chunk_classes={cc_tensor.tolist()}")

    # 6. Pack interleaved weight
    pack_interleaved_weight(weight, buf, cc_tensor)
    print(f"  Packed interleaved weight.")

    # 7. Move chunk_classes to GPU
    cc_gpu = cc_tensor.to(device)

    # 8. Run partition GEMV
    for trial in range(num_trials):
        part_out = cuda_decode_gemv.nosplit_matvec_half_partition(
            x, buf, None, cc_gpu, sm_class_lo, sm_class_hi, n,
        )

        # 9. Compare
        max_diff = (ref_out.float() - part_out.float()).abs().max().item()
        mean_diff = (ref_out.float() - part_out.float()).abs().mean().item()
        
        if trial == 0:
            print(f"  Partition GEMV output: mean={part_out.float().mean():.6f}, std={part_out.float().std():.6f}")
            print(f"  Max abs diff: {max_diff:.6f}")
            print(f"  Mean abs diff: {mean_diff:.6f}")

        # For fp16, max diff should be very small
        if max_diff > 1e-2:
            print(f"  FAIL: max_diff={max_diff} > 1e-2 (trial {trial})")
            # Print first few differing positions
            diffs = (ref_out.float() - part_out.float()).abs()
            top_indices = diffs.view(-1).topk(min(5, n)).indices
            for idx in top_indices:
                i = idx.item()
                print(f"    row {i}: ref={ref_out.view(-1)[i].item():.6f}, "
                      f"part={part_out.view(-1)[i].item():.6f}, "
                      f"diff={diffs.view(-1)[i].item():.6f}")
            return False

    print(f"  PASS (max_diff={max_diff:.6f} across {num_trials} trials)")
    return True


def test_layout_consistency():
    """Verify that pack_interleaved_weight produces consistent data."""
    print(f"\n{'='*60}")
    print(f"Testing layout consistency (CPU verification)")
    print(f"{'='*60}")

    device = torch.device("cuda")
    n, k = 256, 2048  # 128 row-pairs = 1 chunk exactly

    weight = torch.randn(n, k, dtype=torch.float16, device=device)
    buf_bytes = compute_interleaved_size(n, k)
    buf = torch.zeros(buf_bytes // 2, dtype=torch.float16, device=device)

    sm_result = probe_sm_classes(device=device)
    cc_results = probe_all_chunk_classes(buf, sm_result.sm_class_map)
    cc = torch.tensor([cc_results[0].chunk_class], dtype=torch.int32)
    print(f"  chunk_class = {cc[0].item()}")

    pack_interleaved_weight(weight, buf, cc)

    # Verify: read back each row-pair from both copy locations
    errors = 0
    cc_val = cc[0].item()
    lut_a = _PC0_PAGES if cc_val == 0 else _PC1_PAGES
    lut_b = _PC1_PAGES if cc_val == 0 else _PC0_PAGES

    for pair_idx in range(n // 2):
        period = pair_idx // 16
        intra = pair_idx % 16

        page_a = period * 32 + lut_a[intra]
        page_b = period * 32 + lut_b[intra]

        # Read Copy A
        off_a = page_a * _HALVES_PER_PAGE
        row0_a = buf[off_a: off_a + _HALVES_PER_ROW]
        row1_a = buf[off_a + _HALVES_PER_ROW: off_a + _HALVES_PER_PAGE]

        # Read Copy B
        off_b = page_b * _HALVES_PER_PAGE
        row0_b = buf[off_b: off_b + _HALVES_PER_ROW]
        row1_b = buf[off_b + _HALVES_PER_ROW: off_b + _HALVES_PER_PAGE]

        # Verify Copy A matches original
        r0 = pair_idx * 2
        r1 = pair_idx * 2 + 1
        if not torch.equal(row0_a, weight[r0]):
            errors += 1
            print(f"  MISMATCH Copy A row {r0}")
        if not torch.equal(row1_a, weight[r1]):
            errors += 1
            print(f"  MISMATCH Copy A row {r1}")
        # Verify Copy B matches original (same data)
        if not torch.equal(row0_b, weight[r0]):
            errors += 1
            print(f"  MISMATCH Copy B row {r0}")
        if not torch.equal(row1_b, weight[r1]):
            errors += 1
            print(f"  MISMATCH Copy B row {r1}")

    if errors == 0:
        print(f"  PASS: all {n} rows verified in both copies")
    else:
        print(f"  FAIL: {errors} mismatches")
    return errors == 0


if __name__ == "__main__":
    torch.manual_seed(42)
    results = []

    # Test layout consistency first
    results.append(("layout_consistency", test_layout_consistency()))

    # Test various shapes
    for n in [2048, 6144, 12288]:
        results.append((f"gemv_{n}x2048", test_partition_gemv(n, 2048)))

    print(f"\n{'='*60}")
    print("Summary:")
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    all_pass = all(p for _, p in results)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
