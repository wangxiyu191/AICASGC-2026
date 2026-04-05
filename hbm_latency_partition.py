"""HBM latency partition utilities for A800.

Provides two core capabilities:
  1. probe_sm_classes()  -- Determine which SM IDs belong to sm_class 0 vs 1.
  2. probe_chunk_class() -- For a given 2MiB-aligned GPU memory region,
                            determine its chunk_class (the page_class <-> sm_class
                            fast/slow polarity).

Background
----------
A800 HBM cold-miss latency follows a three-layer XOR checkerboard:

    latency_class(smid, addr) = sm_class(smid) XOR page_class(addr) XOR chunk_class(addr)

    latency_class == 0  →  fast path  (~451 cycles)
    latency_class == 1  →  slow path  (~593 cycles)

    page_class(addr) = ((addr >> 13) & 1) ^ ((addr >> 17) & 1)   (8 KB granularity)
    sm_class(smid)   ∈ {0, 1}   (fixed per SM, 56 vs 52 split on A800-108SM)
    chunk_class(addr) ∈ {0, 1}  (2 MiB granularity, depends on physical page mapping)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import torch
import numpy as np

import cuda_decode_gemv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHUNK_2MIB = 2 * 1024 * 1024        # 2 MiB -- chunk_class granularity
_CHUNK_1MIB = 1 * 1024 * 1024        # 1 MiB -- probe unit (half of 2MiB)
_STRIDE_BYTES = 256
_ELEM_SIZE = 8                        # int64
_STEP_ELEMS = _STRIDE_BYTES // _ELEM_SIZE          # 32
_CHAIN_LEN_1MIB = _CHUNK_1MIB // _STRIDE_BYTES     # 4096
_ELEMS_1MIB = _CHUNK_1MIB // _ELEM_SIZE            # 131072
_PAGE_BYTES = 8192
_DEFAULT_FLUSH_MB = 64
_SEPARATION_THRESHOLD = 50.0   # cycles; expect ~142, anything >50 is clear


def page_class(addr_offset: int) -> int:
    """Compute page_class from an address offset within a 2MiB chunk.

    ``page_class = ((addr >> 13) & 1) ^ ((addr >> 17) & 1)``
    """
    return ((addr_offset >> 13) & 1) ^ ((addr_offset >> 17) & 1)


# Pre-computed page_class for every 256B access in a 1MiB span.
_PAGE_CLASS_TABLE = np.array(
    [page_class(i * _STRIDE_BYTES) for i in range(_CHAIN_LEN_1MIB)],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class SmClassResult:
    """Result of :func:`probe_sm_classes`."""
    sm_class_0: List[int]       # SM IDs with sm_class == 0
    sm_class_1: List[int]       # SM IDs with sm_class == 1
    sm_class_map: Dict[int, int]  # smid -> 0 or 1
    num_sms: int
    mean_separation: float      # average |delta| in cycles


@dataclass
class ChunkClassResult:
    """Result of :func:`probe_chunk_class`."""
    chunk_class: int            # 0 or 1
    mean_fast: float            # mean latency on the fast pages (cycles)
    mean_slow: float            # mean latency on the slow pages (cycles)
    separation: float           # |mean_slow - mean_fast|  (cycles)
    probed_smid: int            # which SM actually ran the probe
    probed_sm_class: int        # sm_class of that SM


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_flush_buffer(num_blocks: int, flush_mb: int,
                        device: torch.device) -> torch.Tensor:
    elems = num_blocks * (flush_mb * 1024 * 1024 // 4)
    return torch.randint(0, 2**31 - 1, (elems,),
                         device=device, dtype=torch.int32)


def _build_sequential_chains(buf_int64: torch.Tensor, num_chunks: int):
    """Write sequential pointer chains into *buf_int64* (one per 1MiB chunk).

    Returns a ``start`` tensor on the same device.
    """
    cpu = torch.zeros(num_chunks * _ELEMS_1MIB, dtype=torch.int64)
    starts = torch.empty(num_chunks, dtype=torch.int64)
    for c in range(num_chunks):
        base = c * _ELEMS_1MIB
        nodes = base + torch.arange(_CHAIN_LEN_1MIB, dtype=torch.int64) * _STEP_ELEMS
        cpu[nodes[:-1]] = nodes[1:]
        cpu[nodes[-1]] = nodes[-1]
        starts[c] = nodes[0]
    buf_int64[:num_chunks * _ELEMS_1MIB].copy_(cpu, non_blocking=False)
    return starts.to(device=buf_int64.device, non_blocking=False)


def _analyse_latency(latency_1d: np.ndarray, pc_table: np.ndarray):
    """Return (mean_pc0, mean_pc1) from one 1MiB trace."""
    m0 = pc_table == 0
    m1 = pc_table == 1
    return float(np.mean(latency_1d[m0])), float(np.mean(latency_1d[m1]))


# ---------------------------------------------------------------------------
# Public API  (1) -- probe SM classes
# ---------------------------------------------------------------------------
def _run_one_probe_pass(buf_int64: torch.Tensor, num_sms: int,
                        flush_mb: int):
    """Run a single pointer-chase probe on *num_sms* contiguous 1MiB regions.

    Returns (signs, smids, deltas) where:
      signs[block_idx] = 0 if delta<0, else 1
      smids[block_idx] = SM ID that ran that block
      deltas[block_idx] = raw delta value
    """
    starts = _build_sequential_chains(buf_int64, num_sms)
    torch.cuda.synchronize()
    flush_buf = _build_flush_buffer(num_sms, flush_mb, buf_int64.device)
    latency, smid_t, _, _ = cuda_decode_gemv.l2_sequential_pointer_chase_trace(
        buf_int64, starts, flush_buf,
        chain_len=_CHAIN_LEN_1MIB, workspace={},
    )
    torch.cuda.synchronize()

    lat_cpu = latency.cpu().numpy()[:, 1:]
    smid_cpu = smid_t.cpu().numpy()
    pc = _PAGE_CLASS_TABLE[1:]

    signs = {}
    smids = {}
    deltas = {}
    for i in range(num_sms):
        m0, m1 = _analyse_latency(lat_cpu[i], pc)
        d = m0 - m1
        smids[i] = int(smid_cpu[i])
        signs[i] = 0 if d < 0 else 1
        deltas[i] = d
    return signs, smids, deltas


def _compute_chunk_pairs(base_addr: int, num_sms: int,
                         offset_mib: int = 0):
    """Return list of (block_idx_a, block_idx_b) sharing a 2MiB chunk."""
    chunk_to_idxs: dict[int, list[int]] = {}
    for i in range(num_sms):
        addr = base_addr + (i + offset_mib) * _CHUNK_1MIB
        chunk = addr // _CHUNK_2MIB
        chunk_to_idxs.setdefault(chunk, []).append(i)
    pairs = []
    for idxs in chunk_to_idxs.values():
        if len(idxs) == 2:
            pairs.append(tuple(idxs))
    return pairs


def probe_sm_classes(
    *,
    device: torch.device | str = "cuda",
    flush_mb: int = _DEFAULT_FLUSH_MB,
) -> SmClassResult:
    """Probe all SMs to determine their sm_class (0 or 1).

    Uses a **two-pass** approach to eliminate chunk_class interference:
    within each 2MiB chunk two SMs share the same chunk_class, so XOR-ing
    their delta signs gives ``sm_class(a) XOR sm_class(b)`` (chunk_class
    cancels).  Two passes with a 1MiB offset produce overlapping pairs
    that form a connected chain across all SMs, enabling full propagation.

    Parameters
    ----------
    device : torch.device or str
        CUDA device.
    flush_mb : int
        Megabytes of L2-flush buffer per block.

    Returns
    -------
    SmClassResult
    """
    from collections import deque

    device = torch.device(device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    # Allocate (num_sms + 1) x 1MiB so we can do two offset passes.
    buf = torch.empty((num_sms + 1) * _ELEMS_1MIB, dtype=torch.int64,
                       device=device)
    base_addr = buf.data_ptr()

    # ---- Pass 1: chains at offsets [0, num_sms) MiB ----
    buf_p1 = buf[:num_sms * _ELEMS_1MIB]
    signs1, smids1, deltas1 = _run_one_probe_pass(buf_p1, num_sms, flush_mb)

    # ---- Pass 2: chains at offsets [1, num_sms+1) MiB ----
    buf_p2 = buf[_ELEMS_1MIB:]
    signs2, smids2, deltas2 = _run_one_probe_pass(buf_p2, num_sms, flush_mb)

    # ---- Build pair constraints from both passes ----
    pairs1 = _compute_chunk_pairs(base_addr, num_sms, offset_mib=0)
    pairs2 = _compute_chunk_pairs(base_addr, num_sms, offset_mib=1)

    # Adjacency on SM IDs: (neighbor_smid, relative_class)
    adj: dict[int, list[tuple[int, int]]] = {}
    all_smids = set()

    for a, b in pairs1:
        sa, sb = smids1[a], smids1[b]
        rel = signs1[a] ^ signs1[b]  # 0 = same sm_class, 1 = different
        adj.setdefault(sa, []).append((sb, rel))
        adj.setdefault(sb, []).append((sa, rel))
        all_smids.update([sa, sb])

    for a, b in pairs2:
        sa, sb = smids2[a], smids2[b]
        rel = signs2[a] ^ signs2[b]
        adj.setdefault(sa, []).append((sb, rel))
        adj.setdefault(sb, []).append((sa, rel))
        all_smids.update([sa, sb])

    # ---- BFS to propagate relative sm_class ----
    relative_cls: dict[int, int] = {}
    start_smid = smids1[0]
    relative_cls[start_smid] = 0
    queue: deque[int] = deque([start_smid])

    while queue:
        curr = queue.popleft()
        for neighbor, rel in adj.get(curr, []):
            if neighbor not in relative_cls:
                relative_cls[neighbor] = relative_cls[curr] ^ rel
                queue.append(neighbor)

    # Handle any SMs not reached by BFS (shouldn't happen but be safe).
    for smid in all_smids:
        if smid not in relative_cls:
            # Fallback: use pass1 sign directly (old behaviour).
            for i in range(num_sms):
                if smids1[i] == smid:
                    relative_cls[smid] = signs1[i]
                    break

    # ---- Majority convention: larger group = sm_class 0 ----
    grp0 = sorted(s for s, c in relative_cls.items() if c == 0)
    grp1 = sorted(s for s, c in relative_cls.items() if c == 1)

    if len(grp1) > len(grp0):
        # Flip so majority is class 0.
        grp0, grp1 = grp1, grp0
        relative_cls = {s: 1 - c for s, c in relative_cls.items()}

    sm_class_map: dict[int, int] = dict(relative_cls)

    # Mean separation from pass 1 raw deltas.
    mean_sep = float(np.mean([abs(v) for v in deltas1.values()]))
    return SmClassResult(
        sm_class_0=sorted(grp0),
        sm_class_1=sorted(grp1),
        sm_class_map=sm_class_map,
        num_sms=num_sms,
        mean_separation=mean_sep,
    )


# ---------------------------------------------------------------------------
# Public API  (2) -- probe chunk_class for one 2MiB region
# ---------------------------------------------------------------------------
def probe_chunk_class(
    addr: int,
    sm_class_map: Dict[int, int],
    *,
    device: torch.device | str = "cuda",
    flush_mb: int = _DEFAULT_FLUSH_MB,
) -> ChunkClassResult:
    """Probe the chunk_class of the 2MiB region containing *addr*.

    The address must already be accessible on *device* (i.e. a valid GPU
    pointer, typically from ``tensor.data_ptr()``).  The caller must ensure
    the region ``[aligned_addr, aligned_addr + 2MiB)`` is fully allocated.

    .. warning:: This function **overwrites** the first 1MiB of the aligned
       region with pointer-chain data.

    Parameters
    ----------
    addr : int
        Any GPU virtual address inside the target 2MiB region.
    sm_class_map : dict[int, int]
        Mapping ``smid -> 0|1``, as returned by :func:`probe_sm_classes`.
    device : torch.device or str
        CUDA device.
    flush_mb : int
        Megabytes of L2-flush buffer per block.

    Returns
    -------
    ChunkClassResult
    """
    device = torch.device(device)

    # Align down to 2MiB boundary.
    aligned = addr & ~(_CHUNK_2MIB - 1)
    # Create a tensor view over the first 1MiB of this 2MiB page.
    buf = torch.tensor([], dtype=torch.int64, device=device)
    buf.set_(
        torch.Storage._new_shared_cuda(
            device.index if device.index is not None else 0,
            _ELEMS_1MIB * _ELEM_SIZE,
            allocator=None,
        ),
    )
    # ^ We cannot safely manufacture a tensor from a raw pointer in pure
    #   PyTorch.  Instead we use a small helper that builds the chain in a
    #   *separate* buffer, and then copies it into the target memory.  But
    #   that defeats the purpose (the physical address would differ).
    #
    # Practical approach: the caller should pass a tensor that covers the
    # 2MiB region; we provide a convenience wrapper below.

    # --- fall through to the tensor-based implementation ---
    raise NotImplementedError(
        "Use probe_chunk_class_from_tensor() for the tensor-based API."
    )


def probe_chunk_class_from_tensor(
    tensor: torch.Tensor,
    sm_class_map: Dict[int, int],
    *,
    offset_bytes: int = 0,
    flush_mb: int = _DEFAULT_FLUSH_MB,
) -> ChunkClassResult:
    """Probe chunk_class using a live GPU tensor's memory.

    Finds the first 2MiB-aligned address at or after
    ``tensor.data_ptr() + offset_bytes``, overwrites the first 1MiB of that
    2MiB page with a pointer chain, and probes it from a single SM.

    Parameters
    ----------
    tensor : torch.Tensor
        A contiguous CUDA tensor whose memory covers the region of interest.
    sm_class_map : dict[int, int]
        Mapping ``smid -> 0|1``.
    offset_bytes : int
        Extra byte offset into *tensor* before searching for 2MiB alignment.
    flush_mb : int
        Megabytes of L2-flush buffer.

    Returns
    -------
    ChunkClassResult
    """
    assert tensor.is_cuda and tensor.is_contiguous()
    base = tensor.data_ptr() + offset_bytes
    total = tensor.numel() * tensor.element_size() - offset_bytes

    # Find 2MiB-aligned start.
    misalign = base % _CHUNK_2MIB
    skip = ((_CHUNK_2MIB - misalign) % _CHUNK_2MIB)
    aligned = base + skip
    usable = total - skip
    if usable < _CHUNK_2MIB:
        raise ValueError(
            f"Not enough space for a full 2MiB region after alignment "
            f"(usable={usable}, need={_CHUNK_2MIB})"
        )

    # View as int64 starting from the aligned address.
    skip_elems = skip // _ELEM_SIZE
    flat = tensor.contiguous().view(-1).view(torch.int64)
    start_elem = skip_elems + (offset_bytes // _ELEM_SIZE)
    buf = flat[start_elem: start_elem + _ELEMS_1MIB]

    # Build chain & probe with 1 block.
    _build_sequential_chains(buf, 1)
    torch.cuda.synchronize()

    starts = torch.zeros(1, dtype=torch.int64, device=tensor.device)
    flush_buf = _build_flush_buffer(1, flush_mb, tensor.device)

    latency, smid_t, _, _ = cuda_decode_gemv.l2_sequential_pointer_chase_trace(
        buf, starts, flush_buf, chain_len=_CHAIN_LEN_1MIB,
    )
    torch.cuda.synchronize()

    lat = latency.cpu().numpy()[0, 1:]
    pc = _PAGE_CLASS_TABLE[1:]
    smid = int(smid_t.cpu().item())
    sm_cls = sm_class_map.get(smid, -1)

    mean_pc0, mean_pc1 = _analyse_latency(lat, pc)
    delta = mean_pc0 - mean_pc1
    pc0_is_slow = int(delta > 0)

    # chunk_class = sm_class XOR (1 if pc0 is slow else 0)
    chunk_cls = sm_cls ^ pc0_is_slow if sm_cls >= 0 else -1

    fast = min(mean_pc0, mean_pc1)
    slow = max(mean_pc0, mean_pc1)

    return ChunkClassResult(
        chunk_class=chunk_cls,
        mean_fast=fast,
        mean_slow=slow,
        separation=slow - fast,
        probed_smid=smid,
        probed_sm_class=sm_cls,
    )


# ---------------------------------------------------------------------------
# Convenience: batch probe all 2MiB chunks in a tensor
# ---------------------------------------------------------------------------
def probe_all_chunk_classes(
    tensor: torch.Tensor,
    sm_class_map: Dict[int, int],
    *,
    flush_mb: int = _DEFAULT_FLUSH_MB,
    max_blocks_per_batch: int = 108,
) -> List[ChunkClassResult]:
    """Probe chunk_class for every 2MiB-aligned region in *tensor*.

    Parameters
    ----------
    tensor : torch.Tensor
        Contiguous CUDA tensor.
    sm_class_map : dict[int, int]
        Mapping ``smid -> 0|1``.
    flush_mb : int
        L2-flush MB per block.
    max_blocks_per_batch : int
        Max concurrent blocks (default 108 = A800 SM count).

    Returns
    -------
    list[ChunkClassResult]
        One entry per 2MiB chunk, in address order.
    """
    assert tensor.is_cuda and tensor.is_contiguous()
    total_bytes = tensor.numel() * tensor.element_size()
    base = tensor.data_ptr()

    misalign = base % _CHUNK_2MIB
    skip = ((_CHUNK_2MIB - misalign) % _CHUNK_2MIB)
    usable = total_bytes - skip
    # Each 2MiB chunk is probed via its first 1MiB half.
    num_chunks_2mib = usable // _CHUNK_2MIB

    if num_chunks_2mib == 0:
        return []

    # Reinterpret as int64, offset to aligned start.
    flat = tensor.contiguous().view(-1).view(torch.int64)
    skip_elems = skip // _ELEM_SIZE
    aligned_buf = flat[skip_elems:]

    # Build chains for ALL first-half 1MiB regions (stride = 2MiB = 2*_ELEMS_1MIB).
    cpu = torch.zeros(num_chunks_2mib * 2 * _ELEMS_1MIB, dtype=torch.int64)
    starts_all = torch.empty(num_chunks_2mib, dtype=torch.int64)
    for c in range(num_chunks_2mib):
        region_base = c * 2 * _ELEMS_1MIB  # first 1MiB of each 2MiB
        nodes = region_base + torch.arange(_CHAIN_LEN_1MIB, dtype=torch.int64) * _STEP_ELEMS
        cpu[nodes[:-1]] = nodes[1:]
        cpu[nodes[-1]] = nodes[-1]
        starts_all[c] = nodes[0]

    # Only overwrite the first 1MiB of each 2MiB chunk.
    buf_len = num_chunks_2mib * 2 * _ELEMS_1MIB
    aligned_buf[:buf_len].copy_(cpu[:buf_len], non_blocking=False)
    starts_all = starts_all.to(device=tensor.device, non_blocking=False)
    torch.cuda.synchronize()

    # Probe in batches.
    results: List[ChunkClassResult] = []
    workspace: dict = {}
    aligned_addr = base + skip

    for batch_start in range(0, num_chunks_2mib, max_blocks_per_batch):
        batch_end = min(batch_start + max_blocks_per_batch, num_chunks_2mib)
        n = batch_end - batch_start
        starts_batch = starts_all[batch_start:batch_end]

        flush_buf = _build_flush_buffer(n, flush_mb, tensor.device)
        latency, smid_t, _, _ = cuda_decode_gemv.l2_sequential_pointer_chase_trace(
            aligned_buf, starts_batch, flush_buf,
            chain_len=_CHAIN_LEN_1MIB, workspace=workspace,
        )
        torch.cuda.synchronize()

        lat_cpu = latency.cpu().numpy()
        smid_cpu = smid_t.cpu().numpy()

        for i in range(n):
            idx = batch_start + i
            lat = lat_cpu[i, 1:]
            pc = _PAGE_CLASS_TABLE[1:]
            smid = int(smid_cpu[i])
            sm_cls = sm_class_map.get(smid, -1)
            mean_pc0, mean_pc1 = _analyse_latency(lat, pc)
            delta = mean_pc0 - mean_pc1
            pc0_is_slow = int(delta > 0)
            chunk_cls = sm_cls ^ pc0_is_slow if sm_cls >= 0 else -1
            fast = min(mean_pc0, mean_pc1)
            slow = max(mean_pc0, mean_pc1)
            results.append(ChunkClassResult(
                chunk_class=chunk_cls,
                mean_fast=fast,
                mean_slow=slow,
                separation=slow - fast,
                probed_smid=smid,
                probed_sm_class=sm_cls,
            ))

    return results


# ---------------------------------------------------------------------------
# Interleaved partition layout for k=2048 GEMV
# ---------------------------------------------------------------------------
_PC0_PAGES = [0, 2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31]
_PC1_PAGES = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30]
_PAIRS_PER_CHUNK = 128   # row-pairs per 2MB chunk (128 pages per copy × 1 pair/page)
_PAGES_PER_CHUNK = 256   # 2MB / 8KB
_HALVES_PER_PAGE = 4096  # 8KB / 2 bytes
_HALVES_PER_ROW = 2048   # k=2048


def compute_interleaved_size(n_rows: int, k: int = 2048) -> int:
    """Compute interleaved buffer size in bytes (2MB aligned)."""
    assert k == 2048, f"only k=2048 supported, got k={k}"
    row_pairs = (n_rows + 1) // 2
    chunks_needed = (row_pairs + _PAIRS_PER_CHUNK - 1) // _PAIRS_PER_CHUNK
    return chunks_needed * _CHUNK_2MIB


def create_partition_buffer(
    weight_shapes: List[tuple],
    device: str = "cuda",
) -> dict:
    """Allocate a unified buffer and probe SM/chunk classes.

    Parameters
    ----------
    weight_shapes : list of (N, K) tuples
        Shapes of weights to be packed. All must have K=2048.
    device : str
        CUDA device.

    Returns
    -------
    dict with keys:
        sm_class_lo, sm_class_hi : int  (uint64 bitfields)
        sm_class_map : dict
        buffer : torch.Tensor  (fp16, flat, on GPU)
        chunk_classes_list : list of torch.Tensor  (int32, on CPU, per weight)
        offsets : list of int  (byte offset of each weight region)
    """
    device = torch.device(device)

    # 1. Compute per-weight sizes and total
    sizes = []
    offsets = []
    total_bytes = 0
    for n, k in weight_shapes:
        sz = compute_interleaved_size(n, k)
        offsets.append(total_bytes)
        sizes.append(sz)
        total_bytes += sz

    # 2. Allocate one big buffer
    total_halves = total_bytes // 2
    buf = torch.empty(total_halves, dtype=torch.float16, device=device)

    # 3. Probe SM classes
    sm_result = probe_sm_classes(device=device)

    # 4. Build sm_class bitfields
    sm_class_lo = 0  # bits 0-63
    sm_class_hi = 0  # bits 64+
    for smid, cls in sm_result.sm_class_map.items():
        if smid < 64:
            sm_class_lo |= (cls << smid)
        else:
            sm_class_hi |= (cls << (smid - 64))

    # 5. Probe chunk classes for the entire buffer
    all_cc = probe_all_chunk_classes(buf, sm_result.sm_class_map)

    # 6. Map chunk classes to per-weight arrays
    base_addr = buf.data_ptr()
    misalign = base_addr % _CHUNK_2MIB
    skip_bytes = ((_CHUNK_2MIB - misalign) % _CHUNK_2MIB)

    chunk_classes_list = []
    for i, (n, k) in enumerate(weight_shapes):
        offset = offsets[i]
        sz = sizes[i]
        num_chunks = sz // _CHUNK_2MIB
        first_chunk = (offset + skip_bytes) // _CHUNK_2MIB
        cc_arr = torch.tensor(
            [all_cc[first_chunk + j].chunk_class for j in range(num_chunks)],
            dtype=torch.int32,
        )
        chunk_classes_list.append(cc_arr)

    return {
        "sm_class_lo": sm_class_lo,
        "sm_class_hi": sm_class_hi,
        "sm_class_map": sm_result.sm_class_map,
        "buffer": buf,
        "chunk_classes_list": chunk_classes_list,
        "offsets": offsets,
    }


def pack_interleaved_weight(
    weight: torch.Tensor,
    buf: torch.Tensor,
    chunk_classes: torch.Tensor,
) -> None:
    """Pack a [N, K=2048] weight into an interleaved dual-copy layout.

    Both Copy A and Copy B contain identical data, placed on complementary
    page_class pages so each SM class reads only from its fast pages.

    Parameters
    ----------
    weight : torch.Tensor
        [N, K] fp16, row-major, contiguous.
    buf : torch.Tensor
        Flat fp16 buffer (sub-region of the unified buffer), size >= compute_interleaved_size(N, K).
    chunk_classes : torch.Tensor
        [num_chunks] int32, chunk_class per 2MB chunk.
    """
    assert weight.is_cuda and weight.is_contiguous()
    assert weight.dtype == torch.float16
    n, k = weight.shape
    assert k == _HALVES_PER_ROW

    num_pairs = (n + 1) // 2
    num_chunks = chunk_classes.numel()
    cc_cpu = chunk_classes.cpu().numpy()

    for pair_idx in range(num_pairs):
        chunk_idx = pair_idx // _PAIRS_PER_CHUNK
        pair_in_chunk = pair_idx % _PAIRS_PER_CHUNK
        period = pair_in_chunk // 16
        intra = pair_in_chunk % 16
        cc = int(cc_cpu[chunk_idx])

        # Copy A: pages where page_class == cc (fast for SM class 0)
        # Copy B: pages where page_class == cc^1 (fast for SM class 1)
        lut_a = _PC0_PAGES if cc == 0 else _PC1_PAGES
        lut_b = _PC1_PAGES if cc == 0 else _PC0_PAGES

        page_a = period * 32 + lut_a[intra]
        page_b = period * 32 + lut_b[intra]

        abs_page_a = chunk_idx * _PAGES_PER_CHUNK + page_a
        abs_page_b = chunk_idx * _PAGES_PER_CHUNK + page_b

        row0 = pair_idx * 2
        row1 = pair_idx * 2 + 1

        # Write Copy A
        dst_offset_a = abs_page_a * _HALVES_PER_PAGE
        if row0 < n:
            buf[dst_offset_a: dst_offset_a + _HALVES_PER_ROW] = weight[row0]
        if row1 < n:
            buf[dst_offset_a + _HALVES_PER_ROW: dst_offset_a + _HALVES_PER_PAGE] = weight[row1]

        # Write Copy B (same data, different page)
        dst_offset_b = abs_page_b * _HALVES_PER_PAGE
        if row0 < n:
            buf[dst_offset_b: dst_offset_b + _HALVES_PER_ROW] = weight[row0]
        if row1 < n:
            buf[dst_offset_b + _HALVES_PER_ROW: dst_offset_b + _HALVES_PER_PAGE] = weight[row1]


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Quick self-test: probe SM classes, then probe chunk_class on a fresh buffer."""
    import sys
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    print(f"Device: {props.name}, {props.multi_processor_count} SMs")
    print()

    # Known ground truth from CSV analysis -- REMOVED.
    # SM classification is hardware-specific and changes across GPU cards.
    # Use probe_sm_classes() at runtime to determine the actual classification.

    # --- (1) probe_sm_classes ---
    print("=" * 70)
    print("(1) probe_sm_classes")
    print("=" * 70)
    sm = probe_sm_classes(device=device)
    print(f"  sm_class=0 ({len(sm.sm_class_0)} SMs): {sm.sm_class_0}")
    print(f"  sm_class=1 ({len(sm.sm_class_1)} SMs): {sm.sm_class_1}")
    print(f"  mean separation: {sm.mean_separation:.1f} cycles")
    assert sm.mean_separation > _SEPARATION_THRESHOLD, \
        f"Separation too low ({sm.mean_separation:.1f}), expected >50"

    # Sanity check: total SM count matches hardware
    total_sms = len(sm.sm_class_0) + len(sm.sm_class_1)
    assert total_sms == props.multi_processor_count, \
        f"Probed {total_sms} SMs but device has {props.multi_processor_count}"

    print("  PASS")
    print()

    # --- (2) probe_chunk_class_from_tensor ---
    print("=" * 70)
    print("(2) probe_chunk_class_from_tensor (fresh 4MiB buffer)")
    print("=" * 70)
    buf = torch.empty(4 * 1024 * 1024 // 2, dtype=torch.float16, device=device)
    r = probe_chunk_class_from_tensor(buf, sm.sm_class_map)
    print(f"  chunk_class={r.chunk_class}, fast={r.mean_fast:.1f}, "
          f"slow={r.mean_slow:.1f}, sep={r.separation:.1f}, "
          f"smid={r.probed_smid} (sm_class={r.probed_sm_class})")
    assert r.separation > _SEPARATION_THRESHOLD
    print("  PASS")
    print()

    # --- (3) probe_all_chunk_classes ---
    print("=" * 70)
    print("(3) probe_all_chunk_classes (32MiB buffer)")
    print("=" * 70)
    big = torch.empty(32 * 1024 * 1024 // 2, dtype=torch.float16, device=device)
    results = probe_all_chunk_classes(big, sm.sm_class_map)
    print(f"  probed {len(results)} x 2MiB chunks")
    for i, r in enumerate(results):
        print(f"    chunk {i}: cc={r.chunk_class}, sep={r.separation:.1f}, "
              f"smid={r.probed_smid}")
    all_ok = all(r.separation > _SEPARATION_THRESHOLD for r in results)
    assert all_ok, "Some chunks have low separation"
    print("  ALL PASS")


if __name__ == "__main__":
    _self_test()
