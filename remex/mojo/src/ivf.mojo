"""Coarse IVF index over the Matryoshka tier — Mojo port of `remex.ivf`.

Mirrors `remex.ivf.IVFCoarseIndex` (Python). Partitions a `PackedVectors`
corpus into `2**n_bits` cells via a *data-oblivious* hash (no k-means,
no training). Visiting only `nprobe` cells per query trades recall for
stage-1 latency in two-stage retrieval.

Two hash modes — both reproduce the Python cell assignments byte-for-byte
when seeded the same way:

  - `MODE_LSH` — random-hyperplane SimHash. `n_bits` hyperplanes are drawn
    from the NumPy-compatible Ziggurat normal RNG (`NumpyNormalRNG`), so
    `(d, n_bits, seed)` produces the *same* hyperplane matrix as
    `np.random.default_rng(seed).standard_normal((n_bits, d))`. Cell ID
    is the bit-pattern of `sign(H @ s)` where `s = 2*MSB(idx) - 1` is the
    1-bit reconstruction in rotated space.

  - `MODE_ROTATED_PREFIX` — sign of the first `n_bits` post-rotation
    coordinates, taken directly as the MSBs of the encoded indices. Free
    given the existing rotation matrix.

Multi-probe ordering: cells are ranked by Hamming distance to the query
hash, ties broken by ascending cell ID. `nprobe = n_cells` recovers a
flat coarse scan exactly. This is enforced by both
`tests/test_ivf.py::TestSearchCoarse::test_full_nprobe_matches_search_adc`
(Python) and `tests/test_ivf.mojo::test_full_nprobe_matches_adc_search`
(Mojo).

Closes issue #61 (Mojo parity with Python `IVFCoarseIndex` from PR #58 / #53).
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from src.codebook import NestedCodebook
from src.matrix import Matrix
from src.packed_vectors import PackedVectors, unpack_at, unpack_rows
from src.quantizer import Quantizer
from src.rng_numpy import NumpyNormalRNG


alias MODE_LSH = 0
alias MODE_ROTATED_PREFIX = 1


fn _popcount32(x: UInt32) -> Int:
    """Hamming weight of a 32-bit integer (SWAR popcount)."""
    var v = x
    v = v - ((v >> UInt32(1)) & UInt32(0x55555555))
    v = (v & UInt32(0x33333333)) + ((v >> UInt32(2)) & UInt32(0x33333333))
    v = (v + (v >> UInt32(4))) & UInt32(0x0F0F0F0F)
    v = (v * UInt32(0x01010101)) >> UInt32(24)
    return Int(v)


struct IVFCoarseIndex(Movable):
    """Inverted-file index over a `PackedVectors` corpus.

    Owning struct: `__init__` allocates the cell metadata + (optionally)
    the LSH hyperplane matrix; `__del__` frees them. The underlying
    `PackedVectors` is *not* owned — callers must keep it alive for the
    lifetime of the index (parity with the Python class, which stores
    `compressed` by reference).

    Layout matches `remex.ivf.IVFCoarseIndex`:
      - `cell_ids`   (n,)             uint16 cell ID per corpus row
      - `sorted_idx` (n,)             int64 corpus row indices sorted by cell
      - `cell_offsets` (n_cells + 1,) int64 CSR-style offsets into sorted_idx
      - `hyperplanes` (n_bits, d)     float32 row-major (LSH only; otherwise
                                      a 1-Float32 sentinel buffer)
    """
    var cell_ids: UnsafePointer[UInt16, MutExternalOrigin]
    var sorted_idx: UnsafePointer[Int, MutExternalOrigin]
    var cell_offsets: UnsafePointer[Int, MutExternalOrigin]
    var hyperplanes: UnsafePointer[Float32, MutExternalOrigin]
    var has_hyperplanes: Bool
    var n: Int
    var d: Int
    var bits: Int
    var n_bits: Int
    var n_cells: Int
    var mode: Int
    var seed: UInt64

    def __init__(out self, n: Int, d: Int, bits: Int,
                 n_bits: Int, n_cells: Int, mode: Int, seed: UInt64,
                 has_hyperplanes: Bool):
        self.n = n
        self.d = d
        self.bits = bits
        self.n_bits = n_bits
        self.n_cells = n_cells
        self.mode = mode
        self.seed = seed
        self.has_hyperplanes = has_hyperplanes
        self.cell_ids = alloc[UInt16](n)
        self.sorted_idx = alloc[Int](n)
        self.cell_offsets = alloc[Int](n_cells + 1)
        var hp_size = n_bits * d if has_hyperplanes else 1
        self.hyperplanes = alloc[Float32](hp_size)

    def __del__(deinit self):
        self.cell_ids.free()
        self.sorted_idx.free()
        self.cell_offsets.free()
        self.hyperplanes.free()

    fn index_nbytes(self) -> Int:
        """In-RAM bytes of the IVF structure (excluding the corpus).

        Mirrors `IVFCoarseIndex.index_nbytes`. Counts only the CSR layout
        + the hyperplane matrix (when LSH); the underlying `PackedVectors`
        is reported separately by the caller.
        """
        var total = self.n * 2  # cell_ids: uint16
        total += self.n * 8     # sorted_idx: int64
        total += (self.n_cells + 1) * 8  # cell_offsets: int64
        if self.has_hyperplanes:
            total += self.n_bits * self.d * 4  # float32
        return total


# ---------------------------------------------------------------------------
# Cell-ID computation
# ---------------------------------------------------------------------------


fn _pack_bits_to_cell(sign_bits: UnsafePointer[UInt8, MutExternalOrigin],
                      n_bits: Int) -> UInt32:
    """Pack `n_bits` {0,1} values (LSB-first) into a uint32 cell ID.

    Bit `b` of the output comes from `sign_bits[b]`. Matches Python's
    `_pack_bits_to_cell` (which ORs `sign_bits[:, b] << b` per column).
    """
    var cid: UInt32 = UInt32(0)
    for b in range(n_bits):
        if sign_bits[b] != UInt8(0):
            cid = cid | (UInt32(1) << UInt32(b))
    return cid


def _cell_ids_rotated_prefix(mut ivf: IVFCoarseIndex, packed: PackedVectors) raises:
    """Compute cell IDs from the first `n_bits` MSBs of the encoded indices.

    1-bit MSB extraction == sign of the rotated coordinate, so this is a
    free deterministic hash. Mirrors `IVFCoarseIndex._cell_ids_rotated_prefix`.
    """
    var n = packed.n
    var d = packed.d
    var bits = packed.bits
    var n_bits = ivf.n_bits
    var shift = bits - 1

    var chunk = 65536
    var unpacked = alloc[UInt8](chunk * d)
    var sign_bits = alloc[UInt8](n_bits)

    var start = 0
    while start < n:
        var end = start + chunk
        if end > n:
            end = n
        var n_rows = end - start
        unpack_rows(packed, start, end, unpacked)
        for r in range(n_rows):
            var row = unpacked + r * d
            for b in range(n_bits):
                sign_bits[b] = (row[b] >> UInt8(shift)) & UInt8(1)
            ivf.cell_ids[start + r] = UInt16(Int(_pack_bits_to_cell(sign_bits, n_bits)))
        start = end

    unpacked.free()
    sign_bits.free()


def _draw_hyperplanes(mut ivf: IVFCoarseIndex) raises:
    """Sample `(n_bits, d)` standard-normal hyperplanes via the
    NumPy-compatible Ziggurat RNG.

    Order matches Python's `np.random.default_rng(seed).standard_normal((n_bits, d))`:
    row-major fill, draw `(b, k)` at flat index `b * d + k`. End-to-end this
    means a Mojo `IVFCoarseIndex(n_bits, mode='lsh', seed=S)` gets the same
    hyperplane matrix — and therefore the same cell IDs — as the Python one
    seeded identically.
    """
    var rng = NumpyNormalRNG(ivf.seed)
    var total = ivf.n_bits * ivf.d
    for i in range(total):
        ivf.hyperplanes[i] = Float32(rng.next_normal())


def _cell_ids_lsh(mut ivf: IVFCoarseIndex, packed: PackedVectors) raises:
    """Random-hyperplane LSH on the 1-bit reconstruction of the corpus.

    Mirrors `IVFCoarseIndex._cell_ids_lsh`: for each row, build
    `signed = 2 * MSB(idx) - 1` (a length-d ±1 vector in float32),
    project onto every hyperplane, and threshold at zero. Reduction
    order is `sum_k H[b, k] * signed[k]` per (row, b), matching Python's
    `signed @ H.T`.
    """
    var n = packed.n
    var d = packed.d
    var bits = packed.bits
    var n_bits = ivf.n_bits
    var shift = bits - 1

    var chunk = 4096
    var unpacked = alloc[UInt8](chunk * d)
    var signed = alloc[Float32](d)
    var sign_bits = alloc[UInt8](n_bits)

    var start = 0
    while start < n:
        var end = start + chunk
        if end > n:
            end = n
        var n_rows = end - start
        unpack_rows(packed, start, end, unpacked)
        for r in range(n_rows):
            var row = unpacked + r * d
            for k in range(d):
                var msb = Float32((row[k] >> UInt8(shift)) & UInt8(1))
                signed[k] = Float32(2.0) * msb - Float32(1.0)
            for b in range(n_bits):
                var hplane = ivf.hyperplanes + b * d
                var s: Float32 = Float32(0.0)
                for k in range(d):
                    s += signed[k] * hplane[k]
                sign_bits[b] = UInt8(1) if s > Float32(0.0) else UInt8(0)
            ivf.cell_ids[start + r] = UInt16(Int(_pack_bits_to_cell(sign_bits, n_bits)))
        start = end

    unpacked.free()
    signed.free()
    sign_bits.free()


def _build_inverted_lists(mut ivf: IVFCoarseIndex) raises:
    """Build CSR `(sorted_idx, cell_offsets)` from `cell_ids`.

    Counting sort, stable by row index — equivalent to numpy's
    `argsort(kind='stable')` over `cell_ids`. Within-cell ordering is
    ascending row index, matching Python.
    """
    var n = ivf.n
    var n_cells = ivf.n_cells

    # Histogram of cell_ids
    for c in range(n_cells + 1):
        ivf.cell_offsets[c] = 0
    for i in range(n):
        var c = Int(ivf.cell_ids[i])
        ivf.cell_offsets[c + 1] += 1
    # Prefix-sum to get CSR offsets
    for c in range(1, n_cells + 1):
        ivf.cell_offsets[c] += ivf.cell_offsets[c - 1]

    # Place: walk i in order, drop into the next free slot of its cell.
    # Use a scratch cursor copy so cell_offsets stays as the canonical CSR.
    var cursor = alloc[Int](n_cells)
    for c in range(n_cells):
        cursor[c] = ivf.cell_offsets[c]
    for i in range(n):
        var c = Int(ivf.cell_ids[i])
        ivf.sorted_idx[cursor[c]] = i
        cursor[c] += 1
    cursor.free()


def build_ivf(q: Quantizer, packed: PackedVectors,
              n_bits: Int, mode: Int, seed: UInt64) raises -> IVFCoarseIndex:
    """Construct an IVFCoarseIndex over `packed` using `q.R` for hashing.

    Args:
        q: Quantizer that produced `packed` (provides `R` for query
           rotation and `bits` for the MSB shift).
        packed: PackedVectors corpus to index.
        n_bits: 1..16, number of hash bits → `n_cells = 2**n_bits`.
        mode: `MODE_LSH` (0) or `MODE_ROTATED_PREFIX` (1).
        seed: LSH hyperplane seed (ignored for rotated_prefix).

    Returns: an owning `IVFCoarseIndex`. The underlying `packed` is *not*
    copied — keep it alive for the lifetime of the returned index.
    """
    if n_bits < 1 or n_bits > 16:
        raise Error("build_ivf: n_bits must be 1..16")
    if mode != MODE_LSH and mode != MODE_ROTATED_PREFIX:
        raise Error("build_ivf: mode must be MODE_LSH or MODE_ROTATED_PREFIX")
    if mode == MODE_ROTATED_PREFIX and n_bits > q.d:
        raise Error("build_ivf: n_bits exceeds quantizer.d for rotated_prefix")

    var n_cells = 1 << n_bits
    var has_hp = (mode == MODE_LSH)
    var ivf = IVFCoarseIndex(packed.n, q.d, q.bits, n_bits, n_cells,
                             mode, seed, has_hp)

    if mode == MODE_LSH:
        _draw_hyperplanes(ivf)
        _cell_ids_lsh(ivf, packed)
    else:
        _cell_ids_rotated_prefix(ivf, packed)

    _build_inverted_lists(ivf)
    return ivf^


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


fn _q_rot(q: Quantizer, query: UnsafePointer[Float32, MutExternalOrigin],
          mut out: UnsafePointer[Float32, MutExternalOrigin]):
    """`out = q.R @ query`. Plain row-major matvec; reduction order matches
    `Quantizer._dot_f32` modulo SIMD blocking — IVF uses scalar accumulators
    everywhere downstream, so cell-ID assignment is unaffected.
    """
    var d = q.d
    for i in range(d):
        var s: Float32 = Float32(0.0)
        var rrow = q.R.data + i * d
        for j in range(d):
            s += rrow[j] * query[j]
        out[i] = s


fn _query_cell_from_rot(ivf: IVFCoarseIndex,
                        q_rot: UnsafePointer[Float32, MutExternalOrigin]) -> Int:
    """Cell ID for a pre-rotated query. Mirrors Python `_query_cell_from_rot`."""
    var cid: Int = 0
    if ivf.mode == MODE_ROTATED_PREFIX:
        for b in range(ivf.n_bits):
            if q_rot[b] > Float32(0.0):
                cid = cid | (1 << b)
    else:
        for b in range(ivf.n_bits):
            var hplane = ivf.hyperplanes + b * ivf.d
            var s: Float32 = Float32(0.0)
            for k in range(ivf.d):
                s += hplane[k] * q_rot[k]
            if s > Float32(0.0):
                cid = cid | (1 << b)
    return cid


def query_cell(ivf: IVFCoarseIndex, q: Quantizer,
               query: UnsafePointer[Float32, MutExternalOrigin]) raises -> Int:
    """Compute the cell ID for a single query vector (rotates internally).

    Mirrors `IVFCoarseIndex.query_cell`.
    """
    var q_rot = alloc[Float32](q.d)
    _q_rot(q, query, q_rot)
    var cid = _query_cell_from_rot(ivf, q_rot)
    q_rot.free()
    return cid


def probe_cells(ivf: IVFCoarseIndex, query_cell: Int, nprobe: Int,
                mut out_cells: UnsafePointer[Int, MutExternalOrigin]) raises -> Int:
    """Write the top `nprobe` cells (by Hamming distance) into `out_cells`.

    Tie-break: ascending cell ID, matching Python's
    `np.lexsort((all_cells, hd))`. Done as a bucket sort over Hamming
    distance — n_cells <= 65536 and there are at most n_bits + 1 buckets.
    Returns the number of cells written (in [0, min(nprobe, n_cells)]).
    """
    if nprobe <= 0:
        return 0
    if nprobe >= ivf.n_cells:
        for c in range(ivf.n_cells):
            out_cells[c] = c
        return ivf.n_cells

    var n_buckets = ivf.n_bits + 1
    var counts = alloc[Int](n_buckets)
    for b in range(n_buckets):
        counts[b] = 0
    for c in range(ivf.n_cells):
        var hd = _popcount32(UInt32(c) ^ UInt32(query_cell))
        counts[hd] += 1
    var bucket_off = alloc[Int](n_buckets)
    bucket_off[0] = 0
    for b in range(1, n_buckets):
        bucket_off[b] = bucket_off[b - 1] + counts[b - 1]

    # Walk cells ascending and place into bucket slots — within-bucket
    # order ends up ascending by cell ID, which is what np.lexsort gives.
    var cursor = alloc[Int](n_buckets)
    for b in range(n_buckets):
        cursor[b] = bucket_off[b]
    var sorted_cells = alloc[Int](ivf.n_cells)
    for c in range(ivf.n_cells):
        var hd = _popcount32(UInt32(c) ^ UInt32(query_cell))
        sorted_cells[cursor[hd]] = c
        cursor[hd] += 1

    var k = nprobe if nprobe <= ivf.n_cells else ivf.n_cells
    for i in range(k):
        out_cells[i] = sorted_cells[i]

    counts.free()
    bucket_off.free()
    cursor.free()
    sorted_cells.free()
    return k


def candidate_indices(ivf: IVFCoarseIndex, query_cell: Int, nprobe: Int,
                      mut out_idx: UnsafePointer[Int, MutExternalOrigin]) raises -> Int:
    """Write candidate corpus row indices for the top `nprobe` cells into
    `out_idx` and return the count.

    `out_idx` must be sized for `ivf.n` (the worst case at `nprobe == n_cells`).
    Mirrors `IVFCoarseIndex.candidate_indices`.
    """
    var cells = alloc[Int](ivf.n_cells)
    var n_visited = probe_cells(ivf, query_cell, nprobe, cells)
    var written: Int = 0
    for ci in range(n_visited):
        var c = cells[ci]
        var lo = ivf.cell_offsets[c]
        var hi = ivf.cell_offsets[c + 1]
        for j in range(lo, hi):
            out_idx[written] = ivf.sorted_idx[j]
            written += 1
    cells.free()
    return written


def candidate_count(ivf: IVFCoarseIndex, q: Quantizer,
                    query: UnsafePointer[Float32, MutExternalOrigin],
                    nprobe: Int) raises -> Int:
    """Total candidate corpus rows in the top `nprobe` cells (no allocation).

    Mirrors `IVFCoarseIndex.candidate_count`.
    """
    var qc = query_cell(ivf, q, query)
    var cells = alloc[Int](ivf.n_cells)
    var n_visited = probe_cells(ivf, qc, nprobe, cells)
    var total: Int = 0
    for ci in range(n_visited):
        var c = cells[ci]
        total += ivf.cell_offsets[c + 1] - ivf.cell_offsets[c]
    cells.free()
    return total


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_coarse(ivf: IVFCoarseIndex, q: Quantizer, nested: NestedCodebook,
                  packed: PackedVectors,
                  query: UnsafePointer[Float32, MutExternalOrigin],
                  k: Int, nprobe: Int, precision: Int,
                  mut top_idx: UnsafePointer[Int, MutExternalOrigin],
                  mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises -> Int:
    """IVF coarse ADC: visit `nprobe` cells, score by ADC, return top-k.

    Mirrors `IVFCoarseIndex.search_coarse`. Equivalent to `Quantizer.adc_search`
    restricted to the union of the `nprobe` cells closest (in Hamming
    distance) to the query's hash.

    Args:
        ivf, q, nested, packed: index + quantizer + nested codebook + corpus.
        query: (d,) raw query vector.
        k: number of coarse candidates to return.
        nprobe: cells to visit. `n_cells` recovers a flat scan exactly.
        precision: 0 = full precision (use `q.cb.centroids`); otherwise
            1..bits (use `nested.get_table(precision)` and right-shift
            indices by `bits - precision`).
        top_idx, top_scores: output buffers, must be sized for `min(k, n)`.

    Returns: number of results written (= `min(k, n_visited)`).
    """
    var d = q.d
    var bits = q.bits
    var use_full = (precision == 0 or precision == bits)
    if not use_full and (precision < 1 or precision > bits):
        raise Error("search_coarse: precision must be 0 or 1..bits")

    var q_rot = alloc[Float32](d)
    _q_rot(q, query, q_rot)

    var qc = _query_cell_from_rot(ivf, q_rot)
    var cand = alloc[Int](ivf.n)
    var n_cand = candidate_indices(ivf, qc, nprobe, cand)
    if n_cand == 0:
        cand.free()
        q_rot.free()
        return 0

    # Build ADC table: outer(q_rot, centroids[at precision])
    var n_levels: Int
    var centroids: UnsafePointer[Float32, MutExternalOrigin]
    if use_full:
        n_levels = q.cb.n_levels
        centroids = q.cb.centroids
    else:
        n_levels = 1 << precision
        centroids = nested.get_table(precision)
    var shift = 0 if use_full else bits - precision

    var table = alloc[Float32](d * n_levels)
    for j in range(d):
        var qj = q_rot[j]
        var trow = j * n_levels
        for c in range(n_levels):
            table[trow + c] = qj * centroids[c]

    # Score each candidate by unpacking its row and looking up table entries.
    var row = alloc[UInt8](d)
    var scores = alloc[Float32](n_cand)
    for ci in range(n_cand):
        var i = cand[ci]
        unpack_at(packed, i, row)
        var s: Float32 = Float32(0.0)
        for j in range(d):
            var c_full = Int(row[j])
            var c = c_full >> shift if shift > 0 else c_full
            s += table[j * n_levels + c]
        scores[ci] = s * packed.norms[i]

    # Top-k by score (descending). Mirrors `adc_search`'s O(n*k) selection.
    var kk = k if k <= n_cand else n_cand
    var used = alloc[UInt8](n_cand)
    for i in range(n_cand):
        used[i] = UInt8(0)
    for outer in range(kk):
        var best_i: Int = -1
        var best_s: Float32 = Float32(0.0)
        for ci in range(n_cand):
            if used[ci] == UInt8(0):
                if best_i < 0 or scores[ci] > best_s:
                    best_i = ci
                    best_s = scores[ci]
        top_idx[outer] = cand[best_i]
        top_scores[outer] = best_s
        used[best_i] = UInt8(1)

    used.free()
    scores.free()
    row.free()
    table.free()
    cand.free()
    q_rot.free()
    return kk


def search_twostage(ivf: IVFCoarseIndex, q: Quantizer, nested: NestedCodebook,
                    packed: PackedVectors,
                    query: UnsafePointer[Float32, MutExternalOrigin],
                    k: Int, candidates: Int, nprobe: Int,
                    coarse_precision: Int,
                    mut top_idx: UnsafePointer[Int, MutExternalOrigin],
                    mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises -> Int:
    """IVF coarse + full-precision rerank.

    Mirrors `IVFCoarseIndex.search_twostage`. Stage 1 is `search_coarse` at
    `coarse_precision` over `nprobe` cells, keeping the top `candidates`
    rows by coarse score. Stage 2 reranks those at full precision and
    returns the top `k`.

    Args:
        coarse_precision: 1..bits. Pass `0` to mean "full precision";
            for parity with Python's `Quantizer.search_twostage` default,
            callers typically pass `max(1, q.bits - 2)`.

    Returns: number of results written (= `min(k, candidates, n_visited)`).
    """
    var d = q.d
    var bits = q.bits

    # Stage 1: coarse ADC over the IVF candidate set.
    var stage1_idx = alloc[Int](candidates)
    var stage1_scores = alloc[Float32](candidates)
    var n_stage1 = search_coarse(ivf, q, nested, packed, query,
                                 candidates, nprobe, coarse_precision,
                                 stage1_idx, stage1_scores)
    if n_stage1 == 0:
        stage1_idx.free()
        stage1_scores.free()
        return 0

    # Stage 2: full-precision rerank.
    var q_rot = alloc[Float32](d)
    _q_rot(q, query, q_rot)
    var fine_centroids = q.cb.centroids
    var fine_scores = alloc[Float32](n_stage1)
    var row = alloc[UInt8](d)
    for ci in range(n_stage1):
        var i = stage1_idx[ci]
        unpack_at(packed, i, row)
        var s: Float32 = Float32(0.0)
        for j in range(d):
            var c = Int(row[j])
            s += q_rot[j] * fine_centroids[c]
        fine_scores[ci] = s * packed.norms[i]

    # Top-k by fine score.
    var kk = k if k <= n_stage1 else n_stage1
    var used = alloc[UInt8](n_stage1)
    for i in range(n_stage1):
        used[i] = UInt8(0)
    for outer in range(kk):
        var best_i: Int = -1
        var best_s: Float32 = Float32(0.0)
        for ci in range(n_stage1):
            if used[ci] == UInt8(0):
                if best_i < 0 or fine_scores[ci] > best_s:
                    best_i = ci
                    best_s = fine_scores[ci]
        top_idx[outer] = stage1_idx[best_i]
        top_scores[outer] = best_s
        used[best_i] = UInt8(1)

    used.free()
    fine_scores.free()
    row.free()
    q_rot.free()
    stage1_idx.free()
    stage1_scores.free()
    return kk
