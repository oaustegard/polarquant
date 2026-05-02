"""Quantizer: encode + ADC search.

Mirrors `remex.core.Quantizer` for the data-oblivious encode/search path.
The encode kernel fuses rotation and per-coordinate quantization in a
single pass per row. The search uses an ADC-style score table keyed by
the rotated query and the codebook centroids.
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from std.sys.info import simd_width_of
from src.codebook import Codebook, NestedCodebook, lloyd_max_codebook
from src.matrix import Matrix
from src.rotation import haar_rotation, haar_rotation_numpy
from src.packing import pack, packed_nbytes


alias _W = simd_width_of[DType.float32]()
alias _NB = 8  # encode_batch row-block size; see _dot_block_8 below.


fn _dot_f32(a: UnsafePointer[Float32, MutExternalOrigin],
            b: UnsafePointer[Float32, MutExternalOrigin],
            n: Int) -> Float32:
    """SIMD-vectorized dot product of two contiguous Float32 buffers.

    A single `_W`-wide FMA accumulator covers the body; the tail (n % _W)
    runs scalar. For d == _W (e.g. d=16 on AVX-512 with W=16), this is
    exactly one load + one FMA + one horizontal reduce — same shape as
    a BLAS sdot kernel, so reduction order matches what NumPy produces
    on most CPUs.
    """
    var acc = SIMD[DType.float32, _W](0)
    var i = 0
    while i + _W <= n:
        var av = a.load[width=_W](i)
        var bv = b.load[width=_W](i)
        acc = av.fma(bv, acc)
        i += _W
    var s: Float32 = acc.reduce_add()
    while i < n:
        s += a[i] * b[i]
        i += 1
    return s


fn _dot_block_8(r_row: UnsafePointer[Float32, MutExternalOrigin],
                x_panel: UnsafePointer[Float32, MutExternalOrigin],
                n: Int,
                dst: UnsafePointer[Float32, MutExternalOrigin]):
    """Eight dot products against the same `r_row`, fused over a single j sweep.

    For io in [0, 8), computes `dst[io] = sum_j r_row[j] * x_panel[io*n + j]`.
    `x_panel` is a contiguous 8-row panel (rows ii..ii+7 of X when X is
    row-major, so callers can pass `X + ii*d` directly — no packing).

    The fused kernel issues one `_W`-wide load of r_row per j-step and
    eight FMAs into eight SIMD accumulators. Compared to eight separate
    `_dot_f32` calls this drops R load count by 8× and keeps R[k, :] in
    registers across all eight rows, which is the whole point of NB-row
    blocking — `R` is `~d²·4` bytes and doesn't fit in L2 at d=384, so
    the unblocked path reloads R from L3/memory once per row.

    Per-row reduction order is identical to `_dot_f32`: one `_W`-wide
    SIMD-FMA accumulator + horizontal reduce + scalar tail. Multiplication
    is fp-commutative, so `(X*R)+acc` matches `_dot_f32`'s `(R*X)+acc`
    bit-exactly. That's how `tests/test_encode.mojo` stays byte-identical.
    """
    var a0 = SIMD[DType.float32, _W](0)
    var a1 = SIMD[DType.float32, _W](0)
    var a2 = SIMD[DType.float32, _W](0)
    var a3 = SIMD[DType.float32, _W](0)
    var a4 = SIMD[DType.float32, _W](0)
    var a5 = SIMD[DType.float32, _W](0)
    var a6 = SIMD[DType.float32, _W](0)
    var a7 = SIMD[DType.float32, _W](0)

    var x0 = x_panel
    var x1 = x_panel + n
    var x2 = x_panel + 2 * n
    var x3 = x_panel + 3 * n
    var x4 = x_panel + 4 * n
    var x5 = x_panel + 5 * n
    var x6 = x_panel + 6 * n
    var x7 = x_panel + 7 * n

    var j = 0
    while j + _W <= n:
        var rv = r_row.load[width=_W](j)
        a0 = x0.load[width=_W](j).fma(rv, a0)
        a1 = x1.load[width=_W](j).fma(rv, a1)
        a2 = x2.load[width=_W](j).fma(rv, a2)
        a3 = x3.load[width=_W](j).fma(rv, a3)
        a4 = x4.load[width=_W](j).fma(rv, a4)
        a5 = x5.load[width=_W](j).fma(rv, a5)
        a6 = x6.load[width=_W](j).fma(rv, a6)
        a7 = x7.load[width=_W](j).fma(rv, a7)
        j += _W

    var s0: Float32 = a0.reduce_add()
    var s1: Float32 = a1.reduce_add()
    var s2: Float32 = a2.reduce_add()
    var s3: Float32 = a3.reduce_add()
    var s4: Float32 = a4.reduce_add()
    var s5: Float32 = a5.reduce_add()
    var s6: Float32 = a6.reduce_add()
    var s7: Float32 = a7.reduce_add()

    while j < n:
        var rv = r_row[j]
        s0 += rv * x0[j]
        s1 += rv * x1[j]
        s2 += rv * x2[j]
        s3 += rv * x3[j]
        s4 += rv * x4[j]
        s5 += rv * x5[j]
        s6 += rv * x6[j]
        s7 += rv * x7[j]
        j += 1

    dst[0] = s0
    dst[1] = s1
    dst[2] = s2
    dst[3] = s3
    dst[4] = s4
    dst[5] = s5
    dst[6] = s6
    dst[7] = s7


fn _sumsq_f64(a: UnsafePointer[Float32, MutExternalOrigin], n: Int) -> Float64:
    """Sum of squares with float64 accumulator (returned as float64).

    Float64 accumulation eliminates BLAS-vs-SIMD reduction-order ULP
    divergence — required for byte-identical norm output vs Python.
    The caller computes sqrt in float64 and casts to float32, matching
    the Python pipeline `np.sqrt(np.sum(X.astype(f64)**2)).astype(f32)`.
    """
    var acc = SIMD[DType.float64, _W](0)
    var i = 0
    while i + _W <= n:
        var av32 = a.load[width=_W](i)
        var av64 = av32.cast[DType.float64]()
        acc = av64.fma(av64, acc)
        i += _W
    var s: Float64 = acc.reduce_add()
    while i < n:
        s += Float64(a[i]) * Float64(a[i])
        i += 1
    return s


fn _heap_sift_down(scores: UnsafePointer[Float32, MutExternalOrigin],
                   indices: UnsafePointer[Int, MutExternalOrigin],
                   root: Int, size: Int):
    """Sift `root` downward to restore min-heap property keyed by `scores`.

    Min-heap: every parent has a score <= each of its children. The root
    holds the smallest score in the heap — i.e. the one to evict when a
    strictly-larger candidate arrives.
    """
    var r = root
    while True:
        var left = 2 * r + 1
        var right = 2 * r + 2
        var smallest = r
        if left < size and scores[left] < scores[smallest]:
            smallest = left
        if right < size and scores[right] < scores[smallest]:
            smallest = right
        if smallest == r:
            return
        var ts = scores[r]
        var ti = indices[r]
        scores[r] = scores[smallest]
        indices[r] = indices[smallest]
        scores[smallest] = ts
        indices[smallest] = ti
        r = smallest


fn _heap_build_min(scores: UnsafePointer[Float32, MutExternalOrigin],
                   indices: UnsafePointer[Int, MutExternalOrigin],
                   size: Int):
    """Bottom-up heapify over `size` parallel score/index slots."""
    var i = (size - 2) // 2
    while i >= 0:
        _heap_sift_down(scores, indices, i, size)
        i -= 1


def _searchsorted(boundaries: UnsafePointer[Float32, MutExternalOrigin],
                  n_b: Int, x: Float32) -> Int:
    """numpy-default 'left' binary search: smallest i such that x < boundaries[i],
    or n_b if x >= boundaries[n_b-1]. Result is in [0, n_b]."""
    var lo = 0
    var hi = n_b
    while lo < hi:
        var mid = (lo + hi) >> 1
        if x < boundaries[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


struct Quantizer(Movable):
    var R: Matrix
    var cb: Codebook
    var d: Int
    var bits: Int
    var seed: UInt64

    def __init__(out self, d: Int, bits: Int, seed: UInt64):
        """Build a Quantizer using the NumPy-compatible RNG.

        Bit-identical to `Python remex.Quantizer(d, bits, seed)` at float32.
        Use `Quantizer.from_xoshiro_seed` for the legacy fast self-contained
        path (not parity-compatible with Python).
        """
        self.d = d
        self.bits = bits
        self.seed = seed
        self.R = haar_rotation_numpy(d, seed)
        self.cb = lloyd_max_codebook(d, bits)

    @staticmethod
    def from_xoshiro_seed(d: Int, bits: Int, seed: UInt64) raises -> Quantizer:
        """Legacy self-contained Mojo path using xoshiro256++ + Marsaglia.

        Faster initialization (no Ziggurat tables) but NOT bit-identical
        to a Python `Quantizer(seed=S)`. Kept for users who want a
        standalone Mojo workflow without Python parity.
        """
        var R = haar_rotation(d, seed)
        var cb = lloyd_max_codebook(d, bits)
        return Quantizer(R^, cb^, d, bits, seed)

    def __init__(out self, var R: Matrix, var cb: Codebook,
                 d: Int, bits: Int, seed: UInt64):
        """Construct from already-built parameters (used when loading from disk)."""
        self.R = R^
        self.cb = cb^
        self.d = d
        self.bits = bits
        self.seed = seed


def encode_batch(q: Quantizer,
                 X: UnsafePointer[Float32, MutExternalOrigin],
                 n: Int,
                 mut indices_out: UnsafePointer[UInt8, MutExternalOrigin],
                 mut norms_out: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """Encode `n` rows of (n, d) float32 X into uint8 indices_out (n, d) and norms_out (n,).

    Schedule: NB=8 row blocking around the rotation matvec. For each block of
    8 rows we sweep k in [0, d) and call `_dot_block_8`, which fuses 8 dot
    products against the same R[k, :]. R[k, :] is loaded once per k and
    reused across 8 X-rows, cutting R memory traffic by 8× — the unblocked
    path reloads the full ~d²·4 bytes of R per row, which at d=384 doesn't
    fit in L2 and was the remaining gap to NumPy's BLAS after PR #37.

    The rotated coordinates live in an (NB, d) scratch buffer per block —
    never an (n, d) intermediate. Tail rows (n % 8) fall through the
    per-row `_dot_f32` path used pre-blocking, so partial-block correctness
    matches the original byte-for-byte.
    """
    var d = q.d
    var n_b = q.cb.n_levels - 1
    var rot_block = alloc[Float32](_NB * d)
    var inv_block = alloc[Float32](_NB)
    var dot_out = alloc[Float32](_NB)

    var ii = 0
    while ii < n:
        var nb = _NB if ii + _NB <= n else n - ii

        # 1. Norms + invs for the block. Reads X[ii:ii+nb, :] sequentially —
        # warms the panel into L1 ahead of the matvec sweep below.
        # Norm computed in float64 then cast to float32, matching Python's
        # `np.sqrt(np.sum(X.astype(f64)**2)).astype(f32)` order — needed for
        # byte-identical norms vs Python (issue #40).
        for io in range(nb):
            var i = ii + io
            var nm = Float32(sqrt(_sumsq_f64(X + i * d, d)))
            norms_out[i] = nm
            inv_block[io] = Float32(1.0) / nm if nm > Float32(1e-8) else Float32(1.0 / 1e-8)

        # 2. Rotation matvec. Full-block fast path uses the fused 8-way kernel;
        # the partial-block tail uses per-row _dot_f32 (same as pre-blocking).
        if nb == _NB:
            var x_panel = X + ii * d
            for k in range(d):
                _dot_block_8(q.R.data + k * d, x_panel, d, dot_out)
                rot_block[0 * d + k] = dot_out[0] * inv_block[0]
                rot_block[1 * d + k] = dot_out[1] * inv_block[1]
                rot_block[2 * d + k] = dot_out[2] * inv_block[2]
                rot_block[3 * d + k] = dot_out[3] * inv_block[3]
                rot_block[4 * d + k] = dot_out[4] * inv_block[4]
                rot_block[5 * d + k] = dot_out[5] * inv_block[5]
                rot_block[6 * d + k] = dot_out[6] * inv_block[6]
                rot_block[7 * d + k] = dot_out[7] * inv_block[7]
        else:
            for k in range(d):
                var r_row = q.R.data + k * d
                for io in range(nb):
                    var s = _dot_f32(r_row, X + (ii + io) * d, d)
                    rot_block[io * d + k] = s * inv_block[io]

        # 3. Searchsorted + write indices for the block.
        for io in range(nb):
            var base = (ii + io) * d
            var rb = rot_block + io * d
            for k in range(d):
                indices_out[base + k] = UInt8(_searchsorted(q.cb.boundaries, n_b, rb[k]))

        ii += nb

    rot_block.free()
    inv_block.free()
    dot_out.free()


def adc_search(q: Quantizer,
               indices: UnsafePointer[UInt8, MutExternalOrigin],
               norms: UnsafePointer[Float32, MutExternalOrigin],
               n: Int,
               query: UnsafePointer[Float32, MutExternalOrigin],
               k: Int,
               mut top_idx: UnsafePointer[Int, MutExternalOrigin],
               mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """ADC top-k search.

    Builds a (d, n_levels) lookup table = outer(R @ query, centroids), then
    accumulates scores per row, then takes top-k by score (descending).
    """
    var d = q.d
    var n_levels = q.cb.n_levels

    # q_rot = R @ query (SIMD dot per output coord)
    var q_rot = alloc[Float32](d)
    for i in range(d):
        q_rot[i] = _dot_f32(q.R.data + i * d, query, d)

    # table[j, c] = q_rot[j] * centroid[c]
    var table = alloc[Float32](d * n_levels)
    for j in range(d):
        var qj = q_rot[j]
        var trow = j * n_levels
        for c in range(n_levels):
            table[trow + c] = qj * q.cb.centroids[c]

    # Score each row: s_i = sum_j table[j, idx[i, j]] * norms[i].
    var scores = alloc[Float32](n)
    for i in range(n):
        var s: Float32 = Float32(0.0)
        var base = i * d
        for j in range(d):
            var c = Int(indices[base + j])
            s += table[j * n_levels + c]
        scores[i] = s * norms[i]

    # Top-k: simple O(n*k) selection (k is small typically). For each output
    # slot, scan remaining for max and mark used.
    var used = alloc[UInt8](n)
    for i in range(n):
        used[i] = UInt8(0)
    var kk = k if k <= n else n
    for outer in range(kk):
        var best_i: Int = -1
        var best_s: Float32 = Float32(0.0)
        for i in range(n):
            if used[i] == UInt8(0):
                if best_i < 0 or scores[i] > best_s:
                    best_i = i
                    best_s = scores[i]
        top_idx[outer] = best_i
        top_scores[outer] = best_s
        used[best_i] = UInt8(1)

    used.free()
    scores.free()
    table.free()
    q_rot.free()


fn _accum_row_f32(dst: UnsafePointer[Float32, MutExternalOrigin],
                  r_row: UnsafePointer[Float32, MutExternalOrigin],
                  scale: Float32, n: Int):
    """Fused multiply-add row accumulation: dst[j] += scale * r_row[j].

    Used by `decode_batch` for the rotation matvec. Iterating with the
    outer loop over k and an inner broadcast-FMA over j means R is
    accessed sequentially row-by-row and `dst` is reused without a
    transpose — same access pattern as the encode hot loop.
    """
    var bcast = SIMD[DType.float32, _W](scale)
    var j = 0
    while j + _W <= n:
        var rv = r_row.load[width=_W](j)
        var ov = dst.load[width=_W](j)
        dst.store(j, bcast.fma(rv, ov))
        j += _W
    while j < n:
        dst[j] += scale * r_row[j]
        j += 1


def decode_batch(q: Quantizer,
                 nested: NestedCodebook,
                 indices: UnsafePointer[UInt8, MutExternalOrigin],
                 norms: UnsafePointer[Float32, MutExternalOrigin],
                 n: Int,
                 precision: Int,
                 mut X_hat_out: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """Reconstruct float32 vectors from packed indices + norms.

    Mirrors `Quantizer.decode` in `remex/core.py`. `precision == 0` (or
    `precision == q.bits`) means full precision and uses the full-precision
    centroid table from `q.cb`; lower precisions use the matching nested
    table from `nested` and right-shift the indices by `q.bits - precision`.

    Output layout: `X_hat_out` is a contiguous (n, d) float32 buffer.
    For each row i, the formula is:

        X_hat_rot[i, k] = centroids[indices[i, k] >> shift]
        X_hat_unit[i, j] = sum_k X_hat_rot[i, k] * R[k, j]
        X_hat[i, j]      = X_hat_unit[i, j] * norms[i]

    `nested` may be unused when precision is full; pass any
    `NestedCodebook` with `max_bits == q.bits` for that case (cheap to
    build via `nested_codebooks_from(q.cb, d)`).
    """
    var d = q.d
    var bits = q.bits
    if precision < 0 or precision > bits:
        raise Error("decode_batch: precision must be 0..bits")
    if precision != 0 and precision != bits and nested.max_bits != bits:
        raise Error("decode_batch: nested.max_bits must equal q.bits")

    var use_full = (precision == 0 or precision == bits)
    var shift = 0 if use_full else bits - precision

    var centroids: UnsafePointer[Float32, MutExternalOrigin]
    if use_full:
        centroids = q.cb.centroids
    else:
        centroids = nested.get_table(precision)

    # Per-row dequantized rotated vector. Keeping it as a single d-long
    # scratch buffer avoids materializing the full (n, d) X_hat_rot matrix.
    var xhrot = alloc[Float32](d)

    for i in range(n):
        var base = i * d
        # Stage 1: lookup centroids per coordinate.
        for k in range(d):
            var c = Int(indices[base + k])
            if shift > 0:
                c = c >> shift
            xhrot[k] = centroids[c]

        # Stage 2: rotate back. Zero the output row, then accumulate
        # `xhrot[k] * R[k, :]` for k in [0, d). R is row-major so R[k, :]
        # is a contiguous run of d float32s starting at q.R.data + k*d.
        for j in range(d):
            X_hat_out[base + j] = Float32(0.0)
        for k in range(d):
            _accum_row_f32(X_hat_out + base, q.R.data + k * d, xhrot[k], d)

        # Stage 3: rescale by per-row norm.
        var nm = norms[i]
        for j in range(d):
            X_hat_out[base + j] *= nm

    xhrot.free()


def search_twostage(q: Quantizer,
                    nested: NestedCodebook,
                    indices: UnsafePointer[UInt8, MutExternalOrigin],
                    norms: UnsafePointer[Float32, MutExternalOrigin],
                    n: Int,
                    query: UnsafePointer[Float32, MutExternalOrigin],
                    k: Int,
                    candidates: Int,
                    coarse_precision: Int,
                    mut top_idx: UnsafePointer[Int, MutExternalOrigin],
                    mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """Two-stage retrieval: coarse ADC scan + full-precision rerank.

    Stage 1: ADC scan at `coarse_precision` over the full corpus. For each
    row i, the score uses the right-shifted index `indices[i, j] >> shift`
    against the coarse centroid table at `coarse_precision` bits. Keeps the
    top `candidates` rows by coarse score.

    Stage 2: Re-score the candidate rows at full precision (`q.bits`) using
    the full-precision centroid table. Return the top-k by fine score.

    `indices` is a contiguous (n, d) uint8 buffer of full-precision codes
    (already unpacked). `norms` is a length-`n` float32 buffer. `nested`
    must have been built with `max_bits == q.bits`.
    """
    var d = q.d
    var bits = q.bits
    if coarse_precision < 1 or coarse_precision > bits:
        raise Error("search_twostage: coarse_precision must be 1..bits")
    if nested.max_bits != bits:
        raise Error("search_twostage: nested.max_bits must equal q.bits")

    var shift = bits - coarse_precision
    var n_levels_coarse = 1 << coarse_precision
    var coarse_k = candidates if candidates <= n else n
    if coarse_k < k:
        raise Error("search_twostage: candidates must be >= k")

    # q_rot = R @ query (SIMD dot per output coord) — same shape as adc_search.
    var q_rot = alloc[Float32](d)
    for i in range(d):
        q_rot[i] = _dot_f32(q.R.data + i * d, query, d)

    # Stage 1: build coarse lookup table and score each row.
    var coarse_centroids = nested.get_table(coarse_precision)
    var coarse_table = alloc[Float32](d * n_levels_coarse)
    for j in range(d):
        var qj = q_rot[j]
        var trow = j * n_levels_coarse
        for c in range(n_levels_coarse):
            coarse_table[trow + c] = qj * coarse_centroids[c]

    var coarse_scores = alloc[Float32](n)
    for i in range(n):
        var s: Float32 = Float32(0.0)
        var base = i * d
        for j in range(d):
            var c_full = Int(indices[base + j])
            var c_coarse = c_full >> shift if shift > 0 else c_full
            s += coarse_table[j * n_levels_coarse + c_coarse]
        coarse_scores[i] = s * norms[i]
    coarse_table.free()

    # Pick the top `coarse_k` candidates by coarse score using a min-heap.
    # The heap root tracks the smallest-score-currently-kept; replace it
    # whenever a strictly-larger candidate arrives. Strict ">" matches the
    # tie-break of the previous selection loop (same-score candidates do
    # not displace each other), so the candidate-set MEMBERSHIP — and
    # therefore the rerank output — is identical. Order within the set
    # doesn't matter for the rerank, so heap order is fine.
    # O(n log k) ≈ 90× fewer comparisons than the prior O(n·k) at the
    # default (n=10000, candidates=500) settings — closes issue #49.
    var heap_scores = alloc[Float32](coarse_k)
    var heap_idx = alloc[Int](coarse_k)
    for i in range(coarse_k):
        heap_scores[i] = coarse_scores[i]
        heap_idx[i] = i
    _heap_build_min(heap_scores, heap_idx, coarse_k)
    for i in range(coarse_k, n):
        if coarse_scores[i] > heap_scores[0]:
            heap_scores[0] = coarse_scores[i]
            heap_idx[0] = i
            _heap_sift_down(heap_scores, heap_idx, 0, coarse_k)
    var cand_idx = alloc[Int](coarse_k)
    for ci in range(coarse_k):
        cand_idx[ci] = heap_idx[ci]
    heap_scores.free()
    heap_idx.free()
    coarse_scores.free()

    # Stage 2: full-precision rerank on the candidate set.
    # fine_score(i) = sum_j q_rot[j] * full_centroids[indices[i, j]] * norms[i]
    var fine_centroids = q.cb.centroids
    var fine_scores = alloc[Float32](coarse_k)
    for ci in range(coarse_k):
        var i = cand_idx[ci]
        var base = i * d
        var s: Float32 = Float32(0.0)
        for j in range(d):
            var c = Int(indices[base + j])
            s += q_rot[j] * fine_centroids[c]
        fine_scores[ci] = s * norms[i]

    # Top-k by fine score.
    var kk = k if k <= coarse_k else coarse_k
    var used2 = alloc[UInt8](coarse_k)
    for i in range(coarse_k):
        used2[i] = UInt8(0)
    for outer in range(kk):
        var best_i: Int = -1
        var best_s: Float32 = Float32(0.0)
        for ci in range(coarse_k):
            if used2[ci] == UInt8(0):
                if best_i < 0 or fine_scores[ci] > best_s:
                    best_i = ci
                    best_s = fine_scores[ci]
        top_idx[outer] = cand_idx[best_i]
        top_scores[outer] = best_s
        used2[best_i] = UInt8(1)

    used2.free()
    fine_scores.free()
    cand_idx.free()
    q_rot.free()
