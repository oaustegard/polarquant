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
from src.rotation import haar_rotation
from src.packing import pack, packed_nbytes


alias _W = simd_width_of[DType.float32]()


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


fn _sumsq_f32(a: UnsafePointer[Float32, MutExternalOrigin], n: Int) -> Float32:
    """SIMD-vectorized sum of squares: sum_i a[i] * a[i]."""
    var acc = SIMD[DType.float32, _W](0)
    var i = 0
    while i + _W <= n:
        var av = a.load[width=_W](i)
        acc = av.fma(av, acc)
        i += _W
    var s: Float32 = acc.reduce_add()
    while i < n:
        s += a[i] * a[i]
        i += 1
    return s


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
        self.d = d
        self.bits = bits
        self.seed = seed
        self.R = haar_rotation(d, seed)
        self.cb = lloyd_max_codebook(d, bits)

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

    Hot path: per row, compute norm, normalize, rotate, searchsorted into boundaries.
    The rotated coordinates live in a stack/heap buffer per row — never
    materialized as an (n, d) intermediate.
    """
    var d = q.d
    var n_b = q.cb.n_levels - 1
    var rotated = alloc[Float32](d)
    for i in range(n):
        var base = i * d
        var nm = sqrt(_sumsq_f32(X + base, d))
        norms_out[i] = nm
        var inv = Float32(1.0) / nm if nm > Float32(1e-8) else Float32(1.0 / 1e-8)

        # Rotate: rotated[k] = (sum_j R[k, j] * X[i, j]) / nm.
        # Each row of R and X[i, :] is contiguous — SIMD dot per output coord.
        for k in range(d):
            rotated[k] = _dot_f32(q.R.data + k * d, X + base, d) * inv

        # Searchsorted per coordinate
        for k in range(d):
            indices_out[base + k] = UInt8(_searchsorted(q.cb.boundaries, n_b, rotated[k]))
    rotated.free()


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

    # Pick the top `coarse_k` candidates by coarse score (selection-style).
    # Order within the candidate set doesn't matter — only membership does.
    var cand_idx = alloc[Int](coarse_k)
    var used = alloc[UInt8](n)
    for i in range(n):
        used[i] = UInt8(0)
    for outer in range(coarse_k):
        var best_i: Int = -1
        var best_s: Float32 = Float32(0.0)
        for i in range(n):
            if used[i] == UInt8(0):
                if best_i < 0 or coarse_scores[i] > best_s:
                    best_i = i
                    best_s = coarse_scores[i]
        cand_idx[outer] = best_i
        used[best_i] = UInt8(1)
    used.free()
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
