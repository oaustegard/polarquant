"""Quantizer: encode + ADC search.

Mirrors `remex.core.Quantizer` for the data-oblivious encode/search path.
The encode kernel fuses rotation and per-coordinate quantization in a
single pass per row. The search uses an ADC-style score table keyed by
the rotated query and the codebook centroids.
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from std.sys.info import simd_width_of
from src.codebook import Codebook, lloyd_max_codebook
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
