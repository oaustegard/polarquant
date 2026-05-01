"""Lloyd-Max scalar quantizer codebook for N(0, 1/d).

Mirrors `remex.codebook.lloyd_max_codebook` and `remex.codebook.nested_codebooks`.
The Mojo version uses `+/-INF_SENTINEL = ±50` for the outer interval edges —
far enough into the tails that cdf(±50*sigma) is 1.0/0.0 in float64.
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from src.mathx import normal_cdf, normal_pdf


comptime N_ITER_DEFAULT = 300
comptime INF_SENTINEL = 50.0  # any value where cdf(x*sigma) saturates
comptime NESTED_MAX_BITS = 8   # supported max precision (matches Python)


struct Codebook(Movable):
    """Holds a Lloyd-Max codebook for a fixed (d, bits)."""
    var boundaries: UnsafePointer[Float32, MutExternalOrigin]   # length n_levels - 1
    var centroids: UnsafePointer[Float32, MutExternalOrigin]    # length n_levels
    var n_levels: Int
    var bits: Int

    def __init__(out self, bits: Int):
        self.bits = bits
        self.n_levels = 1 << bits
        # Allocate centroids first; boundaries gets at least one slot.
        self.centroids = alloc[Float32](self.n_levels)
        var n_b = self.n_levels - 1
        if n_b < 1:
            n_b = 1
        self.boundaries = alloc[Float32](n_b)

    def get_centroid(self, j: Int) -> Float32:
        return self.centroids[j]

    def get_boundary(self, j: Int) -> Float32:
        return self.boundaries[j]

    def __del__(deinit self):
        self.boundaries.free()
        self.centroids.free()


def lloyd_max_codebook(d: Int, bits: Int, n_iter: Int = N_ITER_DEFAULT) -> Codebook:
    """Build a Lloyd-Max codebook for N(0, 1/d) coordinates."""
    var n_levels = 1 << bits
    var sigma = 1.0 / sqrt(Float64(d))
    var sigma2 = sigma * sigma

    # Float64 working buffers for numerical stability
    var c = alloc[Float64](n_levels)
    var b_lo = alloc[Float64](n_levels)
    var b_hi = alloc[Float64](n_levels)

    # Initialize centroids: linspace(-3*sigma, 3*sigma, n_levels)
    if n_levels == 1:
        c[0] = 0.0
    else:
        var lo = -3.0 * sigma
        var hi = 3.0 * sigma
        var step = (hi - lo) / Float64(n_levels - 1)
        for i in range(n_levels):
            c[i] = lo + Float64(i) * step

    for _ in range(n_iter):
        # Compute bounds
        b_lo[0] = -INF_SENTINEL
        b_hi[n_levels - 1] = INF_SENTINEL
        for k in range(n_levels - 1):
            var mid = Float64(0.5) * (c[k] + c[k + 1])
            b_hi[k] = mid
            b_lo[k + 1] = mid

        # Update centroids: conditional mean on each interval
        for j in range(n_levels):
            var lo_ = b_lo[j]
            var hi_ = b_hi[j]
            var ca = normal_cdf(lo_, sigma)
            var cdfb = normal_cdf(hi_, sigma)
            var prob = cdfb - ca
            if prob > Float64(1e-15):
                var pa = normal_pdf(lo_, sigma)
                var pb = normal_pdf(hi_, sigma)
                var newc = sigma2 * (pa - pb) / prob
                c[j] = newc

    var cb = Codebook(bits)
    for j in range(n_levels):
        cb.centroids[j] = Float32(c[j])
    for j in range(n_levels - 1):
        var mid = Float64(0.5) * (c[j] + c[j + 1])
        cb.boundaries[j] = Float32(mid)

    c.free()
    b_lo.free()
    b_hi.free()
    return cb^


struct NestedCodebook(Movable):
    """Matryoshka-style nested centroid tables for precisions 1..max_bits.

    Layout: a single flat buffer holds 2 + 4 + 8 + ... + 2^max_bits = 2^(max_bits+1) - 2
    Float32 values. `offsets[bits]` is the start offset of the level for that
    precision, so the table for `bits == b` lives at `centroids[offsets[b] : offsets[b] + 2^b]`.
    """
    var centroids: UnsafePointer[Float32, MutExternalOrigin]
    # offsets[b] = start of the level for precision b (b in 1..max_bits).
    # offsets[max_bits + 1] = total length. Sized for max_bits = 8 (10 slots).
    var offsets: InlineArray[Int, 10]
    var max_bits: Int

    def __init__(out self, max_bits: Int) raises:
        if max_bits < 1 or max_bits > NESTED_MAX_BITS:
            raise Error("nested codebook: max_bits must be 1..8")
        self.max_bits = max_bits
        self.offsets = InlineArray[Int, 10](fill=0)
        var off = 0
        for b in range(1, max_bits + 1):
            self.offsets[b] = off
            off += (1 << b)
        self.offsets[max_bits + 1] = off
        self.centroids = alloc[Float32](off)
        for i in range(off):
            self.centroids[i] = Float32(0.0)

    def get_table(self, bits: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        """Pointer to the centroid table at the given precision level."""
        return self.centroids + self.offsets[bits]

    def get_centroid(self, bits: Int, idx: Int) -> Float32:
        return self.centroids[self.offsets[bits] + idx]

    def n_levels(self, bits: Int) -> Int:
        return 1 << bits

    def __del__(deinit self):
        self.centroids.free()


def nested_codebooks_from(cb: Codebook, d: Int) raises -> NestedCodebook:
    """Build nested centroid tables from an already-computed max-bits codebook.

    The Gaussian distribution is successively refinable: the top `b` bits of
    a `max_bits`-bit Lloyd-Max code are themselves a valid `b`-bit code into
    the table built by probability-weighted averaging of the corresponding
    centroid groups.

    Using a precomputed `cb` (rather than re-running Lloyd-Max) lets callers
    derive nested centroids that exactly match the centroids saved in a
    Python-written `.params` file — required for the parity test.
    """
    var max_bits = cb.bits
    var n_max = cb.n_levels
    var sigma = 1.0 / sqrt(Float64(d))
    var nested = NestedCodebook(max_bits)

    # Top level: copy the max-precision centroids verbatim.
    var max_table = nested.get_table(max_bits)
    for j in range(n_max):
        max_table[j] = cb.centroids[j]

    if max_bits == 1:
        return nested^

    # Per-bin probability mass under N(0, 1/d). Boundaries are the midpoints
    # of consecutive centroids, with ±INF_SENTINEL for the outer edges
    # (matches `remex.codebook.nested_codebooks` which uses ±inf there).
    var probs = alloc[Float64](n_max)
    var bounds = alloc[Float64](n_max + 1)
    bounds[0] = -INF_SENTINEL
    bounds[n_max] = INF_SENTINEL
    for j in range(n_max - 1):
        bounds[j + 1] = Float64(0.5) * (Float64(cb.centroids[j]) + Float64(cb.centroids[j + 1]))
    for j in range(n_max):
        probs[j] = normal_cdf(bounds[j + 1], sigma) - normal_cdf(bounds[j], sigma)

    # Coarser levels: weighted average of the max-bits centroids in each
    # contiguous group of size `n_max // n_target`.
    var target_bits = max_bits - 1
    while target_bits >= 1:
        var n_target = 1 << target_bits
        var group_size = n_max // n_target
        var target_table = nested.get_table(target_bits)
        for g in range(n_target):
            var start = g * group_size
            var total_prob: Float64 = 0.0
            var weighted: Float64 = 0.0
            var simple: Float64 = 0.0
            for k in range(group_size):
                var idx = start + k
                total_prob += probs[idx]
                weighted += probs[idx] * Float64(cb.centroids[idx])
                simple += Float64(cb.centroids[idx])
            if total_prob > 1e-15:
                target_table[g] = Float32(weighted / total_prob)
            else:
                target_table[g] = Float32(simple / Float64(group_size))
        target_bits -= 1

    probs.free()
    bounds.free()
    return nested^


def nested_codebooks(d: Int, max_bits: Int) raises -> NestedCodebook:
    """Build a Lloyd-Max codebook for `(d, max_bits)` and return its nested tables.

    Convenience wrapper around `lloyd_max_codebook` + `nested_codebooks_from`.
    For tests that need parity with a Python-written `.params` file, call
    `nested_codebooks_from(loaded_cb, d)` directly so the centroids match.
    """
    var cb = lloyd_max_codebook(d, max_bits)
    return nested_codebooks_from(cb, d)
