"""Tests for `PackedVectors`: pack/unpack round-trips at all supported bit
widths plus `at_precision` parity vs Python's `PackedVectors.at_precision`.

The bit-width round-trips are self-contained (random uint8 values, mask
to `bits`, pack into PackedVectors via `from_indices`, unpack back, byte
compare). The at_precision parity is driven by a Python fixture written
once before the test runs.

Setup (run before this test):

    python -c "
    import numpy as np
    from remex import Quantizer, PackedVectors

    np.random.seed(0)
    n, d, bits = 80, 16, 4
    target_bits = 2

    X = np.random.randn(n, d).astype(np.float32)
    q = Quantizer(d=d, bits=bits, seed=42)
    cv = q.encode(X)
    packed = PackedVectors.from_compressed(cv)
    np.save('/tmp/_pv_indices.npy', cv.indices.astype(np.float32))

    packed_low = packed.at_precision(target_bits)
    np.save('/tmp/_pv_indices_at.npy',
            packed_low.unpack_rows(0, packed_low.n).astype(np.float32))

    np.save('/tmp/_pv_meta.npy',
            np.array([[n, d, bits, target_bits]], dtype=np.float32))
    "
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.npy import load_npy_2d_f32
from src.packed_vectors import (
    PackedVectors, from_indices, unpack_rows, unpack_all, unpack_at, at_precision,
)
from src.rng import Xoshiro256pp


def _roundtrip_random(n: Int, d: Int, bits: Int, seed: UInt64) raises:
    """Generate random uint8 indices, pack into PackedVectors, unpack, compare."""
    var rng = Xoshiro256pp(seed)
    var mask = UInt64((1 << bits) - 1) if bits < 8 else UInt64(0xFF)

    var indices = alloc[UInt8](n * d)
    for i in range(n * d):
        indices[i] = UInt8(rng.next_u64() & mask)
    var norms = alloc[Float32](n)
    for i in range(n):
        norms[i] = Float32(i) * Float32(0.5)

    var pv = from_indices(indices, norms, n, d, bits)

    # Round-trip 1: unpack_all
    var out_all = alloc[UInt8](n * d)
    unpack_all(pv, out_all)
    for i in range(n * d):
        assert_equal(Int(out_all[i]), Int(indices[i]))

    # Round-trip 2: unpack_rows over a sub-range
    if n >= 4:
        var sub_start = 1
        var sub_end = n - 1
        var sub_count = sub_end - sub_start
        var sub_out = alloc[UInt8](sub_count * d)
        unpack_rows(pv, sub_start, sub_end, sub_out)
        for i in range(sub_count):
            for j in range(d):
                assert_equal(Int(sub_out[i * d + j]),
                             Int(indices[(sub_start + i) * d + j]))
        sub_out.free()

    # Round-trip 3: unpack_at single row
    if n > 0:
        var single = alloc[UInt8](d)
        unpack_at(pv, n // 2, single)
        for j in range(d):
            assert_equal(Int(single[j]), Int(indices[(n // 2) * d + j]))
        single.free()

    # Norms preserved
    for i in range(n):
        assert_true(_abs(pv.norms[i] - norms[i]) < Float32(1e-7))

    indices.free()
    norms.free()
    out_all.free()


def _abs(x: Float32) -> Float32:
    return -x if x < Float32(0.0) else x


def test_roundtrips_all_widths() raises:
    """Round-trip across all supported bit widths and a mix of d values.

    The d list covers both row-aligned cases (d * bits divisible by 8) and
    row-padded cases (e.g. d=10 at 3 bits → 30 bits per row, 6 row_bytes).
    """
    var widths = [1, 2, 3, 4, 8]
    # (n, d) pairs — vary d so we hit row-aligned and row-padded layouts.
    var ns = [1, 8, 100]
    var ds = [4, 8, 10, 16, 17]
    for bw in widths:
        for n in ns:
            for d in ds:
                _roundtrip_random(n, d, bw, UInt64(bw * 1000 + d))
        print("  bits =", bw, " roundtrips ok")


def test_at_precision_parity() raises:
    """Match Python's `PackedVectors.at_precision(target_bits).unpack_rows()`.

    Loads the n, d, bits, target_bits and the expected unpacked indices
    written by the Python fixture above, builds a Mojo PackedVectors from
    the original indices, calls `at_precision(target_bits)`, and asserts
    every uint8 cell matches.
    """
    var meta = load_npy_2d_f32(String("/tmp/_pv_meta.npy"))
    var n = Int(meta.get(0, 0))
    var d = Int(meta.get(0, 1))
    var bits = Int(meta.get(0, 2))
    var target_bits = Int(meta.get(0, 3))

    var orig = load_npy_2d_f32(String("/tmp/_pv_indices.npy"))
    var expected_at = load_npy_2d_f32(String("/tmp/_pv_indices_at.npy"))
    if orig.rows != n or orig.cols != d:
        raise Error("test_at_precision_parity: orig shape mismatch")
    if expected_at.rows != n or expected_at.cols != d:
        raise Error("test_at_precision_parity: expected_at shape mismatch")

    # Build indices buffer from the float32 npy (values are 0..2^bits-1).
    var indices = alloc[UInt8](n * d)
    for i in range(n):
        for j in range(d):
            indices[i * d + j] = UInt8(Int(orig.get(i, j)))
    var norms = alloc[Float32](n)
    for i in range(n):
        norms[i] = Float32(0.0)

    var pv = from_indices(indices, norms, n, d, bits)
    var pv_low = at_precision(pv, target_bits)
    assert_equal(pv_low.bits, target_bits)
    assert_equal(pv_low.n, n)
    assert_equal(pv_low.d, d)

    var unpacked = alloc[UInt8](n * d)
    unpack_all(pv_low, unpacked)

    var n_diff: Int = 0
    for i in range(n):
        for j in range(d):
            var got = Int(unpacked[i * d + j])
            var exp = Int(expected_at.get(i, j))
            if got != exp:
                n_diff += 1
    print("at_precision(", target_bits, ") cell mismatches:", n_diff,
          "out of", n * d)
    assert_equal(n_diff, 0)

    # at_precision(self.bits) returns a copy that round-trips identically.
    var pv_same = at_precision(pv, bits)
    var roundtrip = alloc[UInt8](n * d)
    unpack_all(pv_same, roundtrip)
    var n_diff_same: Int = 0
    for i in range(n * d):
        if Int(roundtrip[i]) != Int(indices[i]):
            n_diff_same += 1
    assert_equal(n_diff_same, 0)
    print("at_precision(bits) self-copy round-trip ok")

    indices.free()
    norms.free()
    unpacked.free()
    roundtrip.free()


def main() raises:
    test_roundtrips_all_widths()
    test_at_precision_parity()
    print("[test_packed_vectors] all passed")
