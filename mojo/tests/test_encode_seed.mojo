"""End-to-end --seed byte-parity test for the NumPy-compatible RNG path.

Counterpart to `test_encode.mojo` (which tests --params parity). This
test verifies that `Quantizer(d, bits, seed)` in Mojo — using the new
NumPy-bit-identical RNG path (PCG64 + SeedSequence + Ziggurat) plus the
existing Householder QR — produces a `.pq` byte-identical to Python's
`save_pq(Quantizer(d, bits, seed=...).encode(X))` at 1–4 bits.

The Python test runner generates the X.npy + expected .pq and writes
them to /tmp; this binary loads them, encodes in Mojo, and asserts
byte equality.
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.npy import load_npy_2d_f32
from src.pq_format import load_pq
from src.quantizer import Quantizer, encode_batch
from src.packing import pack, unpack, packed_nbytes


def main() raises:
    var X = load_npy_2d_f32(String("/tmp/_seed_parity_X.npy"))
    var expected = load_pq(String("/tmp/_seed_parity_ref.pq"))
    var d = X.cols
    var n = X.rows
    var bits = expected.bits
    var seed = UInt64(42)

    # Build the quantizer using the NumPy-compatible RNG (default path).
    var q = Quantizer(d, bits, seed)

    # Encode. Copy X into a fresh buffer to dodge a known UnsafePointer
    # borrow oddity around Npy2D.data crossing function boundaries.
    var X_buf = alloc[Float32](n * d)
    for i in range(n):
        for j in range(d):
            X_buf[i * d + j] = X.get(i, j)

    var indices = alloc[UInt8](n * d)
    var norms_out = alloc[Float32](n)
    encode_batch(q, X_buf, n, indices, norms_out)
    X_buf.free()

    # Unpack the Python-encoded reference's packed indices for per-coord diff.
    var expected_indices = alloc[UInt8](n * d)
    unpack(expected.packed_indices, n * d, bits, expected_indices)

    var idx_mismatches = 0
    for i in range(n):
        for j in range(d):
            if indices[i * d + j] != expected_indices[i * d + j]:
                idx_mismatches += 1
    var norm_mismatches = 0
    for i in range(n):
        if norms_out[i] != expected.norms[i]:
            norm_mismatches += 1

    print("seed-parity test (d =", d, ", bits =", bits, ", seed =", seed, ")")
    print("  index mismatches:", idx_mismatches, "/", n * d)
    print("  norm mismatches:", norm_mismatches, "/", n)
    assert_equal(idx_mismatches, 0)
    assert_equal(norm_mismatches, 0)

    # Also verify the packed bytes byte-for-byte (matches what save_pq writes).
    var nb = packed_nbytes(n * d, bits)
    var packed_mojo = alloc[UInt8](nb)
    pack(indices, n * d, bits, packed_mojo)
    var packed_mismatches = 0
    for i in range(nb):
        if packed_mojo[i] != expected.packed_indices[i]:
            packed_mismatches += 1
    print("  packed-byte mismatches:", packed_mismatches, "/", nb)
    assert_equal(packed_mismatches, 0)
    expected_indices.free()

    indices.free()
    norms_out.free()
    packed_mojo.free()
    print("test_encode_seed: ok")
    print("[test_encode_seed] passed")
