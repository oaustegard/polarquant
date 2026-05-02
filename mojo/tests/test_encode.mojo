"""Encode parity test.

The Python test runner generates a Quantizer's R + codebook (via Python)
and dumps them to /tmp/_parity.params, plus an X.npy of test vectors and
the Python-encoded indices and norms (as a .pq file). This test loads
those, encodes X with the same R + codebook in Mojo, and asserts the
packed output is byte-identical.
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.codebook import Codebook
from src.matrix import Matrix
from src.npy import load_npy_2d_f32
from src.params_format import load_params
from src.pq_format import load_pq, save_pq
from src.quantizer import Quantizer, encode_batch
from src.packing import pack, packed_nbytes


def main() raises:
    # Load matching Python-generated params + X + expectederence .pq
    var X = load_npy_2d_f32(String("/tmp/_parity_X.npy"))
    var expected = load_pq(String("/tmp/_parity_ref.pq"))
    var d = X.cols
    var n = X.rows
    var bits = expected.bits

    var R = Matrix(d, d)
    var cb = Codebook(bits)
    load_params(String("/tmp/_parity.params"), R, cb)

    var q = Quantizer(R^, cb^, d, bits, UInt64(0))

    # Encode in Mojo. Copy X into a fresh buffer to dodge any borrow weirdness
    # on Npy2D.data crossing the function boundary.
    var X_buf = alloc[Float32](n * d)
    for i in range(n):
        for j in range(d):
            X_buf[i * d + j] = X.get(i, j)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)
    encode_batch(q, X_buf, n, indices, norms)
    X_buf.free()

    # Compare unpacked indices first (easier to debug than packed)
    # Build a temp packed of mojo indices, then compare to expected.packed_indices
    var packed = alloc[UInt8](expected.packed_bytes)
    pack(indices, n * d, bits, packed)

    var max_norm_diff: Float32 = Float32(0.0)
    for i in range(n):
        var a: Float32 = norms[i]
        var b: Float32 = expected.norms[i]
        var diff = a - b
        if diff < Float32(0.0):
            diff = -diff
        if diff > max_norm_diff:
            max_norm_diff = diff
    print("max norm diff (Mojo vs Python):", max_norm_diff)
    assert_true(max_norm_diff < Float32(1e-5))

    var n_diff: Int = 0
    for i in range(expected.packed_bytes):
        if packed[i] != expected.packed_indices[i]:
            n_diff += 1
    print("packed-byte differences:", n_diff, "out of", expected.packed_bytes)
    assert_equal(n_diff, 0)

    indices.free()
    norms.free()
    packed.free()
    print("[test_encode] parity ok — Mojo encode is bit-identical to Python")
