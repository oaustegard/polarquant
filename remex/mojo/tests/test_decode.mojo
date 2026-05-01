"""Parity test for `decode_batch`.

A Python runner builds a Quantizer with `(d, bits, seed)`, dumps the params,
encodes a corpus to .pq, then saves the Python-decoded output at full
precision and at a coarser precision. This test reconstructs the same
two outputs in Mojo from the loaded params + .pq + nested codebook and
asserts max abs diff < 1e-5 in both cases.

Setup (run before this test):

    python -c "
    import numpy as np
    from remex import Quantizer, save_pq, save_params

    np.random.seed(0)
    n, d, bits = 80, 16, 4
    coarse_precision = 2

    X = np.random.randn(n, d).astype(np.float32)
    q = Quantizer(d=d, bits=bits, seed=42)
    save_params('/tmp/_decode.params', q)
    cv = q.encode(X)
    save_pq('/tmp/_decode.pq', cv)

    np.save('/tmp/_decode_X.npy', X)
    np.save('/tmp/_decode_full.npy', q.decode(cv).astype(np.float32))
    np.save('/tmp/_decode_coarse.npy',
            q.decode(cv, precision=coarse_precision).astype(np.float32))
    np.save('/tmp/_decode_meta.npy',
            np.array([[coarse_precision]], dtype=np.float32))
    "
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.codebook import Codebook, NestedCodebook, nested_codebooks_from
from src.matrix import Matrix
from src.npy import load_npy_2d_f32
from src.params_format import load_params
from src.pq_format import load_pq
from src.quantizer import Quantizer, decode_batch
from src.packing import unpack


def _abs(x: Float32) -> Float32:
    return -x if x < Float32(0.0) else x


def main() raises:
    var meta = load_npy_2d_f32(String("/tmp/_decode_meta.npy"))
    var coarse_precision = Int(meta.get(0, 0))

    var pq = load_pq(String("/tmp/_decode.pq"))
    var d = pq.d
    var n = pq.n
    var bits = pq.bits

    var expected_full = load_npy_2d_f32(String("/tmp/_decode_full.npy"))
    var expected_coarse = load_npy_2d_f32(String("/tmp/_decode_coarse.npy"))
    if expected_full.rows != n or expected_full.cols != d:
        raise Error("expected_full shape mismatch")
    if expected_coarse.rows != n or expected_coarse.cols != d:
        raise Error("expected_coarse shape mismatch")

    var R = Matrix(d, d)
    var cb = Codebook(bits)
    load_params(String("/tmp/_decode.params"), R, cb)
    var nested = nested_codebooks_from(cb, d)
    var q_quant = Quantizer(R^, cb^, d, bits, UInt64(42))

    # Unpack indices once. Copy norms into a fresh buffer (UnsafePointer
    # field aliasing workaround — same as in test_encode.mojo).
    var indices = alloc[UInt8](n * d)
    unpack(pq.packed_indices, n * d, bits, indices)
    var norms_local = alloc[Float32](n)
    for i in range(n):
        norms_local[i] = pq.norms[i]

    var X_hat_full = alloc[Float32](n * d)
    decode_batch(q_quant, nested, indices, norms_local, n, 0, X_hat_full)

    var max_diff_full: Float32 = Float32(0.0)
    for i in range(n):
        for j in range(d):
            var a = X_hat_full[i * d + j]
            var b = expected_full.get(i, j)
            var diff = _abs(a - b)
            if diff > max_diff_full:
                max_diff_full = diff
    print("decode full-precision max abs diff:", max_diff_full)
    assert_true(max_diff_full < Float32(1e-5))

    var X_hat_coarse = alloc[Float32](n * d)
    decode_batch(q_quant, nested, indices, norms_local, n,
                 coarse_precision, X_hat_coarse)

    var max_diff_coarse: Float32 = Float32(0.0)
    for i in range(n):
        for j in range(d):
            var a = X_hat_coarse[i * d + j]
            var b = expected_coarse.get(i, j)
            var diff = _abs(a - b)
            if diff > max_diff_coarse:
                max_diff_coarse = diff
    print("decode coarse-precision (", coarse_precision, "bit) max abs diff:",
          max_diff_coarse)
    assert_true(max_diff_coarse < Float32(1e-5))

    indices.free()
    norms_local.free()
    X_hat_full.free()
    X_hat_coarse.free()
    print("[test_decode] parity ok — Mojo decode matches Python to max abs diff < 1e-5")
