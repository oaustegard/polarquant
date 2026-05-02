"""GPU encode parity test.

Mirrors `test_encode.mojo`: loads the same Python-generated
`(R, codebook, X, expected_pq)` fixtures, encodes via `gpu_encode_batch`,
and asserts the packed indices match the Python reference (modulo
documented FP-order tolerance — see issue #42).

Skipped when no GPU is available (the kernels are stubbed pending #42),
so this binary is safe to run on CI / M1 hosts and only does real work
on a MAX-supported GPU host.

Fixtures expected at:
  /tmp/_parity_X.npy
  /tmp/_parity.params
  /tmp/_parity_ref.pq
(Generate with the snippet in remex/mojo/README.md § Tests.)
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.codebook import Codebook
from src.matrix import Matrix
from src.npy import load_npy_2d_f32
from src.params_format import load_params
from src.pq_format import load_pq
from src.quantizer import Quantizer
from src.packing import pack
from src.gpu.device import is_gpu_available
from src.gpu.encode import gpu_encode_batch


def main() raises:
    if not is_gpu_available():
        print("test_gpu_encode: SKIP (no GPU available — see issue #42)")
        return

    var X = load_npy_2d_f32(String("/tmp/_parity_X.npy"))
    var expected = load_pq(String("/tmp/_parity_ref.pq"))
    var d = X.cols
    var n = X.rows
    var bits = expected.bits

    var R = Matrix(d, d)
    var cb = Codebook(bits)
    load_params(String("/tmp/_parity.params"), R, cb)

    var q = Quantizer(R^, cb^, d, bits, UInt64(0))

    var X_buf = alloc[Float32](n * d)
    for i in range(n):
        for j in range(d):
            X_buf[i * d + j] = X.get(i, j)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)
    gpu_encode_batch(q, X_buf, n, indices, norms)
    X_buf.free()

    var packed = alloc[UInt8](expected.packed_bytes)
    pack(indices, n * d, bits, packed)

    # Byte-identical packed indices vs Python reference. A blocked GPU
    # GEMM may flip a coordinate that sits on a boundary; if the strict
    # check fails on a real GPU run, switch to a "fraction-of-coordinates
    # within tolerance" assertion and document the threshold.
    for i in range(expected.packed_bytes):
        assert_equal(Int(packed[i]), Int(expected.packed_indices[i]))

    print("test_gpu_encode: OK (", n, "vectors, d =", d, ", bits =", bits, ")")

    indices.free()
    norms.free()
    packed.free()
