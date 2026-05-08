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

    # Byte-identical packed indices vs Python reference, modulo a
    # documented FP-order tolerance. The Metal FMA pipeline fuses
    # multiply-add into a single rounding step; the CPU SIMD path does
    # two roundings. For coordinates within ~1 ULP of a quantization
    # boundary this can flip the resulting nibble by 1. Issue #42
    # acceptance: count mismatched bytes as a fraction of total and
    # require it stay below the threshold below.
    var mismatched_bytes = 0
    var max_diff = 0
    for i in range(expected.packed_bytes):
        var got = Int(packed[i])
        var want = Int(expected.packed_indices[i])
        if got != want:
            mismatched_bytes += 1
            var delta = got - want if got > want else want - got
            if delta > max_diff:
                max_diff = delta

    var pct = Float64(mismatched_bytes) / Float64(expected.packed_bytes) * 100.0
    print("test_gpu_encode: OK (", n, "vectors, d =", d, ", bits =", bits,
          "); diverged bytes =", mismatched_bytes, "/", expected.packed_bytes,
          "(", pct, "%); max byte delta =", max_diff)

    # 0.5% byte-divergence cap — well within the issue #42 "boundary-adjacent
    # coordinates" expectation. Tighten if a future kernel does FMA-aware
    # accumulation; loosen only with a written rationale.
    assert_true(pct < Float64(0.5),
                String("GPU encode divergence above 0.5% tolerance"))

    indices.free()
    norms.free()
    packed.free()
