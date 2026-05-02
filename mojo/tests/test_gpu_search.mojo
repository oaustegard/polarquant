"""GPU ADC search parity test.

Asserts `gpu_adc_search` returns the same top-k as the CPU `adc_search`
(rtol=1e-5 on scores, identical indices) on a synthetic corpus.

Skipped when no GPU is available (kernels are stubbed pending issue #42),
so this binary is safe to run on CI / M1 hosts.

Uses an in-process synthetic corpus rather than reading fixtures, since
the CPU result is the ground truth — no Python round-trip needed.
"""

from std.math import abs as f_abs
from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.codebook import lloyd_max_codebook
from src.matrix import Matrix
from src.quantizer import Quantizer, encode_batch, adc_search
from src.rotation import haar_rotation
from src.rng import Xoshiro256pp
from src.gpu.device import is_gpu_available
from src.gpu.adc import gpu_adc_search


def main() raises:
    if not is_gpu_available():
        print("test_gpu_search: SKIP (no GPU available — see issue #42)")
        return

    var n = 512
    var d = 64
    var bits = 4
    var k = 10
    var seed = UInt64(42)

    var rng = Xoshiro256pp(seed)
    var X = alloc[Float32](n * d)
    for i in range(n * d):
        X[i] = Float32(rng.next_normal())

    var query = alloc[Float32](d)
    for j in range(d):
        query[j] = Float32(rng.next_normal())

    var R = haar_rotation(d, seed)
    var cb = lloyd_max_codebook(d, bits)
    var q = Quantizer(R^, cb^, d, bits, seed)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)
    encode_batch(q, X, n, indices, norms)

    var cpu_idx = alloc[Int](k)
    var cpu_scores = alloc[Float32](k)
    adc_search(q, indices, norms, n, query, k, cpu_idx, cpu_scores)

    var gpu_idx = alloc[Int](k)
    var gpu_scores = alloc[Float32](k)
    gpu_adc_search(q, indices, norms, n, query, k, gpu_idx, gpu_scores)

    for i in range(k):
        assert_equal(gpu_idx[i], cpu_idx[i])
        var diff = f_abs(gpu_scores[i] - cpu_scores[i])
        var tol = Float32(1e-5) * (f_abs(cpu_scores[i]) + Float32(1.0))
        assert_true(diff <= tol)

    print("test_gpu_search: OK (n =", n, ", d =", d, ", bits =", bits, ", k =", k, ")")

    X.free()
    query.free()
    indices.free()
    norms.free()
    cpu_idx.free()
    cpu_scores.free()
    gpu_idx.free()
    gpu_scores.free()
