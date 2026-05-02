"""Time `gpu_encode_batch` on a synthetic (n, d) corpus.

Usage: bench_gpu_encode --n N --d D --bits B [--seed S]

Mirrors `bench_encode.mojo`. Skips with a clear message when no GPU is
reachable (the kernels are stubbed pending issue #42).

The Mojo CPU bench is the wrong baseline for this — issue #42 wants the
new `bench/RESULTS.md § Mojo port` GPU row compared against a
CuPy/PyTorch run on the *same* GPU host. The Python-side baseline
generation is left to `bench/compare.py` once the kernels land.
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.memory import alloc
from src.codebook import lloyd_max_codebook
from src.matrix import Matrix
from src.quantizer import Quantizer
from src.rotation import haar_rotation
from src.rng import Xoshiro256pp
from src.gpu.device import is_gpu_available
from src.gpu.encode import gpu_encode_batch


def _arg_idx(args: List[String], flag: String) -> Int:
    for i in range(len(args)):
        if args[i] == flag:
            return i
    return -1


def _flag_int(args: List[String], flag: String, default: Int) raises -> Int:
    var i = _arg_idx(args, flag)
    if i < 0:
        return default
    return Int(args[i + 1])


def main() raises:
    var args = argv()
    var sub = List[String]()
    for i in range(1, len(args)):
        sub.append(String(args[i]))

    if not is_gpu_available():
        print("bench_gpu_encode: SKIP (no GPU available — see issue #42)")
        return

    var n = _flag_int(sub, String("--n"), 10000)
    var d = _flag_int(sub, String("--d"), 384)
    var bits = _flag_int(sub, String("--bits"), 4)
    var seed = UInt64(_flag_int(sub, String("--seed"), 42))

    print("bench_gpu_encode: n =", n, "d =", d, "bits =", bits)

    var rng = Xoshiro256pp(seed)
    var X = alloc[Float32](n * d)
    for i in range(n * d):
        X[i] = Float32(rng.next_normal())

    var R = haar_rotation(d, seed)
    var cb = lloyd_max_codebook(d, bits)
    var q = Quantizer(R^, cb^, d, bits, seed)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)

    # Warmup (drives lazy device init + first kernel JIT).
    gpu_encode_batch(q, X, n, indices, norms)

    var t0 = perf_counter_ns()
    gpu_encode_batch(q, X, n, indices, norms)
    var t1 = perf_counter_ns()
    var dt_ns = Int(t1 - t0)
    var us_per_vec = Float64(dt_ns) / Float64(n) / 1000.0
    print("  encode time:", dt_ns / 1000000, "ms total,", us_per_vec, "us / vector")

    X.free()
    indices.free()
    norms.free()
