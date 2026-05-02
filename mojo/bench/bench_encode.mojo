"""Time encode_batch on a synthetic (n, d) corpus.

Usage: bench_encode --n N --d D --bits B [--seed S]
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.memory import alloc
from src.codebook import lloyd_max_codebook
from src.matrix import Matrix
from src.quantizer import Quantizer, encode_batch
from src.rotation import haar_rotation
from src.rng import Xoshiro256pp


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

    var n = _flag_int(sub, String("--n"), 10000)
    var d = _flag_int(sub, String("--d"), 384)
    var bits = _flag_int(sub, String("--bits"), 4)
    var seed = UInt64(_flag_int(sub, String("--seed"), 42))

    print("bench_encode: n =", n, "d =", d, "bits =", bits)

    # Synthesize standard-normal X
    var rng = Xoshiro256pp(seed)
    var X = alloc[Float32](n * d)
    for i in range(n * d):
        X[i] = Float32(rng.next_normal())

    # Build quantizer (one-time setup, not timed)
    var R = haar_rotation(d, seed)
    var cb = lloyd_max_codebook(d, bits)
    var q = Quantizer(R^, cb^, d, bits, seed)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)

    # Warmup
    encode_batch(q, X, n, indices, norms)

    var t0 = perf_counter_ns()
    encode_batch(q, X, n, indices, norms)
    var t1 = perf_counter_ns()
    var dt_ns = Int(t1 - t0)
    var us_per_vec = Float64(dt_ns) / Float64(n) / 1000.0
    print("  encode time:", dt_ns / 1000000, "ms total,", us_per_vec, "us / vector")

    X.free()
    indices.free()
    norms.free()
