"""Time search_twostage on a synthetic compressed corpus.

Usage: bench_twostage --n N --d D --bits B --queries Q --k K
                      [--candidates C] [--coarse-precision K] [--seed S]

Default `coarse_precision` is `max(1, bits - 2)`, mirroring the Python
default in `Quantizer.search_twostage`.
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.memory import alloc
from src.codebook import lloyd_max_codebook, nested_codebooks_from
from src.matrix import Matrix
from src.quantizer import Quantizer, encode_batch, search_twostage
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
    var queries = _flag_int(sub, String("--queries"), 100)
    var k = _flag_int(sub, String("--k"), 10)
    var candidates = _flag_int(sub, String("--candidates"), 500)
    var default_coarse = bits - 2 if bits - 2 >= 1 else 1
    var coarse_precision = _flag_int(sub, String("--coarse-precision"), default_coarse)
    var seed = UInt64(_flag_int(sub, String("--seed"), 42))

    print("bench_twostage: n =", n, "d =", d, "bits =", bits,
          "queries =", queries, "k =", k,
          "candidates =", candidates,
          "coarse_precision =", coarse_precision)

    # Build corpus.
    var rng = Xoshiro256pp(seed)
    var X = alloc[Float32](n * d)
    for i in range(n * d):
        X[i] = Float32(rng.next_normal())

    var R = haar_rotation(d, seed)
    var cb = lloyd_max_codebook(d, bits)
    # Build nested *before* moving cb into the Quantizer.
    var nested = nested_codebooks_from(cb, d)
    var q = Quantizer(R^, cb^, d, bits, seed)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)
    encode_batch(q, X, n, indices, norms)
    X.free()

    # Build random queries.
    var Q_buf = alloc[Float32](queries * d)
    for i in range(queries * d):
        Q_buf[i] = Float32(rng.next_normal())

    var top_idx = alloc[Int](k)
    var top_scores = alloc[Float32](k)

    # Warmup.
    var qbuf0 = alloc[Float32](d)
    for j in range(d):
        qbuf0[j] = Q_buf[j]
    search_twostage(q, nested, indices, norms, n, qbuf0,
                    k, candidates, coarse_precision, top_idx, top_scores)
    qbuf0.free()

    var t0 = perf_counter_ns()
    for qi in range(queries):
        var qbuf = alloc[Float32](d)
        for j in range(d):
            qbuf[j] = Q_buf[qi * d + j]
        search_twostage(q, nested, indices, norms, n, qbuf,
                        k, candidates, coarse_precision, top_idx, top_scores)
        qbuf.free()
    var t1 = perf_counter_ns()
    var dt_ns = Int(t1 - t0)
    var ms_per_q = Float64(dt_ns) / Float64(queries) / 1000000.0
    print("  search time:", dt_ns / 1000000, "ms total,", ms_per_q, "ms / query")

    indices.free()
    norms.free()
    Q_buf.free()
    top_idx.free()
    top_scores.free()
