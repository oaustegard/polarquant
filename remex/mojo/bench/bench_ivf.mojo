"""Time IVFCoarseIndex.search_twostage at varying nprobe.

Mirrors `bench_twostage` but goes through the IVF stage-1 instead of a
flat coarse scan. Reports per-query latency at each `--nprobe` value
plus the candidate-pool size, so the latency-recall trade-off can be
read off alongside the candidate-fraction sweep that the Python
`bench/specter2_eval.py` produces.

Usage:
    bench_ivf --n N --d D --bits B --queries Q --k K
              [--n-bits NB] [--mode lsh|rotated_prefix]
              [--candidates C] [--coarse-precision K] [--seed S]

The nprobe sweep is fixed at `[1, 2, 4, 8, ..., n_cells]`, mirroring
the doubling schedule used by the Python IVF benchmark.

Default `coarse_precision` is `max(1, bits - 2)`, matching `bench_twostage`.
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.memory import alloc
from src.codebook import lloyd_max_codebook, nested_codebooks_from
from src.ivf import (
    IVFCoarseIndex,
    MODE_LSH,
    MODE_ROTATED_PREFIX,
    build_ivf,
    candidate_count,
    search_twostage,
)
from src.matrix import Matrix
from src.packed_vectors import PackedVectors, from_indices
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


def _flag_str(args: List[String], flag: String,
              default: String) raises -> String:
    var i = _arg_idx(args, flag)
    if i < 0:
        return default
    return args[i + 1]


def _doubling_sweep(n_cells: Int) -> List[Int]:
    """Generate `[1, 2, 4, 8, ..., n_cells]` for the latency sweep.

    Tracks the typical exponential nprobe schedule used in IVF benchmarks
    — each step roughly doubles the candidate pool, so the latency curve
    is readable on a log-x axis.
    """
    var out = List[Int]()
    var v = 1
    while v < n_cells:
        out.append(v)
        v = v * 2
    out.append(n_cells)
    return out^


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
    var n_bits = _flag_int(sub, String("--n-bits"), 8)
    var mode_str = _flag_str(sub, String("--mode"), String("rotated_prefix"))
    var candidates = _flag_int(sub, String("--candidates"), 500)
    var default_coarse = bits - 2 if bits - 2 >= 1 else 1
    var coarse_precision = _flag_int(sub, String("--coarse-precision"),
                                     default_coarse)
    var seed = UInt64(_flag_int(sub, String("--seed"), 42))

    var mode_int: Int
    if mode_str == "lsh":
        mode_int = MODE_LSH
    elif mode_str == "rotated_prefix":
        mode_int = MODE_ROTATED_PREFIX
    else:
        raise Error("--mode must be 'lsh' or 'rotated_prefix'")

    print("bench_ivf: n =", n, "d =", d, "bits =", bits,
          "queries =", queries, "k =", k,
          "n_bits =", n_bits, "mode =", mode_str,
          "candidates =", candidates,
          "coarse_precision =", coarse_precision)

    # Build corpus and quantizer (xoshiro path — fastest startup; cell
    # IDs aren't compared here, only latency).
    var rng = Xoshiro256pp(seed)
    var X = alloc[Float32](n * d)
    for i in range(n * d):
        X[i] = Float32(rng.next_normal())

    var R = haar_rotation(d, seed)
    var cb = lloyd_max_codebook(d, bits)
    var nested = nested_codebooks_from(cb, d)
    var q = Quantizer(R^, cb^, d, bits, seed)

    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)
    encode_batch(q, X, n, indices, norms)
    X.free()

    var packed = from_indices(indices, norms, n, d, bits)
    indices.free()
    norms.free()

    # Build the IVF index — counted separately so we can report the
    # one-time index build cost.
    var t_build0 = perf_counter_ns()
    var ivf = build_ivf(q, packed, n_bits, mode_int, seed)
    var t_build1 = perf_counter_ns()
    var build_ms = Float64(Int(t_build1 - t_build0)) / 1000000.0
    print("  build_ivf:", build_ms, "ms (n_cells =", ivf.n_cells,
          ", index_nbytes =", ivf.index_nbytes(), ")")

    # Build random queries.
    var Q_buf = alloc[Float32](queries * d)
    for i in range(queries * d):
        Q_buf[i] = Float32(rng.next_normal())

    var top_idx = alloc[Int](k)
    var top_scores = alloc[Float32](k)

    var nprobes = _doubling_sweep(ivf.n_cells)

    # Warmup once at nprobe=1.
    var qbuf0 = alloc[Float32](d)
    for j in range(d):
        qbuf0[j] = Q_buf[j]
    var _w = search_twostage(ivf, q, nested, packed, qbuf0, k, candidates, 1,
                             coarse_precision, top_idx, top_scores)
    qbuf0.free()

    print("  nprobe   ms/query   avg_candidates")
    for ni in range(len(nprobes)):
        var nprobe = nprobes[ni]
        if nprobe <= 0 or nprobe > ivf.n_cells:
            continue

        # First sweep: capture average candidate-pool size at this nprobe.
        var total_cands: Int = 0
        for qi in range(queries):
            var qbuf = alloc[Float32](d)
            for j in range(d):
                qbuf[j] = Q_buf[qi * d + j]
            total_cands += candidate_count(ivf, q, qbuf, nprobe)
            qbuf.free()
        var avg_cands = Float64(total_cands) / Float64(queries)

        # Second sweep: timed twostage search at this nprobe.
        var t0 = perf_counter_ns()
        for qi in range(queries):
            var qbuf = alloc[Float32](d)
            for j in range(d):
                qbuf[j] = Q_buf[qi * d + j]
            var _r = search_twostage(ivf, q, nested, packed, qbuf,
                                     k, candidates, nprobe,
                                     coarse_precision, top_idx, top_scores)
            qbuf.free()
        var t1 = perf_counter_ns()
        var ms_per_q = Float64(Int(t1 - t0)) / Float64(queries) / 1000000.0
        print(" ", nprobe, "  ", ms_per_q, "  ", avg_cands)

    Q_buf.free()
    top_idx.free()
    top_scores.free()
