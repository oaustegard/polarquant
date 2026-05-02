"""Parity test for `search_twostage`.

A Python runner builds a Quantizer with `(d, bits, seed)`, dumps the params,
encodes a corpus to .pq, and saves a set of queries plus reference
(top-k indices, top-k scores) tuples produced by Python's
`Quantizer.search_twostage`. This test re-runs the same workload in Mojo
against the loaded params + indices + nested codebook and asserts that
the top-k order matches and scores agree to rtol=1e-5.

Setup (run before this test):

    python -c "
    import numpy as np
    from remex import Quantizer, save_pq, save_params

    np.random.seed(0)
    n, d, bits = 200, 16, 4
    n_q, k, candidates, coarse_precision = 8, 5, 50, 2

    X = np.random.randn(n, d).astype(np.float32)
    Q = np.random.randn(n_q, d).astype(np.float32)

    q = Quantizer(d=d, bits=bits, seed=42)
    save_params('/tmp/_twostage.params', q)
    cv = q.encode(X)
    save_pq('/tmp/_twostage.pq', cv)

    np.save('/tmp/_twostage_X.npy', X)
    np.save('/tmp/_twostage_Q.npy', Q)

    expected_idx = np.zeros((n_q, k), dtype=np.float32)
    expected_scores = np.zeros((n_q, k), dtype=np.float32)
    for i in range(n_q):
        ti, ts = q.search_twostage(
            cv, Q[i], k=k, candidates=candidates,
            coarse_precision=coarse_precision)
        expected_idx[i] = ti.astype(np.float32)
        expected_scores[i] = ts

    # Save metadata as a (1, 4) float32 row: [k, candidates, coarse_precision, n_q]
    meta = np.array([[k, candidates, coarse_precision, n_q]], dtype=np.float32)
    np.save('/tmp/_twostage_meta.npy', meta)
    np.save('/tmp/_twostage_expected_idx.npy', expected_idx)
    np.save('/tmp/_twostage_expected_scores.npy', expected_scores)
    "
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc
from src.codebook import Codebook, NestedCodebook, nested_codebooks_from
from src.matrix import Matrix
from src.npy import load_npy_2d_f32
from src.params_format import load_params
from src.pq_format import load_pq
from src.quantizer import Quantizer, search_twostage
from src.packing import unpack


def _abs(x: Float32) -> Float32:
    return -x if x < Float32(0.0) else x


def main() raises:
    var meta = load_npy_2d_f32(String("/tmp/_twostage_meta.npy"))
    var k = Int(meta.get(0, 0))
    var candidates = Int(meta.get(0, 1))
    var coarse_precision = Int(meta.get(0, 2))
    var n_q = Int(meta.get(0, 3))

    var pq = load_pq(String("/tmp/_twostage.pq"))
    var d = pq.d
    var n = pq.n
    var bits = pq.bits

    var Q = load_npy_2d_f32(String("/tmp/_twostage_Q.npy"))
    if Q.rows != n_q or Q.cols != d:
        raise Error("query shape mismatch with metadata + .pq")

    var expected_idx = load_npy_2d_f32(String("/tmp/_twostage_expected_idx.npy"))
    var expected_scores = load_npy_2d_f32(String("/tmp/_twostage_expected_scores.npy"))
    if expected_idx.rows != n_q or expected_idx.cols != k:
        raise Error("expected_idx shape mismatch")
    if expected_scores.rows != n_q or expected_scores.cols != k:
        raise Error("expected_scores shape mismatch")

    # Load (R, max-bits codebook) from the same .params file Python wrote.
    var R = Matrix(d, d)
    var cb = Codebook(bits)
    load_params(String("/tmp/_twostage.params"), R, cb)

    # Derive nested centroid tables from the loaded codebook so they match
    # Python's `_nested` exactly (both built from the same max-bits centroids).
    var nested = nested_codebooks_from(cb, d)

    var q_quant = Quantizer(R^, cb^, d, bits, UInt64(42))

    # Unpack indices once. Copy norms into a fresh buffer (UnsafePointer field
    # aliasing workaround — see `remex.mojo` cmd_search).
    var indices = alloc[UInt8](n * d)
    unpack(pq.packed_indices, n * d, bits, indices)
    var norms_local = alloc[Float32](n)
    for i in range(n):
        norms_local[i] = pq.norms[i]

    var top_idx = alloc[Int](k)
    var top_scores = alloc[Float32](k)

    var max_score_diff: Float32 = Float32(0.0)
    var n_idx_mismatch: Int = 0

    for qi in range(n_q):
        # Copy the qi-th query into a fresh buffer.
        var qbuf = alloc[Float32](d)
        for j in range(d):
            qbuf[j] = Q.get(qi, j)

        search_twostage(
            q_quant, nested, indices, norms_local, n,
            qbuf, k, candidates, coarse_precision,
            top_idx, top_scores,
        )

        for r in range(k):
            var got_idx = top_idx[r]
            var exp_idx = Int(expected_idx.get(qi, r))
            if got_idx != exp_idx:
                n_idx_mismatch += 1
                print("query", qi, "rank", r, ": got idx =", got_idx,
                      "expected =", exp_idx,
                      " (got score =", top_scores[r],
                      "expected score =", expected_scores.get(qi, r), ")")
            var got_s = top_scores[r]
            var exp_s = expected_scores.get(qi, r)
            var rel = _abs(got_s - exp_s)
            var ref_mag = _abs(exp_s)
            if ref_mag > Float32(1.0):
                rel = rel / ref_mag
            if rel > max_score_diff:
                max_score_diff = rel

        qbuf.free()

    print("query count:", n_q, "k =", k,
          "candidates =", candidates, "coarse_precision =", coarse_precision)
    print("max relative score diff:", max_score_diff)
    print("idx mismatches:", n_idx_mismatch)
    assert_equal(n_idx_mismatch, 0)
    assert_true(max_score_diff < Float32(1e-5))

    indices.free()
    norms_local.free()
    top_idx.free()
    top_scores.free()
    print("[test_search_twostage] parity ok — Mojo top-k matches Python to rtol=1e-5")
