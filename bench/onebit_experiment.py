"""
1-bit Matryoshka cache experiment for SPECTER2.

Question: when SPECTER2 embeddings are encoded at 8-bit (durable tier in RDS),
can a 1-bit Matryoshka extraction (right-shift to MSB) serve as a fast in-memory
filter for stage-1 candidate generation in two-stage retrieval?

Sections:
  A. Standalone bit sweep: Quantizer(d=768, bits=B) for B in {1,2,3,4,8}
  B. Matryoshka extraction: encode at 8-bit, search at precision=1,2,4
  C. Two-stage: 1-bit coarse @ varying candidate budgets, 8-bit rerank
  D. Two-stage: 2-bit coarse @ varying candidate budgets, 8-bit rerank
       (architectural comparison — does 2-bit ever beat 1-bit as coarse?)

Usage:
  bash bench/fetch_specter2_cache.sh        # one-time, pulls 10k float32 cache
  ONEBIT_N=10000 python3 bench/onebit_experiment.py

ONEBIT_N defaults to 500 (smaller smoke-test). The fetcher provides a 10k
corpus; for larger N you'd need to re-encode with bench/specter2_eval.py.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from remex import Quantizer

# Read from the same cache dir bench/specter2_eval.py uses, with the same
# file names the fetcher (bench/fetch_specter2_cache.sh) writes.
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".specter2_cache")
EMB_PATH = os.path.join(CACHE_DIR, "specter2_nlp_broad.npy")

N_PAPERS = int(os.environ.get("ONEBIT_N", "500"))
D = 768


def load_embeddings(n: int) -> np.ndarray:
    """Load the published SPECTER2 NLP-broad embeddings cache. Errors if the
    cache isn't present — we don't try to re-encode here because that's a
    50-minute CPU run; use bench/fetch_specter2_cache.sh first."""
    if not os.path.exists(EMB_PATH):
        raise SystemExit(
            f"missing {EMB_PATH}\n"
            "run: bash bench/fetch_specter2_cache.sh"
        )
    emb = np.load(EMB_PATH)
    if emb.shape[0] < n:
        raise SystemExit(
            f"cache has {emb.shape[0]} vectors, ONEBIT_N={n} requested. "
            f"max ONEBIT_N for the published cache is {emb.shape[0]}."
        )
    return emb[:n]


def exact_knn(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(queries @ corpus.T), axis=1)[:, :k]


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = sum(len(set(p[:k]) & set(t[:k])) for p, t in zip(pred, truth))
    return hits / (len(pred) * k)


def main():
    print("=" * 64)
    print("  1-bit Matryoshka experiment — SPECTER2")
    print("=" * 64)

    emb = load_embeddings(N_PAPERS)
    print(f"  embeddings: {emb.shape}")

    # Distribution sanity check (post-rotation σ, flat-std metric)
    from remex.rotation import haar_rotation
    R = haar_rotation(D, 42)
    norms = np.linalg.norm(emb, axis=1)
    unit = emb / np.maximum(norms, 1e-8)[:, None]
    rot = unit @ R.T
    sig_actual = float(np.std(rot))
    sig_expected = 1.0 / np.sqrt(D)
    print(f"  σ ratio (actual/expected, flat-std metric): "
          f"{sig_actual / sig_expected:.4f}")

    # Split: 100 queries, rest is corpus
    rng = np.random.default_rng(99)
    perm = rng.permutation(emb.shape[0])
    n_q = min(100, emb.shape[0] // 5)
    queries = emb[perm[:n_q]]
    corpus = emb[perm[n_q:]]
    print(f"  corpus={corpus.shape[0]}, queries={n_q}")

    truth = exact_knn(corpus, queries, 100)

    # --- A. Standalone bit sweep ---
    print()
    print("--- A. Standalone bit sweep (independent encode per bit) ---")
    print(f"  {'bits':<6s}{'comp_ratio':>12s}{'R@10':>10s}{'R@100':>10s}"
          f"{'enc_s':>8s}{'srch_s':>8s}")
    standalone = {}
    for bits in [1, 2, 3, 4, 8]:
        pq = Quantizer(d=D, bits=bits)
        t0 = time.time()
        cv = pq.encode(corpus)
        t_enc = time.time() - t0
        t0 = time.time()
        pred, _ = pq.search_batch(cv, queries, k=100)
        t_srch = time.time() - t0
        r10 = recall_at_k(pred[:, :10], truth[:, :10], 10)
        r100 = recall_at_k(pred[:, :100], truth[:, :100], 100)
        print(f"  {bits:<6d}{cv.compression_ratio:>11.1f}x"
              f"{r10:>10.3f}{r100:>10.3f}{t_enc:>8.2f}{t_srch:>8.2f}")
        standalone[bits] = (r10, r100, cv.compression_ratio)

    # --- B. Matryoshka extraction ---
    print()
    print("--- B. Matryoshka extraction from 8-bit encoding (right-shift) ---")
    pq8 = Quantizer(d=D, bits=8)
    cv8 = pq8.encode(corpus)
    print(f"  {'precision':<11s}{'R@10':>10s}{'R@100':>10s}  vs standalone Δ")
    for prec in [1, 2, 4]:
        pred, _ = pq8.search_batch(cv8, queries, k=100, precision=prec)
        r10 = recall_at_k(pred[:, :10], truth[:, :10], 10)
        r100 = recall_at_k(pred[:, :100], truth[:, :100], 100)
        delta = r10 - standalone[prec][0]
        print(f"  {prec:<11d}{r10:>10.3f}{r100:>10.3f}   Δ@10={delta:+.3f}")

    # --- C, D. Two-stage rerank at varying candidate budgets ---
    def twostage_batch(qs, k, candidates, coarse_precision):
        preds = np.zeros((qs.shape[0], k), dtype=np.int64)
        for i, q in enumerate(qs):
            p, _ = pq8.search_twostage(
                cv8, q, k=k, candidates=candidates,
                coarse_precision=coarse_precision,
            )
            preds[i] = p
        return preds

    # Pick candidate budgets that scale with corpus: ~1%, 1.5%, 2%, 3%
    n_corpus = corpus.shape[0]
    budgets = sorted({
        max(100, int(n_corpus * f))
        for f in (0.01, 0.015, 0.02, 0.03)
    })
    budgets = [b for b in budgets if b < n_corpus]

    for section, prec in [("C. 1-bit coarse → 8-bit rerank", 1),
                          ("D. 2-bit coarse → 8-bit rerank", 2)]:
        print()
        print(f"--- {section} ---")
        print(f"  {'candidates':<12s}{'R@10':>10s}{'R@100':>10s}")
        for cand in budgets:
            pred = twostage_batch(queries, k=100, candidates=cand,
                                  coarse_precision=prec)
            r10 = recall_at_k(pred[:, :10], truth[:, :10], 10)
            r100 = recall_at_k(pred[:, :100], truth[:, :100], 100)
            print(f"  {cand:<12d}{r10:>10.3f}{r100:>10.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
