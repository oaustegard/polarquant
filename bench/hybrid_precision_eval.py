"""
Hybrid-precision quantization investigation (issue #34).

Hypothesis (ported from OjaKV arXiv:2509.21623): the per-vector
reconstruction error under 2-bit TurboQuant is heavy-tailed. If so, a
small fraction of high-residual vectors carries most of the error mass
and keeping them at higher precision should yield outsized recall at
near-flat memory cost.

This script runs the three-step investigation from the issue:

  Step 1: characterize the residual-error distribution on a real
          embedding corpus (SPECTER2-encoded abstracts) and a synthetic
          Gaussian baseline. Prints tail-mass statistics and KS / skew
          / kurtosis diagnostics; optionally saves histograms.

  Gate:   if the top-10% of vectors carry <40% of total squared error
          mass, the hypothesis is dead and Step 2 is skipped.

  Step 2: construct a hybrid-precision index — top-k% highest-residual
          vectors stored at higher precision, rest at 2-bit — and
          compare recall@{10,100} vs uniform 2/3/4-bit baselines at
          matched bits/vector.

  Step 3: decision is printed based on whether hybrid beats
          uniform-at-same-bits-per-vector by >2 recall@10 points.

The hybrid search implementation is a bench-only helper that splits
the corpus into two precision tiers, calls remex.Quantizer.search_adc
on each, and merges the top-k. No production code changes.

Usage:

    # Full run, synthetic baseline only (always works, fast)
    python bench/hybrid_precision_eval.py --synthetic-only

    # With cached SPECTER2 corpus (run specter2_eval.py first to cache)
    python bench/hybrid_precision_eval.py --specter2 --plots

    # Custom scenario
    python bench/hybrid_precision_eval.py \
        --specter2 --k-pct 10 --high-bits 4 --low-bits 2
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from remex import Quantizer
from remex.codebook import lloyd_max_codebook

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, ".specter2_cache")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "hybrid_results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


# ---------------------------------------------------------------------------
# Step 1 — residual-error characterization
# ---------------------------------------------------------------------------


def residual_errors(
    corpus: np.ndarray, bits: int, seed: int = 42
) -> tuple[np.ndarray, Quantizer]:
    """Encode + decode at `bits` precision and return relative L2 residuals.

    err[i] = ||x[i] - dequantize(quantize(x[i]))||_2 / ||x[i]||_2
    """
    d = corpus.shape[1]
    q = Quantizer(d=d, bits=bits, seed=seed)
    compressed = q.encode(corpus)
    x_hat = q.decode(compressed)

    diff = corpus.astype(np.float32) - x_hat
    abs_err = np.linalg.norm(diff, axis=1)
    norms = np.linalg.norm(corpus, axis=1)
    rel_err = abs_err / np.maximum(norms, 1e-12)
    return rel_err, q


def tail_mass_stats(rel_err: np.ndarray) -> dict:
    """How much of total squared-error mass sits in the top p% of vectors."""
    n = len(rel_err)
    sq = rel_err.astype(np.float64) ** 2
    total = float(sq.sum())
    order = np.argsort(-rel_err)  # worst-first
    sorted_sq = sq[order]
    cum = np.cumsum(sorted_sq)

    pcts = [1, 5, 10, 20, 30, 50]
    masses = {}
    for p in pcts:
        k = max(1, int(np.ceil(n * p / 100)))
        masses[p] = float(cum[k - 1] / total) if total > 0 else 0.0

    return {
        "total_sq_error": total,
        "rel_err_mean": float(rel_err.mean()),
        "rel_err_median": float(np.median(rel_err)),
        "rel_err_std": float(rel_err.std()),
        "rel_err_min": float(rel_err.min()),
        "rel_err_max": float(rel_err.max()),
        "rel_err_p95": float(np.percentile(rel_err, 95)),
        "rel_err_p99": float(np.percentile(rel_err, 99)),
        "skewness": float(stats.skew(rel_err)),
        "excess_kurtosis": float(stats.kurtosis(rel_err, fisher=True)),
        "mass_top_1pct": masses[1],
        "mass_top_5pct": masses[5],
        "mass_top_10pct": masses[10],
        "mass_top_20pct": masses[20],
        "mass_top_30pct": masses[30],
        "mass_top_50pct": masses[50],
        "n": n,
    }


def print_step1(label: str, bits: int, d: int, rel_err: np.ndarray) -> dict:
    stats_ = tail_mass_stats(rel_err)
    print(f"\n{'='*60}")
    print(f"  Step 1 — residual-error distribution")
    print(f"  Corpus: {label}  (n={rel_err.shape[0]}, d={d})")
    print(f"  Quantization: {bits}-bit TurboQuant (Lloyd-Max + Haar rotation)")
    print(f"{'='*60}")
    print(f"  Relative L2 error  (||x - x̂|| / ||x||):")
    print(f"    min / median / mean / max : "
          f"{stats_['rel_err_min']:.4f} / {stats_['rel_err_median']:.4f} / "
          f"{stats_['rel_err_mean']:.4f} / {stats_['rel_err_max']:.4f}")
    print(f"    std                       : {stats_['rel_err_std']:.4f}")
    print(f"    p95 / p99                 : "
          f"{stats_['rel_err_p95']:.4f} / {stats_['rel_err_p99']:.4f}")
    print(f"    skewness / excess kurt.   : "
          f"{stats_['skewness']:.3f} / {stats_['excess_kurtosis']:.3f}")
    print()
    print(f"  Squared-error tail mass (fraction of total ||err||² carried "
          f"by top-p% worst vectors):")
    for p in (1, 5, 10, 20, 30, 50):
        uniform = p / 100
        concentration = stats_[f"mass_top_{p}pct"] / uniform
        print(f"    top {p:>2d}%  →  {stats_[f'mass_top_{p}pct']:.3f}   "
              f"({concentration:.2f}x uniform)")
    print()
    # Gate
    heavy = stats_["mass_top_10pct"] > 0.40
    print(f"  Hypothesis gate (>40% mass in top 10%): "
          f"{'HOLDS — proceed to Step 2' if heavy else 'FAILS — tail too flat'}")

    return {"label": label, "bits": bits, "d": d, **stats_, "gate_holds": heavy}


def save_residual_plot(label: str, rel_err: np.ndarray, bits: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed, skipping plot)")
        return None

    os.makedirs(PLOTS_DIR, exist_ok=True)
    safe = label.lower().replace(" ", "_").replace("/", "_")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(rel_err, bins=80, alpha=0.75, color="steelblue", edgecolor="white")
    axes[0].axvline(np.median(rel_err), color="k", ls="--", lw=1, label=f"median={np.median(rel_err):.3f}")
    axes[0].axvline(np.percentile(rel_err, 95), color="red", ls="--", lw=1, label=f"p95={np.percentile(rel_err, 95):.3f}")
    axes[0].set_xlabel("relative L2 error  ||x - x̂|| / ||x||")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{label} — residual histogram ({bits}-bit)")
    axes[0].legend()

    # Lorenz-style tail mass curve
    n = len(rel_err)
    sq = (rel_err.astype(np.float64) ** 2)
    order = np.argsort(-rel_err)
    cum = np.cumsum(sq[order]) / sq.sum()
    frac_vectors = np.arange(1, n + 1) / n
    axes[1].plot(frac_vectors * 100, cum, color="darkorange", lw=2, label="cumulative error mass")
    axes[1].plot([0, 100], [0, 1], color="gray", ls=":", lw=1, label="uniform")
    for p in (10, 20):
        k = max(1, int(np.ceil(n * p / 100)))
        axes[1].axvline(p, color="k", ls="--", lw=0.6, alpha=0.5)
        axes[1].annotate(f"{cum[k-1]*100:.0f}% of error",
                         xy=(p, cum[k-1]), xytext=(p + 2, cum[k-1] - 0.08),
                         fontsize=9)
    axes[1].set_xlabel("top-p% worst vectors")
    axes[1].set_ylabel("fraction of total squared error")
    axes[1].set_title(f"{label} — tail-mass curve")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{safe}_{bits}bit_residuals.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Step 2 — hybrid-precision benchmark
# ---------------------------------------------------------------------------


def exact_knn(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    scores = queries @ corpus.T
    return np.argsort(-scores, axis=1)[:, :k]


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k]) & set(t[:k]))
    return hits / (len(pred) * k)


def hybrid_scores(
    corpus: np.ndarray,
    queries: np.ndarray,
    high_bits: int,
    low_bits: int,
    k_pct: float,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Score queries against a hybrid-precision index.

    Uses a **single Matryoshka-nested Quantizer** at ``high_bits`` so the
    codebooks across tiers are probability-consistent. Scoring the two
    tiers with independently-fit Lloyd-Max codebooks at different bit
    widths would produce score shifts (on SPECTER2, 2-bit means ~10%
    smaller than exact, while 8-bit means match exactly) — merging such
    scores without calibration pushes low-tier vectors below the full
    high-tier regardless of relevance.

    Protocol:
      1. Compute per-vector residual under ``low_bits`` to pick the tail.
      2. Top ``k_pct`` of vectors by residual → score at ``high_bits``.
      3. Remaining vectors → score at ``low_bits`` (same Quantizer, via
         Matryoshka right-shift; same Haar rotation; centroids from the
         same nested table).
      4. Concatenate scores into corpus order.

    Memory accounting: high tier stored at ``high_bits``, low tier
    right-shifted to ``low_bits`` for bit-packed storage. Average
    bits/vector = k * high_bits + (1 - k) * low_bits.

    Returns (scores: (n_queries, n), avg_bits_per_vector).
    """
    assert 0 < k_pct < 1
    d = corpus.shape[1]
    n = corpus.shape[0]

    # Tier selection — quick 2-bit reconstruction error picks the tail.
    rel_err_low, _ = residual_errors(corpus, low_bits, seed=seed)
    n_high = max(1, int(np.ceil(n * k_pct)))
    high_idx = np.argpartition(-rel_err_low, n_high)[:n_high]
    high_mask = np.zeros(n, dtype=bool)
    high_mask[high_idx] = True
    low_idx = np.where(~high_mask)[0]

    # Single Quantizer at high_bits; Matryoshka handles low_bits via
    # right-shift of the same indices with nested centroids.
    q = Quantizer(d=d, bits=high_bits, seed=seed)
    compressed = q.encode(corpus)
    comp_high = compressed.subset(high_idx)
    comp_low = compressed.subset(low_idx)

    all_scores = np.empty((queries.shape[0], n), dtype=np.float32)
    for qi, query in enumerate(queries):
        idx_h, sc_h = q.search_adc(
            comp_high, query, k=len(high_idx), precision=None
        )
        idx_l, sc_l = q.search_adc(
            comp_low, query, k=len(low_idx), precision=low_bits
        )
        scores_row = np.empty(n, dtype=np.float32)
        scores_row[high_idx[idx_h]] = sc_h
        scores_row[low_idx[idx_l]] = sc_l
        all_scores[qi] = scores_row

    avg_bits = (n_high * high_bits + (n - n_high) * low_bits) / n
    return all_scores, avg_bits


def uniform_scores(
    corpus: np.ndarray, queries: np.ndarray, bits: int, seed: int = 42
) -> np.ndarray:
    d = corpus.shape[1]
    q = Quantizer(d=d, bits=bits, seed=seed)
    comp = q.encode(corpus)
    n = corpus.shape[0]
    all_scores = np.empty((queries.shape[0], n), dtype=np.float32)
    for qi, query in enumerate(queries):
        idx, sc = q.search_adc(comp, query, k=n)
        row = np.empty(n, dtype=np.float32)
        row[idx] = sc
        all_scores[qi] = row
    return all_scores


def top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((scores.shape[0], k), dtype=np.int64)
    for i in range(scores.shape[0]):
        row = scores[i]
        if k >= len(row):
            out[i] = np.argsort(-row)[:k]
        else:
            part = np.argpartition(-row, k)[:k]
            out[i] = part[np.argsort(-row[part])]
    return out


def run_step2(
    corpus: np.ndarray,
    label: str,
    k_pct: float,
    high_bits: int,
    low_bits: int,
    n_queries: int = 200,
    max_k: int = 100,
    seed: int = 99,
) -> dict:
    rng = np.random.default_rng(seed)
    n = corpus.shape[0]
    perm = rng.permutation(n)
    n_queries = min(n_queries, n // 5)
    query_idx = perm[:n_queries]
    corpus_idx = perm[n_queries:]
    queries = corpus[query_idx]
    search_corpus = corpus[corpus_idx]

    print(f"\n{'='*60}")
    print(f"  Step 2 — recall benchmark: {label}")
    print(f"  corpus {search_corpus.shape[0]}  queries {queries.shape[0]}  "
          f"d={corpus.shape[1]}")
    print(f"  hybrid: top {k_pct*100:.0f}% at {high_bits}-bit, "
          f"rest at {low_bits}-bit")
    print(f"{'='*60}")

    truth = exact_knn(search_corpus, queries, max_k)

    results = []

    # Uniform baselines at 2, 3, 4 bits (skip if > high_bits)
    for bits in sorted({low_bits, max(low_bits, low_bits + 1), high_bits}):
        if bits in (5, 6, 7):  # remex restriction
            continue
        t0 = time.time()
        scores = uniform_scores(search_corpus, queries, bits)
        pred = top_k_from_scores(scores, max_k)
        r10 = recall_at_k(pred[:, :10], truth[:, :10], 10)
        r100 = recall_at_k(pred[:, :100], truth[:, :100], 100)
        dt = time.time() - t0
        results.append({
            "method": f"uniform-{bits}bit",
            "avg_bits": float(bits),
            "recall_10": r10,
            "recall_100": r100,
            "time_s": dt,
        })
        print(f"  uniform-{bits}bit               avg_bits={bits:4.2f}  "
              f"R@10={r10:.3f}  R@100={r100:.3f}  ({dt:.1f}s)")

    # Hybrid
    t0 = time.time()
    h_scores, avg_bits = hybrid_scores(
        search_corpus, queries, high_bits, low_bits, k_pct
    )
    pred = top_k_from_scores(h_scores, max_k)
    r10 = recall_at_k(pred[:, :10], truth[:, :10], 10)
    r100 = recall_at_k(pred[:, :100], truth[:, :100], 100)
    dt = time.time() - t0
    results.append({
        "method": f"hybrid-{high_bits}/{low_bits}bit-top{int(k_pct*100)}%",
        "avg_bits": avg_bits,
        "recall_10": r10,
        "recall_100": r100,
        "time_s": dt,
    })
    print(f"  hybrid {high_bits}/{low_bits}bit  top{int(k_pct*100):>2d}%   "
          f"avg_bits={avg_bits:4.2f}  R@10={r10:.3f}  R@100={r100:.3f}  "
          f"({dt:.1f}s)")

    # Verdict — interpolate uniform curve at the hybrid's avg_bits and
    # compare. This is the "same bits/vector" comparison from the issue.
    uniform_pts = sorted(
        [r for r in results if r["method"].startswith("uniform")],
        key=lambda r: r["avg_bits"],
    )
    hybrid = results[-1]
    print()
    if len(uniform_pts) < 2:
        print("  → Not enough uniform baselines to interpolate.")
    else:
        # Find bracket containing hybrid's avg_bits
        lo = hi = None
        for i in range(len(uniform_pts) - 1):
            if uniform_pts[i]["avg_bits"] <= avg_bits <= uniform_pts[i + 1]["avg_bits"]:
                lo, hi = uniform_pts[i], uniform_pts[i + 1]
                break
        if lo is None:  # hybrid above or below the bracket — use nearest
            ref = min(uniform_pts, key=lambda r: abs(r["avg_bits"] - avg_bits))
            ref_r10 = ref["recall_10"]
            print(f"  Uniform curve does not bracket avg_bits={avg_bits:.2f}; "
                  f"using nearest ({ref['method']}) = {ref_r10:.3f}")
        else:
            t = (avg_bits - lo["avg_bits"]) / (hi["avg_bits"] - lo["avg_bits"])
            ref_r10 = lo["recall_10"] + t * (hi["recall_10"] - lo["recall_10"])
            print(f"  Uniform R@10 interpolated at {avg_bits:.2f} bits "
                  f"(between {lo['method']}={lo['recall_10']:.3f} and "
                  f"{hi['method']}={hi['recall_10']:.3f}) = {ref_r10:.3f}")

        gap = hybrid["recall_10"] - ref_r10
        print(f"    ΔR@10 (hybrid vs uniform-at-matched-bits) = {gap*100:+.1f} pts")
        if gap > 0.02:
            verdict = "SHIP"
            reason = f"hybrid beats uniform curve by {gap*100:.1f} pts at matched bits"
        elif gap > 0:
            verdict = "FILE"
            reason = f"hybrid ties uniform curve ({gap*100:+.1f} pts)"
        else:
            verdict = "FILE"
            reason = f"hybrid loses vs uniform curve ({gap*100:+.1f} pts)"
        print(f"    verdict: {verdict} — {reason}")

    return {"label": label, "k_pct": k_pct, "high_bits": high_bits,
            "low_bits": low_bits, "results": results}


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def synthetic_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Unit-norm Gaussian vectors. The ideal TurboQuant input distribution."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def load_specter2_cached(label: str) -> Optional[np.ndarray]:
    path = os.path.join(CACHE_DIR, f"{label}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specter2", action="store_true",
                        help="Run on cached SPECTER2 corpus (see specter2_eval.py)")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip real-data run; use synthetic Gaussian only")
    parser.add_argument("--n-synthetic", type=int, default=10_000)
    parser.add_argument("--d-synthetic", type=int, default=768)
    parser.add_argument("--low-bits", type=int, default=2,
                        help="Aggressive quantization tier (default 2)")
    parser.add_argument("--high-bits", type=int, default=4,
                        help="High-fidelity tier for the tail (default 4)")
    parser.add_argument("--k-pct", type=float, default=10.0,
                        help="Percent of vectors kept at high precision (default 10)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep k_pct in {5,10,20} and high_bits in {4,8} "
                             "for the matrix in the issue")
    parser.add_argument("--plots", action="store_true",
                        help="Save histogram + tail-mass plots")
    parser.add_argument("--step1-only", action="store_true",
                        help="Stop after residual characterization")
    parser.add_argument("--force-step2", action="store_true",
                        help="Run Step 2 even if gate fails")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    corpora = []
    if not args.specter2 or args.synthetic_only:
        corpora.append(("synthetic Gaussian d=768",
                        synthetic_gaussian(args.n_synthetic, args.d_synthetic, seed=0)))

    if args.specter2 and not args.synthetic_only:
        loaded = load_specter2_cached("specter2_nlp_broad")
        if loaded is None:
            print("SPECTER2 cache missing. Run:  python bench/specter2_eval.py --cached")
            sys.exit(1)
        corpora.append((f"SPECTER2 NLP-broad (n={loaded.shape[0]})", loaded))
        narrow = load_specter2_cached("specter2_transformer_narrow")
        if narrow is not None:
            corpora.append((f"SPECTER2 Transformer-narrow (n={narrow.shape[0]})",
                            narrow))

    all_step1 = []
    all_step2 = []

    for label, corpus in corpora:
        rel_err, _ = residual_errors(corpus, args.low_bits, seed=args.seed)
        step1 = print_step1(label, args.low_bits, corpus.shape[1], rel_err)
        all_step1.append(step1)
        if args.plots:
            path = save_residual_plot(label, rel_err, args.low_bits)
            if path:
                print(f"  plot: {path}")

        if args.step1_only:
            continue
        if not step1["gate_holds"] and not args.force_step2:
            print("  (gate failed — skipping Step 2; pass --force-step2 to run anyway)")
            continue

        if args.sweep:
            sweep_configs = [(kp, hb) for kp in (5, 10, 20) for hb in (4, 8)]
        else:
            sweep_configs = [(args.k_pct, args.high_bits)]
        for kp, hb in sweep_configs:
            step2 = run_step2(
                corpus, label,
                k_pct=kp / 100,
                high_bits=hb,
                low_bits=args.low_bits,
            )
            all_step2.append(step2)

    # Persist structured results
    out_path = os.path.join(RESULTS_DIR, f"results_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "step1": all_step1,
            "step2": all_step2,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
