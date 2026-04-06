"""
SPECTER2 distribution analysis and recall benchmark for remex.

Fetches paper titles+abstracts from the Semantic Scholar API, encodes
them with the SPECTER2 model (allenai/specter2_base, 768-d) locally,
then analyzes whether the post-rotation coordinate distribution matches
the N(0, 1/sqrt(d)) Gaussian assumption that Lloyd-Max relies on.

Two partitions are compared:
  - Broad: "natural language processing" (diverse NLP papers)
  - Narrow: "transformer attention mechanism" (tight subfield)

Key deliverable: where do real SPECTER2 partitions land on remex's
sensitivity curve, and does it change between broad vs narrow fields?

Usage:
    python bench/specter2_eval.py              # full run (~30 min for API + encoding)
    python bench/specter2_eval.py --cached     # skip API/encoding, use saved .npy files
    python bench/specter2_eval.py --plots      # also save distribution plots

Requirements:
    pip install transformers torch   # for SPECTER2 model
    # S2 API for paper metadata: no auth needed, 1 RPS limit
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from remex import Quantizer
from remex.rotation import haar_rotation

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".specter2_cache")
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
SPECTER2_DIM = 768
TARGET_N = 10_000


# ---------------------------------------------------------------------------
# Fetching paper texts from Semantic Scholar + encoding with SPECTER2
# ---------------------------------------------------------------------------


def _s2_api_get(url: str) -> dict:
    """Make a GET request to the S2 API with retries."""
    import urllib.request
    import urllib.error

    for attempt in range(4):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "remex-bench/0.1")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            wait = 2 ** (attempt + 1)
            print(f"    Retry {attempt+1}/4 after error: {e} (wait {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"S2 API request failed after 4 retries: {url}")


def fetch_paper_texts(query: str, target_n: int = TARGET_N) -> list:
    """Fetch paper title+abstract texts from the Semantic Scholar bulk API.

    Returns list of strings formatted as "title [SEP] abstract" for SPECTER2.
    """
    import urllib.parse

    texts = []
    token = None
    page = 0

    print(f"  Fetching paper texts for query='{query}'...")
    print(f"  Target: {target_n} papers with abstracts")

    while len(texts) < target_n:
        params = {
            "query": query,
            "fields": "title,abstract",
        }
        if token is not None:
            params["token"] = token

        url = S2_BASE + "?" + urllib.parse.urlencode(params)
        page += 1

        try:
            data = _s2_api_get(url)
        except RuntimeError as e:
            print(f"    {e}, stopping at {len(texts)} texts")
            break

        papers = data.get("data", [])
        if not papers:
            print(f"    No more papers returned, got {len(texts)} total")
            break

        for paper in papers:
            title = paper.get("title") or ""
            abstract = paper.get("abstract") or ""
            if title and abstract:
                # SPECTER2 format: title [SEP] abstract
                texts.append(f"{title} [SEP] {abstract}")
                if len(texts) >= target_n:
                    break

        token = data.get("token")
        n_so_far = len(texts)
        print(f"    Page {page}: {len(papers)} papers, {n_so_far}/{target_n} with abstracts")

        if token is None:
            print(f"    No continuation token, stopping at {n_so_far}")
            break

        # Rate limit: 1 RPS
        time.sleep(1.0)

    print(f"  Collected {len(texts)} paper texts")
    return texts[:target_n]


def encode_with_specter2(
    texts: list, batch_size: int = 32, checkpoint_path: str = None
) -> np.ndarray:
    """Encode texts with the SPECTER2 model (allenai/specter2_base).

    Supports incremental checkpointing: saves progress every 256 texts
    so encoding can be resumed if interrupted.

    Returns (n, 768) float32 embeddings.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    # Resume from checkpoint if available
    start_idx = 0
    all_embeddings = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        partial = np.load(checkpoint_path)
        start_idx = partial.shape[0]
        all_embeddings.append(partial)
        print(f"  Resuming from checkpoint: {start_idx}/{len(texts)} already encoded")
        if start_idx >= len(texts):
            return partial[:len(texts)]

    model_name = "allenai/specter2_base"
    print(f"  Loading SPECTER2 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}, encoding {len(texts) - start_idx} remaining texts...")

    checkpoint_interval = 256  # save every 256 texts
    t0 = time.time()
    for i in range(start_idx, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(emb)

        done = i + len(batch)
        if done % checkpoint_interval < batch_size or done >= len(texts):
            elapsed = time.time() - t0
            rate = (done - start_idx) / max(elapsed, 0.01)
            remaining = (len(texts) - done) / max(rate, 0.01)
            print(f"    {done}/{len(texts)} encoded ({elapsed:.0f}s, "
                  f"~{remaining:.0f}s remaining)")
            # Save checkpoint
            if checkpoint_path:
                concat = np.concatenate(all_embeddings, axis=0).astype(np.float32)
                np.save(checkpoint_path, concat)

    result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"  Encoded {result.shape[0]} vectors of dimension {result.shape[1]} "
          f"in {time.time()-t0:.1f}s")
    return result


def load_or_fetch(
    query: str, label: str, target_n: int = TARGET_N, use_cache: bool = True
) -> np.ndarray:
    """Load cached embeddings or fetch texts from S2 + encode with SPECTER2."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{label}.npy")
    texts_path = os.path.join(CACHE_DIR, f"{label}_texts.json")
    checkpoint_path = os.path.join(CACHE_DIR, f"{label}_partial.npy")

    if use_cache and os.path.exists(cache_path):
        emb = np.load(cache_path)
        print(f"  Loading cached embeddings from {cache_path} ({emb.shape[0]} vectors)")
        return emb[:target_n]

    # Fetch or load cached texts
    if os.path.exists(texts_path):
        print(f"  Loading cached texts from {texts_path}")
        with open(texts_path) as f:
            texts = json.load(f)
    else:
        texts = fetch_paper_texts(query, target_n)
        if len(texts) == 0:
            raise RuntimeError(f"No paper texts fetched for query '{query}'")
        with open(texts_path, "w") as f:
            json.dump(texts, f)
        print(f"  Saved {len(texts)} texts to {texts_path}")

    texts = texts[:target_n]
    embeddings = encode_with_specter2(texts, checkpoint_path=checkpoint_path)
    np.save(cache_path, embeddings)
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"  Saved final embeddings to {cache_path}")
    return embeddings


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------


def analyze_rotation_distribution(
    corpus: np.ndarray, d: int, label: str, seed: int = 42, save_plots: bool = False
) -> dict:
    """Analyze post-rotation coordinate distribution vs N(0, 1/sqrt(d)).

    Steps:
      1. Normalize + rotate corpus with remex's Haar rotation
      2. Compute per-coordinate sigma across the rotated corpus
      3. Compare empirical marginals against N(0, 1/sqrt(d)) via KS test
      4. Map sigma values onto remex's sensitivity table

    Returns dict with analysis results.
    """
    R = haar_rotation(d, seed)
    norms = np.linalg.norm(corpus, axis=1)
    unit = corpus / np.maximum(norms, 1e-8)[:, None]
    rotated = unit @ R.T  # (n, d) — what Lloyd-Max actually sees

    sigma_expected = 1.0 / np.sqrt(d)

    # Per-coordinate statistics
    sigma_per_coord = np.std(rotated, axis=0)  # (d,) — one sigma per dimension
    mean_per_coord = np.mean(rotated, axis=0)
    kurtosis_per_coord = stats.kurtosis(rotated, axis=0, fisher=True)  # excess kurtosis, Gaussian=0

    # Global statistics
    sigma_global = float(np.std(rotated))
    sigma_ratio = float(np.mean(sigma_per_coord)) / sigma_expected

    # KS test: pick a few random coordinates and test against N(0, sigma_expected)
    rng = np.random.default_rng(42)
    test_coords = rng.choice(d, size=min(20, d), replace=False)
    ks_results = []
    for coord_idx in test_coords:
        coord_vals = rotated[:, coord_idx]
        stat, pval = stats.kstest(coord_vals, "norm", args=(0, sigma_expected))
        ks_results.append({"coord": int(coord_idx), "statistic": stat, "pvalue": pval})

    ks_stats = [r["statistic"] for r in ks_results]
    ks_pvals = [r["pvalue"] for r in ks_results]

    # Summary
    print(f"\n{'='*60}")
    print(f"  Distribution Analysis: {label}")
    print(f"  corpus: {corpus.shape[0]} vectors, d={d}")
    print(f"{'='*60}")
    print(f"  Expected σ (1/√d):        {sigma_expected:.6f}")
    print(f"  Actual σ (global):         {sigma_global:.6f}")
    print(f"  Per-coord σ (mean±std):    {np.mean(sigma_per_coord):.6f} ± {np.std(sigma_per_coord):.6f}")
    print(f"  Per-coord σ range:         [{np.min(sigma_per_coord):.6f}, {np.max(sigma_per_coord):.6f}]")
    print(f"  σ ratio (actual/expected): {sigma_ratio:.4f}")
    print(f"  Per-coord mean (abs avg):  {np.mean(np.abs(mean_per_coord)):.6f}")
    print(f"  Excess kurtosis (avg):     {np.mean(kurtosis_per_coord):.4f} (Gaussian=0)")
    print(f"  Kurtosis std:              {np.std(kurtosis_per_coord):.4f}")
    print(f"  Norm mean±std:             {np.mean(norms):.4f} ± {np.std(norms):.4f}")
    print()
    print(f"  KS test vs N(0, 1/√d) — {len(test_coords)} random coordinates:")
    print(f"    KS statistic (mean):     {np.mean(ks_stats):.4f}")
    print(f"    KS statistic (max):      {np.max(ks_stats):.4f}")
    print(f"    p-value (median):        {np.median(ks_pvals):.4f}")
    n_reject = sum(1 for p in ks_pvals if p < 0.05)
    print(f"    Rejected at α=0.05:      {n_reject}/{len(test_coords)}")

    # Map onto sensitivity table
    print()
    print("  Mapping to remex sensitivity table:")
    # The sensitivity table uses cluster spread σ; here we compare
    # how per-coord σ deviation from expected affects recall.
    # If σ_actual ≈ σ_expected, we're in the "typical" row (σ=0.30).
    # If σ_actual << σ_expected, coordinates are more concentrated than
    # expected → Lloyd-Max boundaries are too wide → worse quantization.
    sigma_median = float(np.median(sigma_per_coord))
    sigma_p5 = float(np.percentile(sigma_per_coord, 5))
    sigma_p95 = float(np.percentile(sigma_per_coord, 95))
    print(f"    Median per-coord σ:      {sigma_median:.6f} (expected {sigma_expected:.6f})")
    print(f"    5th percentile σ:        {sigma_p5:.6f} ({sigma_p5/sigma_expected:.3f}x expected)")
    print(f"    95th percentile σ:       {sigma_p95:.6f} ({sigma_p95/sigma_expected:.3f}x expected)")
    deviation_pct = abs(sigma_ratio - 1.0) * 100
    print(f"    Mean deviation from N(0,1/√d): {deviation_pct:.1f}%")
    if deviation_pct < 5:
        print("    → Gaussian assumption holds well. Expect near-optimal Lloyd-Max performance.")
    elif deviation_pct < 15:
        print("    → Moderate deviation. May see 1-3% recall loss vs optimal at 4-bit.")
    else:
        print("    → Significant deviation. Consider 8-bit or data-adaptive quantization.")

    if save_plots:
        _save_distribution_plots(
            sigma_per_coord, rotated, sigma_expected, test_coords, label
        )

    return {
        "label": label,
        "n": corpus.shape[0],
        "d": d,
        "sigma_expected": sigma_expected,
        "sigma_global": sigma_global,
        "sigma_per_coord_mean": float(np.mean(sigma_per_coord)),
        "sigma_per_coord_std": float(np.std(sigma_per_coord)),
        "sigma_per_coord_min": float(np.min(sigma_per_coord)),
        "sigma_per_coord_max": float(np.max(sigma_per_coord)),
        "sigma_median": sigma_median,
        "sigma_p5": sigma_p5,
        "sigma_p95": sigma_p95,
        "sigma_ratio": sigma_ratio,
        "kurtosis_mean": float(np.mean(kurtosis_per_coord)),
        "kurtosis_std": float(np.std(kurtosis_per_coord)),
        "ks_stat_mean": float(np.mean(ks_stats)),
        "ks_stat_max": float(np.max(ks_stats)),
        "ks_pval_median": float(np.median(ks_pvals)),
        "ks_n_reject_005": n_reject,
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
    }


def _save_distribution_plots(
    sigma_per_coord: np.ndarray,
    rotated: np.ndarray,
    sigma_expected: float,
    test_coords: np.ndarray,
    label: str,
):
    """Save histogram and QQ plots to bench/plots/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed, skipping plots)")
        return

    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    safe_label = label.lower().replace(" ", "_").replace("/", "_")

    # 1. Histogram of per-coordinate sigma values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sigma_per_coord, bins=50, density=True, alpha=0.7, label="Per-coord σ")
    ax.axvline(sigma_expected, color="red", linestyle="--", linewidth=2,
               label=f"Expected σ = 1/√{rotated.shape[1]} = {sigma_expected:.4f}")
    ax.axvline(np.mean(sigma_per_coord), color="green", linestyle="--", linewidth=1.5,
               label=f"Mean σ = {np.mean(sigma_per_coord):.4f}")
    ax.set_xlabel("Per-coordinate σ")
    ax.set_ylabel("Density")
    ax.set_title(f"Per-coordinate σ distribution after rotation — {label}")
    ax.legend()
    path = os.path.join(plot_dir, f"{safe_label}_sigma_hist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # 2. QQ plot: a few random coordinates vs N(0, 1/sqrt(d))
    n_qq = min(4, len(test_coords))
    fig, axes = plt.subplots(1, n_qq, figsize=(4 * n_qq, 4))
    if n_qq == 1:
        axes = [axes]
    for i, coord_idx in enumerate(test_coords[:n_qq]):
        ax = axes[i]
        coord_vals = rotated[:, coord_idx]
        stats.probplot(coord_vals / sigma_expected, dist="norm", plot=ax)
        ax.set_title(f"Coord {coord_idx}")
        ax.get_lines()[0].set_markersize(2)
    fig.suptitle(f"QQ plots vs N(0, 1/√d) — {label}", y=1.02)
    path = os.path.join(plot_dir, f"{safe_label}_qq.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Recall benchmark
# ---------------------------------------------------------------------------


def exact_knn(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Brute-force exact inner product search."""
    scores = queries @ corpus.T
    return np.argsort(-scores, axis=1)[:, :k]


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of true top-k found in predicted top-k."""
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k]) & set(t[:k]))
    return hits / (len(pred) * k)


def benchmark_recall(
    corpus: np.ndarray, d: int, label: str,
    n_queries: int = 500, seed: int = 99,
) -> list:
    """Split corpus into search set + queries, benchmark remex recall."""
    rng = np.random.default_rng(seed)
    n = corpus.shape[0]
    if n <= n_queries:
        n_queries = min(200, n // 5)

    # Random split
    perm = rng.permutation(n)
    query_idx = perm[:n_queries]
    corpus_idx = perm[n_queries:]
    queries = corpus[query_idx]
    search_corpus = corpus[corpus_idx]

    print(f"\n--- Recall Benchmark: {label} ---")
    print(f"  Corpus: {search_corpus.shape[0]}, Queries: {queries.shape[0]}, d={d}")

    # Ground truth
    max_k = 100
    truth = exact_knn(search_corpus, queries, max_k)

    results = []
    bits_list = [2, 3, 4, 8]
    k_list = [10, 100]

    header = f"  {'Method':<20s} {'Comp':>6s} {'R@10':>7s} {'R@100':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for bits in bits_list:
        pq = Quantizer(d=d, bits=bits)
        compressed = pq.encode(search_corpus)
        ratio = compressed.compression_ratio

        pred, _ = pq.search_batch(compressed, queries, k=max_k)

        r10 = recall_at_k(pred[:, :10], truth[:, :10], 10)
        r100 = recall_at_k(pred[:, :100], truth[:, :100], 100)

        print(f"  remex-{bits}bit          {ratio:>5.1f}x {r10:>7.3f} {r100:>7.3f}")
        results.append({
            "method": f"remex-{bits}bit",
            "bits": bits,
            "compression_ratio": ratio,
            "recall_10": r10,
            "recall_100": r100,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compare_partitions(dist_broad: dict, dist_narrow: dict):
    """Compare broad vs narrow partition distribution results."""
    print(f"\n{'='*60}")
    print("  Comparison: Broad vs Narrow Partition")
    print(f"{'='*60}")
    print(f"  {'Metric':<30s} {'Broad':>12s} {'Narrow':>12s}")
    print("  " + "-" * 56)

    rows = [
        ("σ ratio (actual/expected)", "sigma_ratio", ".4f"),
        ("Per-coord σ mean", "sigma_per_coord_mean", ".6f"),
        ("Per-coord σ std", "sigma_per_coord_std", ".6f"),
        ("Per-coord σ 5th pct", "sigma_p5", ".6f"),
        ("Per-coord σ 95th pct", "sigma_p95", ".6f"),
        ("Excess kurtosis (mean)", "kurtosis_mean", ".4f"),
        ("KS stat (mean)", "ks_stat_mean", ".4f"),
        ("KS reject @α=0.05", "ks_n_reject_005", "d"),
        ("Norm mean", "norm_mean", ".4f"),
        ("Norm std", "norm_std", ".4f"),
    ]
    for name, key, fmt in rows:
        b = dist_broad[key]
        n = dist_narrow[key]
        print(f"  {name:<30s} {b:>12{fmt}} {n:>12{fmt}}")

    # Interpretation
    print()
    broad_dev = abs(dist_broad["sigma_ratio"] - 1.0) * 100
    narrow_dev = abs(dist_narrow["sigma_ratio"] - 1.0) * 100
    print(f"  Broad σ deviation:  {broad_dev:.1f}%")
    print(f"  Narrow σ deviation: {narrow_dev:.1f}%")

    if abs(broad_dev - narrow_dev) < 2:
        print("  → Both partitions show similar deviation from Gaussian.")
        print("    Domain specificity does not meaningfully affect the distribution.")
    elif narrow_dev > broad_dev + 5:
        print("  → Narrow partition has notably higher deviation.")
        print("    Domain-specific clustering pushes σ away from the Gaussian assumption.")
        print("    Consider 8-bit for very narrow subfields.")
    else:
        print("  → Narrow partition has comparable or lower deviation.")
        print("    The Gaussian assumption is robust across domain breadth.")


def main():
    parser = argparse.ArgumentParser(description="SPECTER2 distribution analysis for remex")
    parser.add_argument("--cached", action="store_true",
                        help="Use cached .npy files instead of fetching from API")
    parser.add_argument("--plots", action="store_true",
                        help="Save distribution plots (requires matplotlib)")
    parser.add_argument("--skip-recall", action="store_true",
                        help="Skip recall benchmark (distribution analysis only)")
    parser.add_argument("-n", "--num-vectors", type=int, default=TARGET_N,
                        help=f"Number of vectors per partition (default: {TARGET_N})")
    args = parser.parse_args()

    d = SPECTER2_DIM
    target_n = args.num_vectors

    # --- Fetch / load embeddings ---
    print("=" * 60)
    print("  SPECTER2 Distribution Analysis for remex")
    print(f"  Target: {target_n} vectors per partition")
    print("=" * 60)

    corpus_broad = load_or_fetch(
        "natural language processing",
        "specter2_nlp_broad",
        target_n=target_n,
        use_cache=args.cached,
    )
    corpus_narrow = load_or_fetch(
        "transformer attention mechanism",
        "specter2_transformer_narrow",
        target_n=target_n,
        use_cache=args.cached,
    )

    # --- Distribution analysis ---
    dist_broad = analyze_rotation_distribution(
        corpus_broad, d, "SPECTER2 — Broad (NLP)", save_plots=args.plots
    )
    dist_narrow = analyze_rotation_distribution(
        corpus_narrow, d, "SPECTER2 — Narrow (Transformer Attention)", save_plots=args.plots
    )

    # --- Comparison ---
    compare_partitions(dist_broad, dist_narrow)

    # --- Recall benchmark ---
    if not args.skip_recall:
        recall_broad = benchmark_recall(corpus_broad, d, "SPECTER2 — Broad (NLP)")
        recall_narrow = benchmark_recall(corpus_narrow, d, "SPECTER2 — Narrow (Transformer Attention)")

        print(f"\n{'='*60}")
        print("  Recall Comparison: Broad vs Narrow")
        print(f"{'='*60}")
        print(f"  {'Method':<20s} {'Broad R@10':>12s} {'Narrow R@10':>12s} {'Gap':>8s}")
        print("  " + "-" * 54)
        for rb, rn in zip(recall_broad, recall_narrow):
            gap = rb["recall_10"] - rn["recall_10"]
            print(f"  {rb['method']:<20s} {rb['recall_10']:>11.3f} {rn['recall_10']:>11.3f} {gap:>+7.3f}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
