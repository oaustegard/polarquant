# Benchmark Results

**Date**: 2026-04-05 (recall, distribution, SPECTER2) · 2026-05-01 (Mojo twostage refresh, PR [#51](https://github.com/oaustegard/remex/pull/51))
**Library version**: remex 0.5.1
**Hardware**: CPU (NumPy), no GPU
**Seeds**: corpus seed=42, query seed=99, quantizer seed=42

## Synthetic Benchmarks (d=384, 20 clusters, spread=0.3)

### Recall vs bit level (10k corpus, 200 queries)

| Method | Compression | MSE | R@10 | R@100 | Encode (ms) | Search (ms) |
|--------|------------|-----|------|-------|-------------|-------------|
| remex 2-bit | 15.4x | 0.1171 | 0.538 | 0.634 | 79 | 21 |
| remex 3-bit | 10.4x | 0.0343 | 0.719 | 0.800 | 153 | 22 |
| remex 4-bit | 7.8x | 0.0094 | 0.850 | 0.895 | 134 | 21 |
| remex 8-bit | 4.0x | 0.0000 | 0.987 | 0.991 | 238 | 21 |

### Scaling with corpus size (4-bit)

| Corpus | R@10 | R@100 | Encode (ms) | Search (ms) |
|--------|------|-------|-------------|-------------|
| 1k | 0.880 | 0.930 | 12 | 4 |
| 5k | 0.862 | 0.905 | 63 | 13 |
| 10k | 0.850 | 0.895 | 134 | 21 |
| 50k | 0.839 | 0.872 | 689 | 140 |

### Two-stage search (4-bit Matryoshka, 200 candidates)

| Corpus | R@10 | Search (ms) |
|--------|------|-------------|
| 1k | 0.880 | 343 |
| 5k | 0.861 | 1535 |
| 10k | 0.849 | 2894 |
| 50k | 0.837 | 14156 |

Two-stage recall matches single-stage to within 0.5%, validating the Matryoshka coarse-to-fine approach.

### Distribution sensitivity (10k corpus, 200 queries, varying cluster tightness)

| Spread (σ) | 2-bit R@10 | 3-bit R@10 | 4-bit R@10 | 8-bit R@10 |
|-----------|-----------|-----------|-----------|-----------|
| 0.01 | 0.163 | 0.331 | 0.533 | 0.954 |
| 0.05 | 0.478 | 0.693 | 0.831 | 0.984 |
| 0.10 | 0.532 | 0.727 | 0.846 | 0.987 |
| 0.30 | 0.538 | 0.719 | 0.850 | 0.987 |
| 0.50 | 0.540 | 0.717 | 0.859 | 0.986 |
| 1.00 | 0.525 | 0.720 | 0.848 | 0.984 |

**Key finding**: Very tight clusters (σ=0.01) severely degrade recall at all bit levels below 8. At 4-bit, R@10 drops from 0.85 to 0.53 — a 37% loss. At 8-bit, the degradation is only 3%. This is because tight clusters create near-identical vectors where small quantization errors flip rankings.

### Post-rotation distribution analysis (10k vectors, d=384)

| Metric | Value |
|--------|-------|
| Expected σ (1/√d) | 0.051031 |
| Actual σ (global) | 0.051031 |
| Per-dim σ (mean±std) | 0.0510 ± 0.0004 |
| Kurtosis (Gaussian=3) | 2.98 ± 0.05 |
| σ range | [0.0495, 0.0525] |

The rotation produces near-perfect N(0, 1/d) coordinates on synthetic data.

## Real Embedding Benchmarks (all-MiniLM-L6-v2, d=384)

From `bench/real_embedding_eval.py` using 10k corpus and 500 queries encoded by sentence-transformers:

| Method | Compression | MSE | R@10 | R@100 |
|--------|------------|-----|------|-------|
| remex 2-bit | 16x | 0.1164 | 0.517 | 0.860 |
| remex 3-bit | 10.4x | 0.0341 | 0.599 | 0.897 |
| remex 4-bit | 7.8x | 0.0093 | 0.707 | 0.932 |
| remex 8-bit | 2.0x | 0.0000 | 0.974 | 0.995 |
| FAISS PQ (m=96, trained) | 16x | 0.0341 | 0.816 | 0.946 |
| FAISS PQ (m=48, trained) | 32x | 0.0636 | 0.618 | 0.877 |

**Real vs synthetic gap**: 4-bit R@10 drops from 0.85 (synthetic) to 0.707 (real). Real embeddings cluster by topic, amplifying quantization errors within tight clusters.

## SPECTER2 Scientific Embeddings (allenai/specter2_base, d=768)

From `bench/specter2_eval.py` using 1k papers per partition, 500 queries (random split). Papers fetched from Semantic Scholar API, encoded locally with SPECTER2.

### Post-rotation distribution analysis

| Metric | Broad (NLP) | Narrow (Transformer) | Expected |
|--------|------------|---------------------|----------|
| Per-coord σ mean | 0.0140 | 0.0138 | 0.0361 |
| σ ratio (actual/expected) | 0.389 | 0.381 | 1.000 |
| Excess kurtosis | 0.005 | -0.050 | 0.000 |
| KS rejections (α=0.05) | 20/20 | 20/20 | ~1/20 |

The Gaussian assumption does not hold: per-coordinate σ is only 38% of expected. However, both broad and narrow partitions show identical deviation — domain specificity does not matter.

### Recall

| Bits | Compression | Broad R@10 | Broad R@100 | Narrow R@10 | Narrow R@100 |
|------|------------|-----------|------------|------------|-------------|
| 2 | 15.7x | 0.630 | 0.783 | 0.688 | 0.797 |
| 3 | 10.5x | 0.714 | 0.838 | 0.767 | 0.864 |
| 4 | 7.9x | 0.834 | 0.910 | 0.859 | 0.923 |
| 8 | 4.0x | 0.989 | 0.994 | 0.988 | 0.993 |

Despite the 61% σ deviation, 4-bit R@10 > 0.83 and 8-bit is essentially lossless. SPECTER2 4-bit recall (0.834) is notably higher than MiniLM (0.707), likely due to higher dimensionality (768 vs 384) providing more quantization budget.

See [docs/specter2-case-study.md](../docs/specter2-case-study.md) for full analysis.

### 1-bit Matryoshka extraction at scale (10k corpus, 100 queries)

From `bench/onebit_experiment.py` against the published [SPECTER2 NLP-broad 10k cache](https://github.com/oaustegard/claude-container-layers/releases/tag/specter2-nlp-broad-10k) (run `bash bench/fetch_specter2_cache.sh` once, then `ONEBIT_N=10000 python3 bench/onebit_experiment.py`).

#### Standalone bit sweep

| Bits | Compression | R@10 | R@100 |
|------|------------|------|-------|
| 1 | 30.7× | **0.635** | 0.694 |
| 2 | 15.7× | 0.501 | 0.628 |
| 3 | 10.5× | 0.595 | 0.732 |
| 4 | 7.9× | 0.731 | 0.820 |
| 8 | 4.0× | 0.971 | 0.984 |

**1-bit dominates 2-bit and 3-bit on R@10**, and beats 2-bit on R@100. The "valley" extends across 1–3 bits, not just 2-bit. Lloyd-Max wins back at 4-bit. This is consistent with Charikar's SimHash result: sign-bit hashing is asymptotically optimal for cosine similarity preservation, while Lloyd-Max optimizes per-coordinate MSE — a different (and at low bits, worse) objective for inner-product retrieval. The 2-bit and 3-bit Lloyd-Max boundaries land inside the dense Gaussian lobe of post-rotation coordinates, creating systematic ranking errors that the 1-bit sign-only comparison avoids.

#### Matryoshka extraction (encode @ 8-bit, search @ precision)

| Precision | R@10 | R@100 | Δ@10 vs standalone |
|-----------|------|-------|--------------------|
| 1 | 0.635 | 0.694 | **+0.000** (bit-for-bit identical) |
| 2 | 0.381 | 0.531 | −0.120 (~12% nesting penalty) |
| 4 | 0.756 | 0.835 | +0.025 (slightly better than standalone) |

The 1-bit Matryoshka extraction has zero nesting penalty because the MSB of an n-bit Lloyd-Max code *is* the sign bit, which is exactly the standalone 1-bit code. The invariant is enforced in `tests/test_matryoshka.py::TestPrecisionOneBit::test_matryoshka_1bit_equals_standalone_1bit`.

#### Two-stage retrieval: 1-bit coarse → 8-bit rerank

| Candidates | % of corpus | R@10 | R@100 |
|------------|------------|------|-------|
| 100 | 1.0% | 0.966 | 0.694 |
| 150 | 1.5% | **0.971** | 0.824 |
| 200 | 2.0% | 0.972 | 0.888 |
| 300 | 3.0% | 0.971 | 0.946 |

**At 1.5% candidate budget, R@10 matches the full-8-bit ceiling (0.971).** The 1-bit coarse pass narrows the field aggressively but retains the top-10 winners.

#### Two-stage retrieval: 2-bit coarse → 8-bit rerank (for comparison)

| Candidates | R@10 | R@100 | Gap vs 1-bit coarse |
|------------|------|-------|--------------------|
| 100 | 0.851 | 0.531 | −11.5 pp R@10 |
| 150 | 0.910 | 0.649 | −6.1 pp R@10 |
| 200 | 0.937 | 0.724 | −3.5 pp R@10 |
| 300 | 0.948 | 0.820 | −2.3 pp R@10 |

**2-bit coarse is dominated by 1-bit coarse at every candidate budget**, while requiring 2× the in-memory footprint. There is no scenario in this experiment where 2-bit is the right architectural choice.

#### Architectural takeaway for very-large-scale two-stage retrieval

For the Semantic Scholar use case (~100M SPECTER2 vectors, two-stage with in-memory coarse + RDS-resident 8-bit rerank), the recommended configuration is:

- **In-memory coarse tier**: 1-bit Matryoshka extracted from the 8-bit codes (right-shift to MSB). At d=768 this packs to **9.6 GB for 100M vectors**. The coarse score is `popcount(query_signs XOR coarse_signs)` — no Lloyd-Max lookup needed.
- **Rerank tier**: 8-bit Lloyd-Max indices stored in Postgres, fetched by ID for the top-K candidates returned by the coarse pass. At K~1.5% of corpus, R@10 matches the full-8-bit ceiling.
- **What not to do**: 2-bit coarse. Half the recall, twice the RAM, no upside.

#### A note on σ-ratio measurement

The 10k experiment reports σ ratio = 0.9993 using `np.std(rotated_flattened)` over the rotated `(n, d)` matrix. The existing 1k-paper section above reports σ ratio = 0.389 using per-coordinate σ mean. These are different statistics — they coincide for zero-mean-per-coord Gaussian samples but diverge if SPECTER2's post-rotation marginals have non-zero per-coordinate means. Reconciling the two metrics on the same 10k corpus is queued as a follow-up; meanwhile, the recall results above are based on identical rotation + Lloyd-Max code paths as `bench/specter2_eval.py`, so distribution-analysis discrepancies don't affect them.

## Research investigations

- [docs/research/hybrid-precision-quantization.md](../docs/research/hybrid-precision-quantization.md) — Ported OjaKV&rsquo;s &ldquo;keep high-residual vectors at higher precision&rdquo; trick to TurboQuant-style scalar quantization. Verdict: FILE. Residual distribution is near-flat under Haar rotation + Lloyd-Max (top 10% of vectors carry only 10.8% of error mass), and cross-tier score merging is miscalibrated on anisotropic embeddings (e.g. SPECTER2). `search_twostage()` remains the right memory/recall trade-off.

## Memory Profiles (100k vectors, d=384, 8-bit)

| Strategy | Resident RAM | ms/query |
|----------|-------------|----------|
| CompressedVectors (no cache) | 38.8 MB | — |
| CompressedVectors (cached) | 192.4 MB | 3.9 |
| CompressedVectors (cold) | 38.8 MB | 137 |
| PackedVectors (always packed) | 38.8 MB | — |
| search_adc() (no cache) | 38.8 MB | 152 |
| search_twostage() (no cache) | 38.8 MB | 152 |

## Compression Ratios (bit-packed, d=384)

| Bits | Bytes per vector | vs float32 | File size per 10k vectors |
|------|-----------------|------------|--------------------------|
| 2 | 100 | 15.4x | 0.93 MB |
| 3 | 148 | 10.4x | 1.42 MB |
| 4 | 196 | 7.8x | 1.83 MB |
| 8 | 388 | 4.0x | 3.61 MB |

Float32 baseline: 1,536 bytes/vector, 15.36 MB per 10k vectors.

## Mojo port (`polarquant`) vs NumPy

Standalone Mojo binary built from `remex/mojo/` (issue #5). Full Mojo-vs-NumPy
results — including the build matrix, reproduce commands, and PR-by-PR history
— live in [`remex/mojo/bench/RESULTS.md`](../remex/mojo/bench/RESULTS.md).
Headline numbers below.

Since [#40](https://github.com/oaustegard/remex/issues/40), Mojo's
default `--seed S` path uses the same RNG stack as NumPy
(`SeedSequence + PCG64 + Ziggurat`) and produces a `.pq`
**byte-identical** to Python's `save_pq(Quantizer(d, bits, seed=S).encode(X))`
at 1–4 bits. (`--rng xoshiro` opts back into the legacy
xoshiro256++ + Marsaglia path; `--params` remains the canonical
all-bit-widths bridge.) See `remex/mojo/README.md#two-parameter-modes`.

### Wall-clock (n=10k, bits=4, queries=100, k=10, container CPU)

#### d=384 (median of 5 trials)

| Stage              |    NumPy |    Mojo | Speedup |
|--------------------|---------:|--------:|--------:|
| encode (µs/vec)    |     17.0 |    13.3 | **1.27x** |
| ADC search (ms/q)  |     16.4 |     2.7 | **6.0x**  |
| twostage (ms/q)    |     17.6 |     3.2 | **5.5x**  |

#### Scaling across `d`

twostage row is post-[#51](https://github.com/oaustegard/remex/pull/51)
(min-heap coarse top-k, 5-trial median); encode and search rows are 1-trial
indicative pre-#51 measurements.

| `d`  | NumPy encode (µs) | Mojo encode (µs) | encode speedup | NumPy search (ms) | Mojo search (ms) | search speedup | NumPy twostage (ms) | Mojo twostage (ms) | twostage speedup |
|-----:|------------------:|-----------------:|---------------:|------------------:|-----------------:|---------------:|--------------------:|-------------------:|-----------------:|
|   64 |              3.65 |             1.72 |          2.12x |              2.99 |             0.70 |          4.30x |                3.42 |               0.51 |          **6.7x** |
|  256 |             13.21 |             8.24 |          1.60x |             11.63 |             1.85 |          6.28x |               11.40 |               2.01 |          **5.7x** |
|  384 |             ~17.0 |            ~13.3 |          1.27x |             16.40 |             2.71 |          6.01x |               17.59 |               3.18 |          **5.5x** |
|  768 |             44.52 |            39.65 |          1.12x |             36.28 |             5.36 |          6.77x |               37.60 |               6.21 |          **6.1x** |

### Notes

- **Encode** crossed from 1.3x slower (initial port, scalar matvec, `_dot_f32` only) to 1.27x faster after [#37](https://github.com/oaustegard/remex/pull/37) (SIMD vectorization) + [#41](https://github.com/oaustegard/remex/issues/41) (NB=8 row-blocking through `_dot_block_8`). The float64-accumulator norm change in [#40](https://github.com/oaustegard/remex/issues/40) (needed for byte parity vs Python's `np.linalg.norm`) did not measurably regress encode speed — norm is a tiny fraction of the encode hot path.
- **ADC search** wins consistently (4–7×) because Mojo's gather-then-scalar-add inner loop auto-vectorizes well; NumPy's equivalent is bottlenecked on per-chunk `np.outer` + `table[idx]` gathers.
- **Twostage** was the weak spot before [#51](https://github.com/oaustegard/remex/pull/51): the O(n·candidates) selection-style coarse top-k loop made Mojo slower than NumPy at d ≤ 384 (0.20× at d=64). PR #51 replaced that with a min-heap walked once over `n` scores — O(n·k) → O(n log k), ~90× fewer comparisons at the default (n=10k, candidates=500). Mojo `search_twostage` is now consistently 5.5–6.7× faster than NumPy across all d, restoring it as the right default for memory-constrained workloads (single-stage `search`-cached speed at 4× less memory). The next bottleneck on the coarse pass is the per-row gather+arithmetic in ADC scoring, tracked by [#50](https://github.com/oaustegard/remex/issues/50) (~1.5–2× further estimated).
- **Encode advantage at d=768** shrinks to 1.12× — the matvec hits memory bandwidth limits, so Mojo and NumPy's BLAS converge.

### Reproduce

```bash
cd remex/mojo
mojo build -I . polarquant.mojo            -o polarquant
mojo build -I . bench/bench_encode.mojo    -o bench/bench_encode
mojo build -I . bench/bench_search.mojo    -o bench/bench_search
mojo build -I . bench/bench_twostage.mojo  -o bench/bench_twostage
python bench/compare.py --n 10000 --d 384 --bits 4 --queries 100 --k 10
```

## Reproducibility

All benchmarks can be reproduced with:

```bash
python bench/benchmark.py               # synthetic data (no extra deps)
pip install -e ".[bench]"               # for real embedding benchmarks
python bench/real_embedding_eval.py     # needs sentence-transformers + faiss-cpu
```

Seeds: corpus generation uses `np.random.default_rng(42)`, queries use seed 99, quantizer uses seed 42.
