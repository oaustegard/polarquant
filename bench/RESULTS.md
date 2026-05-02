# Benchmark Results

**Date**: 2026-04-05
**Library version**: remex 0.5.0
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

Standalone Mojo binary built from `remex/mojo/` (issue #5).

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
| twostage (ms/q)    |     17.3 |    19.0 | 0.91x   |

#### Scaling across `d` (1 trial each, indicative)

| `d`  | NumPy encode (µs) | Mojo encode (µs) | encode speedup | NumPy search (ms) | Mojo search (ms) | search speedup | NumPy twostage (ms) | Mojo twostage (ms) | twostage speedup |
|-----:|------------------:|-----------------:|---------------:|------------------:|-----------------:|---------------:|--------------------:|-------------------:|-----------------:|
|   64 |              3.65 |             1.72 |          2.12x |              2.99 |             0.70 |          4.30x |                3.24 |              16.59 |            0.20x |
|  256 |             13.21 |             8.24 |          1.60x |             11.63 |             1.85 |          6.28x |               12.38 |              17.60 |            0.70x |
|  384 |             ~17.0 |            ~13.3 |          1.27x |             16.40 |             2.71 |          6.01x |               17.26 |              19.00 |            0.91x |
|  768 |             44.52 |            39.65 |          1.12x |             36.28 |             5.36 |          6.77x |               36.69 |              23.58 |          **1.56x** |

### Notes

- **Encode** crossed from 1.3x slower (initial port, scalar matvec, `_dot_f32` only) to 1.27x faster after [#37](https://github.com/oaustegard/remex/pull/37) (SIMD vectorization) + [#41](https://github.com/oaustegard/remex/issues/41) (NB=8 row-blocking through `_dot_block_8`). The float64-accumulator norm change in [#40](https://github.com/oaustegard/remex/issues/40) (needed for byte parity vs Python's `np.linalg.norm`) did not measurably regress encode speed — norm is a tiny fraction of the encode hot path.
- **ADC search** wins consistently (4–7×) because Mojo's gather-then-scalar-add inner loop auto-vectorizes well; NumPy's equivalent is bottlenecked on per-chunk `np.outer` + `table[idx]` gathers.
- **Twostage** is the weak spot. Mojo's `search_twostage` time is roughly d-independent (~17–24 ms across all d), while NumPy scales with d. Mojo loses at small d (5× slower at d=64) and wins at large d (1.56× faster at d=768). The dominant cost is the O(n·candidates) selection-style coarse top-k loop in `search_twostage` — flagged as "naive" in `remex/mojo/README.md#known-gaps`. A min-heap or quickselect would drop that to O(n log k) ≈ 90× fewer ops at n=10k, candidates=500.
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
