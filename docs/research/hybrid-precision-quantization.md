# Investigation: mixed-precision quantization via residual-error tail selection

**Status:** complete &middot; **Issue:** [#34](https://github.com/oaustegard/remex/issues/34) &middot; **Verdict:** **FILE** &mdash; hypothesis dead at Step&nbsp;1 on both corpora; Step&nbsp;2 confirms no ship-worthy hybrid at matched bits/vector.

## TL;DR

The OjaKV hybrid-fidelity trick does not port to TurboQuant-style scalar
quantization. Two independent reasons, each sufficient on its own:

1. **The residual-error tail is flat.** On both synthetic Gaussian
   (the ideal TurboQuant input) and SPECTER2 embeddings, the top-10% of
   vectors by 2-bit residual carry only 10.8&ndash;11.0% of total
   squared-error mass &mdash; only ~1.08&times; what a uniform distribution
   would produce. There is no heavy tail to exploit.

2. **Cross-tier score merging is miscalibrated on real embeddings.**
   When Lloyd-Max is misspecified (SPECTER2: post-rotation &sigma; is 38%
   of the N(0, 1/d) assumption), 2-bit reconstruction produces a
   systematic downward bias in dot-product scores (&minus;10% absolute
   on SPECTER2) while 4-bit and 8-bit are nearly unbiased. Merging scores
   across precisions ranks *all* low-tier vectors below *all* high-tier
   vectors by raw score, regardless of semantic relevance &mdash; hybrid
   R@10 collapses to approximately the fraction of corpus promoted to the
   high tier (5% &rarr; R@10&asymp;0.059, 10% &rarr; R@10&asymp;0.100,
   20% &rarr; R@10&asymp;0.180).

On synthetic data where Lloyd-Max *is* correctly specified, merging works
but hybrid still loses to uniform interpolation by 2&ndash;21 R@10 points,
because the Matryoshka nesting penalty at 2-bit swamps the tiny advantage
from promoting a near-random 5&ndash;20% of vectors.

**Recommendation:** keep `search_twostage()` as the standard way to
trade memory for recall. It scores all vectors at low precision, then
reranks only the top candidates at full precision &mdash; structurally
avoids both failure modes above.

## Background

OjaKV (arXiv:[2509.21623](https://arxiv.org/abs/2509.21623), Zhu/Yang et al. 2025) reports that the dominant gain in their KV-cache scheme comes not from the online Oja updates they brand the paper around, but from a simple "hybrid storage" policy: keep tokens with high reconstruction error under the current basis at full fidelity, compress the rest. On RULER 0.6&times; their ablation shows this hybrid policy contributes **+18 points** while the Oja updates contribute only **+3**.

OjaKV's core technique is dimension reduction, not bit quantization, so most of the paper doesn't port to remex. But the hybrid-fidelity pattern is axis-agnostic: if a small fraction of vectors carry most of the quantization error, paying a little more memory to keep them at higher precision should trade cheaply for recall.

This investigation tests whether the pattern holds for TurboQuant-style scalar quantization as used in remex.

## Hypothesis

Under 2-bit remex (random orthogonal rotation + Lloyd-Max N(0, 1/d) codebook), the per-vector relative reconstruction error is **heavy-tailed**. Specifically, the top 10% of vectors by residual carry >40% of the total squared-error mass. If this holds, promoting those vectors to higher precision (4-bit or 8-bit) should yield a favorable recall/memory trade-off versus uniform quantization at matched bits/vector.

## Method

Implemented in `bench/hybrid_precision_eval.py`.

**Step 1** &mdash; residual-error distribution. For each vector `x[i]` in a corpus, quantize at 2-bit, decode, and compute `err[i] = ||x[i] - x_hat[i]||_2 / ||x[i]||_2`. Examine the distribution shape (skewness, kurtosis) and cumulative error mass in the top-p% of vectors (tail-mass / Lorenz curve).

**Gate.** If `mass_top_10pct` < 0.40, the tail is too flat for hybrid storage to help; stop.

**Step 2** &mdash; recall benchmark (conditional on gate). Build a hybrid index:

1. Compute per-vector 2-bit residual on the corpus.
2. Top `k_pct` of vectors by residual &rarr; quantize at `high_bits` (4 or 8).
3. Remaining vectors &rarr; quantize at 2-bit.
4. ADC-score both tiers with the *same* Haar rotation; merge into a single score vector.

Baselines: uniform 2-bit, uniform 3-bit, uniform 4-bit, uniform 8-bit. Recall@{10, 100} computed against exact inner-product ground truth on 200 held-out queries.

**Verdict** uses linear interpolation on the uniform curve at the hybrid's average bits/vector. Hybrid &ldquo;ships&rdquo; if it beats the interpolated uniform R@10 by >2 points at matched bits/vector.

## Corpora

| Corpus | n | d | Source |
|--------|---|---|--------|
| Synthetic Gaussian | 10 000 | 768 | Unit-normalized standard normal &mdash; the ideal TurboQuant input distribution. |
| SPECTER2 NLP-broad | 4 017 | 768 | Paper titles + abstracts from Semantic Scholar for &ldquo;natural language processing&rdquo;, encoded with `allenai/specter2_base`. |

The SPECTER2 corpus is a superset of the [existing SPECTER2 case study](../specter2-case-study.md), which already established that this corpus has &sigma; &asymp; 0.38 &times; the expected post-rotation Gaussian &mdash; an interesting test because Lloyd-Max is misspecified, so quantization error might behave differently than on the synthetic baseline.

## Step 1 results

### Synthetic Gaussian (d=768, n=10 000)

```
Relative L2 error  (||x - x̂|| / ||x||):
  min / median / mean / max : 0.3064 / 0.3421 / 0.3422 / 0.3827
  std                       : 0.0094
  p95 / p99                 : 0.3580 / 0.3657
  skewness / excess kurt.   : 0.147 / 0.142

Squared-error tail mass:
  top  1%  →  0.012   (1.16x uniform)
  top  5%  →  0.056   (1.12x uniform)
  top 10%  →  0.110   (1.10x uniform)
  top 20%  →  0.216   (1.08x uniform)
```

The residual distribution is **essentially uniform** over vectors. Top 10% carries 11% of total error mass, only 1.10&times; what a flat distribution would produce. Skewness (0.15) and excess kurtosis (0.14) are near-zero. This is the expected behavior under TurboQuant&rsquo;s design assumptions: a Haar rotation isotropizes the error, and Lloyd-Max bins are optimal for the matching N(0, 1/d) marginal, so no vector is much worse off than the mean.

**Gate on synthetic: FAILS.** The tail is flat.

### SPECTER2 NLP-broad (d=768, n=4 017)

```
Relative L2 error  (||x - x̂|| / ||x||):
  min / median / mean / max : 0.3071 / 0.3308 / 0.3309 / 0.3546
  std                       : 0.0070
  p95 / p99                 : 0.3426 / 0.3480
  skewness / excess kurt.   : 0.127 / 0.131

Squared-error tail mass:
  top  1%  →  0.011   (1.15x uniform)
  top  5%  →  0.055   (1.09x uniform)
  top 10%  →  0.108   (1.08x uniform)
  top 20%  →  0.212   (1.06x uniform)
  top 30%  →  0.315   (1.05x uniform)
  top 50%  →  0.517   (1.03x uniform)
```

Real scientific-paper embeddings produce a residual distribution almost
indistinguishable from synthetic Gaussian. The absolute error level is
slightly lower (mean 0.331 vs 0.342 &mdash; SPECTER2 sits at &sigma;
&asymp; 0.014 per coord after rotation vs 0.036 for the synthetic
baseline, so Lloyd-Max boundaries are too wide for SPECTER2 but the
relative error-spread across vectors is similar). Top 10% carries 10.8%
of total error mass, only 1.08&times; uniform &mdash; flatter than the
synthetic baseline, which is the opposite of what the OjaKV hypothesis
predicts.

**Gate on SPECTER2: FAILS.** Also flat.

## Gate decision

Both corpora fail the &ge;40% mass-in-top-10% gate by a wide margin.
The hypothesis that 2-bit TurboQuant produces a heavy-tailed
per-vector residual distribution is dead. Step 2 was run with
`--force-step2` only to characterize what happens when you ignore the
gate and build a hybrid index anyway &mdash; the numbers below show why
the gate exists.

## Step 2 results

Each config: 200 held-out queries, top-k vs exact inner-product ground
truth, remaining vectors as the search corpus.

### Synthetic Gaussian (n=9 800 after query split, d=768)

| Config | avg bits | R@10 | R@100 | uniform R@10 at matched bits | ΔR@10 |
|---|---:|---:|---:|---:|---:|
| uniform-2bit | 2.00 | 0.549 | 0.638 | &mdash; | &mdash; |
| uniform-3bit | 3.00 | 0.730 | 0.802 | &mdash; | &mdash; |
| uniform-4bit | 4.00 | 0.855 | 0.895 | &mdash; | &mdash; |
| uniform-8bit | 8.00 | 0.987 | 0.991 | &mdash; | &mdash; |
| hybrid 4/2, top 5% | 2.10 | 0.548 | 0.646 | 0.568 | **&minus;1.9** |
| hybrid 4/2, top 10% | 2.20 | 0.567 | 0.658 | 0.586 | **&minus;1.9** |
| hybrid 4/2, top 20% | 2.40 | 0.600 | 0.684 | 0.622 | **&minus;2.2** |
| hybrid 8/2, top 5% | 2.30 | 0.475 | 0.591 | 0.604 | **&minus;12.9** |
| hybrid 8/2, top 10% | 2.60 | 0.491 | 0.605 | 0.658 | **&minus;16.6** |
| hybrid 8/2, top 20% | 3.20 | 0.530 | 0.633 | 0.740 | **&minus;21.0** |

Hybrid 4/2 at 5% and 10% both lose 1.9 pts vs the uniform curve &mdash;
the 4&rarr;2 Matryoshka nesting penalty outweighs the tail-selection
advantage on a flat tail. Hybrid 8/2 is dramatically worse: Matryoshka
8&rarr;2 has a bigger nesting penalty than 4&rarr;2, so the 2-bit tier
on an 8-bit parent encoding is considerably less accurate than a
stand-alone 2-bit Lloyd-Max, and there is nothing in the residual
distribution that recovers that loss.

### SPECTER2 NLP-broad (n=3 817 after query split, d=768)

| Config | avg bits | R@10 | R@100 | uniform R@10 at matched bits | ΔR@10 |
|---|---:|---:|---:|---:|---:|
| uniform-2bit | 2.00 | 0.532 | 0.658 | &mdash; | &mdash; |
| uniform-3bit | 3.00 | 0.650 | 0.751 | &mdash; | &mdash; |
| uniform-4bit | 4.00 | 0.784 | 0.852 | &mdash; | &mdash; |
| uniform-8bit | 8.00 | 0.982 | 0.990 | &mdash; | &mdash; |
| hybrid 4/2, top 5% | 2.10 | **0.059** | 0.176 | 0.543 | **&minus;48.5** |
| hybrid 4/2, top 10% | 2.20 | **0.100** | 0.242 | 0.555 | **&minus;45.6** |
| hybrid 4/2, top 20% | 2.40 | **0.179** | 0.395 | 0.579 | **&minus;40.1** |
| hybrid 8/2, top 5% | 2.30 | **0.052** | 0.180 | 0.567 | **&minus;51.6** |
| hybrid 8/2, top 10% | 2.60 | **0.099** | 0.261 | 0.603 | **&minus;50.4** |
| hybrid 8/2, top 20% | 3.20 | **0.180** | 0.446 | 0.674 | **&minus;49.4** |

Hybrid R@10 closely tracks `k_pct`: top 5% &rarr; 5.9%, top 10% &rarr;
10.0%, top 20% &rarr; 17.9%. That is *exactly* the behaviour you get
when the tier merge is dominated by a per-tier score offset rather than
per-vector relevance &mdash; recall collapses to &ldquo;the fraction of
true top-k that happened to be promoted to the high tier&rdquo;, which
for residual-based promotion on a flat tail is just `k_pct`.

### Diagnosing the SPECTER2 collapse

Per-query score means on SPECTER2 (single query, k_pct=5%, n=3 817):

| | high tier (4-bit, n=191) | low tier (2-bit Matryoshka, n=3 626) |
|---|---:|---:|
| approximate mean | 393.0 | 364.4 |
| exact mean | 396.4 | 406.1 |
| **bias** | **&minus;3.4** | **&minus;41.7** |
| approximate range | [347.4, 434.1] | [310.4, 408.8] |
| exact range | [345.1, 438.0] | [348.4, 450.8] |

The low tier&rsquo;s approximate-score range maxes out at 408.8, below
the high tier&rsquo;s mean of 393.0, so the merge sorts virtually every
high-tier vector above virtually every low-tier vector regardless of
true rank. On synthetic Gaussian the equivalent per-tier bias is
&minus;0.0002 (high) / &minus;0.0000 (low) &mdash; effectively zero,
which is why that merge at least produces sane numbers (just not good
ones).

**Root cause.** SPECTER2 is anisotropic &mdash; the [SPECTER2 case
study](../specter2-case-study.md) documents that its post-rotation
&sigma; is 38% of the N(0, 1/d) assumption. Lloyd-Max boundaries
calibrated for the theoretical variance are too wide, so 2-bit
reconstruction systematically under-uses the outer levels and shrinks
dot-product magnitudes by ~10%. 4-bit and 8-bit are far less sensitive
(more levels &rarr; less clipping to the mismatched distribution), so
the bias only shows up at the low tier. This is not a bug in the
hybrid construction; it is a structural reason why the construction
cannot work on real anisotropic embeddings.

## Verdict

**FILE.** The OjaKV hybrid-storage policy does not port to TurboQuant
scalar quantization. Both gating criteria fail independently:

- Step 1 gate fails: the residual distribution is near-uniform across
  vectors on both synthetic and SPECTER2 corpora. No tail to exploit.
- Step 2 ship criterion fails: no config beats the uniform curve at
  matched bits/vector. On synthetic, hybrid 4/2 loses ~2 pts (Matryoshka
  penalty); on SPECTER2, hybrid loses 40&ndash;52 pts (score-scale bias).

## What to take away

1. **TurboQuant is doing its job.** The Haar rotation plus Lloyd-Max
   codebook is explicitly designed to isotropize the error so that no
   vector suffers disproportionately. The flat residual distribution is
   a success of that design, not a failure &mdash; and it is precisely
   what makes OjaKV-style &ldquo;promote the outliers&rdquo; heuristics
   useless here.

2. **OjaKV&rsquo;s hybrid trick is basis-specific.** It works because
   OjaKV compresses by projecting onto a low-rank basis, and some tokens
   are genuinely far outside the basis. With scalar quantization every
   coordinate gets its own (small) error budget; there is no
   low-rank outlier phenomenon.

3. **Don&rsquo;t merge scores across precisions on anisotropic data.**
   Even with Matryoshka-nested codebooks (which guarantee
   probability-consistent centroids across bit widths), the
   *reconstruction accuracy* depends on the codebook being well-matched
   to the distribution. When it isn&rsquo;t, the bias is bit-width-dependent
   and tier merges by raw score fail catastrophically. The existing
   [`Quantizer.search_twostage()`](../../remex/core.py#L635) pattern
   avoids this entirely by scoring everything at one precision for the
   coarse pass and reranking only candidates at full precision.

4. **Score-scale bias is worth its own investigation.** The finding
   that 2-bit SPECTER2 scores are biased &minus;10% absolute while
   4/8-bit are unbiased hints at a calibration opportunity: a
   distribution-aware Lloyd-Max (fitted to SPECTER2&rsquo;s &sigma;
   &asymp; 0.014 rather than the theoretical 0.036) might both improve
   recall directly and make cross-precision merges feasible. That is
   a separate line of work.

## Reproducing

```bash
pip install -e ".[dev,bench]" transformers torch matplotlib

# Fast path — synthetic only, always works
python bench/hybrid_precision_eval.py --synthetic-only --plots

# With SPECTER2 (first run downloads the model + fetches abstracts)
python bench/specter2_eval.py --cached --skip-recall   # primes corpus cache
python bench/hybrid_precision_eval.py --specter2 --plots

# Full sweep matching the issue spec
python bench/hybrid_precision_eval.py --specter2 --sweep --plots
```

Output artifacts:

- `bench/plots/*.png` &mdash; residual histograms and Lorenz-style tail-mass curves.
- `bench/hybrid_results/results_*.json` &mdash; structured Step 1 + Step 2 data for downstream analysis.
