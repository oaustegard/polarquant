# Mojo `polarquant` vs NumPy `remex` — Benchmark Results

**Last refreshed**: 2026-05-01 (twostage post-[#51](https://github.com/oaustegard/remex/pull/51))
**Library version**: remex 0.5.1 · `polarquant` Mojo binary
**Hardware**: container CPU (no GPU)
**Seeds**: corpus seed=42, query seed=99, quantizer seed=42
**Workload**: n=10000, bits=4, k=10

This file lives next to the comparison driver
([`compare.py`](compare.py)) and the per-stage bench binaries
(`bench_encode.mojo`, `bench_search.mojo`, `bench_twostage.mojo`).
For broader recall, distribution, and SPECTER2 results see
[`bench/RESULTS.md`](../../../bench/RESULTS.md) at the repo root.

## RNG parity (since [#40](https://github.com/oaustegard/remex/issues/40))

Mojo's default `--rng numpy` path uses the same stack as NumPy
(`SeedSequence + PCG64 + Ziggurat`) and produces a `.pq` **byte-identical**
to Python's `save_pq(Quantizer(d, bits, seed=S).encode(X))` at 1–4 bits.
Pass `--rng xoshiro` to opt back into the legacy xoshiro256++ + Marsaglia
path; `--params` remains the canonical all-bit-widths bridge between the
two implementations.

## Headline (d=384, queries=100, median of 5 trials)

| Stage              |    NumPy |    Mojo | Speedup |
|--------------------|---------:|--------:|--------:|
| encode (µs/vec)    |     17.0 |    13.3 | **1.27x** |
| ADC search (ms/q)  |     16.4 |     2.7 | **6.0x**  |
| twostage (ms/q)    |     17.6 |     3.2 | **5.5x**  |

## Scaling across `d`

twostage column is the post-[#51](https://github.com/oaustegard/remex/pull/51)
min-heap coarse top-k, 5-trial median. encode and search columns are 1-trial
indicative measurements (pre-#51, unaffected by it).

| `d`  | NumPy encode (µs) | Mojo encode (µs) | encode speedup | NumPy search (ms) | Mojo search (ms) | search speedup | NumPy twostage (ms) | Mojo twostage (ms) | twostage speedup |
|-----:|------------------:|-----------------:|---------------:|------------------:|-----------------:|---------------:|--------------------:|-------------------:|-----------------:|
|   64 |              3.65 |             1.72 |          2.12x |              2.99 |             0.70 |          4.30x |                3.42 |               0.51 |          **6.7x** |
|  256 |             13.21 |             8.24 |          1.60x |             11.63 |             1.85 |          6.28x |               11.40 |               2.01 |          **5.7x** |
|  384 |             ~17.0 |            ~13.3 |          1.27x |             16.40 |             2.71 |          6.01x |               17.59 |               3.18 |          **5.5x** |
|  768 |             44.52 |            39.65 |          1.12x |             36.28 |             5.36 |          6.77x |               37.60 |               6.21 |          **6.1x** |

## History

| When | PR | Stage | Change | d=384 result |
|------|----|----|--------|--------------|
| 2026-03 | [#37](https://github.com/oaustegard/remex/pull/37) | encode | SIMD vectorization | 1.3× slower → near parity |
| 2026-03 | [#41](https://github.com/oaustegard/remex/issues/41) | encode | NB=8 row-blocking via `_dot_block_8` | 1.27× faster vs NumPy |
| 2026-04 | [#40](https://github.com/oaustegard/remex/issues/40) | encode | float64-accumulator norm for byte parity | unchanged hot path |
| 2026-04 | [#48](https://github.com/oaustegard/remex/pull/48) | bench | Refresh; flagged twostage as weak spot | 19.0 ms (0.91×) |
| 2026-05 | [#51](https://github.com/oaustegard/remex/pull/51) | twostage | Min-heap coarse top-k (O(n·k) → O(n log k)) | 3.18 ms (5.5×) |

## GPU port (Apple M1 native, measured)

**Hardware**: Apple M1 native (no container), Mojo 1.0.0b1, Xcode Metal Toolchain.
Workload: n=10000, d=384, bits=4, seed=42, 1-trial timed after warmup.

| Stage | Mojo CPU | Mojo GPU (M1 Metal) | vs CPU |
|---|---|---|---|
| encode (µs/vec) | 11.7 | 43.1 | 3.7× slower |
| search (ms/q) | 5.74 | **4.42** | **1.30× faster** |

**Search wins**: PR [#66](https://github.com/oaustegard/remex/pull/66)'s
`GPUCorpus` stages the corpus (indices + norms, ~3.87 MB) to the device once
and reuses it across queries. Per-query H2D drops to just the lookup table
(~24 KB) — ~160× less transfer per query. Result: GPU search now beats CPU
search at d=384, where before it was 1.9× slower.

**Encode loses**: PR [#66](https://github.com/oaustegard/remex/pull/66) also
attempted to coalesce `R` reads across warp threads by pre-transposing R on
the host and reading `R_T[j*d + col]`.  On Apple Metal this made encode
*worse*: 43.1 µs/vec → 77.2 µs/vec (1.8× slowdown), reverted in PR
[#67](https://github.com/oaustegard/remex/pull/67).

**Default policy** (`--device auto`, the new default): encode runs on
CPU, search runs on GPU when an accelerator is reachable. CPU encode
beats GPU encode on every measured host today (Apple M1: 3.7× slower on
GPU); NVIDIA/AMD encode is unmeasured, so auto stays on CPU until a
baseline confirms otherwise. `--device cpu` and `--device gpu` are
explicit overrides for benchmarking or for re-measuring the policy.

The transpose trades per-thread sequential reads (`R[ci*d + j]` increments
by 1 in j) for per-thread strided reads (`R_T[j*d + ci]` strides by d).
On NVIDIA SIMT GPUs across-warp coalescing dominates; on Apple's TBDR
architecture the per-thread sequential pattern wins because the L1 +
hardware prefetcher already amortises the strided across-thread cost,
while strided per-thread reads thrash L1.  Lesson: validate GPU
optimisations against the actual silicon — don't port NVIDIA intuition to
Metal blind.

The `bench/RESULTS.md § Mojo port` GPU row will be populated when GPU
numbers are competitive with a CuPy/PyTorch baseline (issue #42 acceptance
criteria) — encode still needs work.  Avenues to investigate:

- One thread per row, full d-element dot in a single thread (minimises
  R-row reuse across blocks; works because n=10k is plenty of parallelism)
- `threadgroup` shared memory tiling — load BLOCK_X rows of R once per
  block, all threads in the block reuse them across j
- Vectorised per-thread loads (`SIMD[Float32, 4]`) inside the existing
  per-thread sequential pattern

## Notes

- **Encode** is bandwidth-limited at d=768 — the matvec hits memory bandwidth
  ceiling, so Mojo and NumPy's BLAS converge to ~1.12×.
- **ADC search** wins consistently (4–7×) because Mojo's gather-then-scalar-add
  inner loop auto-vectorizes well; NumPy's equivalent is bottlenecked on
  per-chunk `np.outer` + `table[idx]` gathers.
- **Twostage** was the weak spot before [#51](https://github.com/oaustegard/remex/pull/51).
  The O(n·candidates) selection-style coarse top-k made Mojo *slower* than
  NumPy at d ≤ 384 (0.20× at d=64), defeating the memory-savings value of
  `search_twostage` vs single-stage `search`. PR #51 replaced the selection
  loop with a min-heap walked once over `n` scores — ~90× fewer comparisons
  at the default (n=10k, candidates=500). Now consistently 5.5–6.7× faster
  than NumPy across all d.
- **Next bottleneck**: per-row gather+arithmetic in coarse-stage ADC scoring,
  tracked by [#50](https://github.com/oaustegard/remex/issues/50). Estimated
  ~1.5–2× further on the coarse pass.

## Build

```bash
cd remex/mojo

mojo build -I . polarquant.mojo            -o polarquant
mojo build -I . bench/bench_encode.mojo    -o bench/bench_encode
mojo build -I . bench/bench_search.mojo    -o bench/bench_search
mojo build -I . bench/bench_twostage.mojo  -o bench/bench_twostage
```

The container needs the Mojo compiler — see
[`../README.md`](../README.md#build) for installation.

## Reproduce

The headline numbers above:

```bash
cd remex/mojo
python bench/compare.py --n 10000 --d 384 --bits 4 --queries 100 --k 10
```

Scaling sweep (re-runs `compare.py` per `d`):

```bash
cd remex/mojo
for d in 64 256 384 768; do
  python bench/compare.py --n 10000 --d $d --bits 4 --queries 100 --k 10
done
```

Individual stages directly:

```bash
cd remex/mojo
./bench/bench_encode    --n 10000 --d 384 --bits 4 --seed 42
./bench/bench_search    --n 10000 --d 384 --bits 4 --queries 100 --k 10 --seed 42
./bench/bench_twostage  --n 10000 --d 384 --bits 4 --queries 100 --k 10 --seed 42 \
                        --candidates 500 --coarse-precision 2
```

## Parity verification

Twostage parity (re-run after any change to coarse top-k or rerank):

```bash
mojo run -I . tests/test_search_twostage.mojo
```

PR #51 result: 0 idx mismatches, scores match Python to 2.3e-7 (well within
`rtol=1e-5`).
