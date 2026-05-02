# remex Mojo port

A pure-Mojo port of the remex Quantizer's encode + ADC search +
decode path, shipped as a standalone CLI binary.

Closes [issue #5](https://github.com/oaustegard/remex/issues/5),
[issue #38](https://github.com/oaustegard/remex/issues/38), and
[issue #39](https://github.com/oaustegard/remex/issues/39).

## What's here

```
mojo/
├── README.md                # This file
├── remex.mojo               # CLI entrypoint (encode + search + decode)
├── src/
│   ├── mathx.mojo           # erf-based normal CDF / PDF
│   ├── rng.mojo             # xoshiro256++ + Marsaglia polar normal
│   ├── matrix.mojo          # Owning Float32 / Float64 matrices
│   ├── rotation.mojo        # Householder QR → Haar orthogonal matrix
│   ├── codebook.mojo        # Lloyd-Max iteration on N(0, 1/d) + Matryoshka nested tables
│   ├── packing.mojo         # 1/2/3/4/8-bit pack and unpack
│   ├── packed_vectors.mojo  # Bit-packed in-memory storage with on-demand unpack
│   ├── npy.mojo             # .npy reader/writer (float32, 2D, C-contiguous)
│   ├── pq_format.mojo       # .pq binary format read/write
│   ├── params_format.mojo   # .params dump (R + boundaries + centroids)
│   ├── quantizer.mojo       # Quantizer struct: encode + ADC search + two-stage + decode
│   └── gpu/                 # GPU/MAX kernels — scaffolding, see issue #42
│       ├── device.mojo      # is_gpu_available() probe
│       ├── encode.mojo      # gpu_encode_batch (stub)
│       └── adc.mojo         # gpu_adc_search (stub)
├── tests/
│   ├── test_rng.mojo
│   ├── test_rotation.mojo
│   ├── test_codebook.mojo
│   ├── test_packing.mojo
│   ├── test_packed_vectors.mojo    # PackedVectors round-trip + at_precision parity
│   ├── test_encode.mojo            # bit-identical encode parity vs Python
│   ├── test_decode.mojo            # decode parity vs Python (full + coarse precision)
│   ├── test_search_twostage.mojo   # top-k parity for search_twostage vs Python
│   ├── test_gpu_encode.mojo        # encode parity for --device gpu (skipped without GPU)
│   └── test_gpu_search.mojo        # ADC parity vs CPU for --device gpu (skipped without GPU)
└── bench/
    ├── RESULTS.md              # Mojo vs NumPy benchmark results
    ├── bench_encode.mojo
    ├── bench_search.mojo
    ├── bench_twostage.mojo
    ├── bench_gpu_encode.mojo       # GPU encode timing (skipped without GPU)
    ├── bench_gpu_search.mojo       # GPU ADC timing (skipped without GPU)
    └── compare.py           # Mojo vs NumPy comparison driver
```

## Build

The container needs the Mojo compiler (~500MB):

```bash
uv pip install --system --break-system-packages modular --no-deps
uv pip install --system --break-system-packages mojo max
```

Build the CLI and the bench binaries:

```bash
cd mojo
mojo build -I . remex.mojo                 -o remex
mojo build -I . bench/bench_encode.mojo    -o bench/bench_encode
mojo build -I . bench/bench_search.mojo    -o bench/bench_search
mojo build -I . bench/bench_twostage.mojo  -o bench/bench_twostage
```

## CLI usage

```bash
# Encode an .npy of float32 vectors → .pq
./remex encode corpus.npy --bits 4 --seed 42 -o corpus.pq

# Search a single (1, d) query against a .pq, top-k
./remex search corpus.pq query.npy --k 10 --seed 42 --top 10

# Memory-efficient two-stage retrieval: coarse ADC scan at reduced
# precision, full-precision rerank on the top `--candidates` rows.
./remex search corpus.pq query.npy --k 10 --seed 42 \
    --twostage --candidates 500 --coarse-precision 2

# Reconstruct float32 vectors from a .pq → .npy. Optional --precision
# uses the nested codebook at a coarser bit width (1..bits).
./remex decode corpus.pq --params P.bin -o reconstructed.npy
./remex decode corpus.pq --params P.bin --precision 2 -o coarse.npy
```

The same `corpus.pq` round-trips through the Python library:

```python
from remex import load_pq, save_pq, Quantizer
cv = load_pq("corpus.pq")            # works either way
print(cv.n, cv.d, cv.bits, cv.compression_ratio)
```

## Two parameter modes — `--seed` vs `--params`

The encode/search path needs (R, boundaries, centroids). The CLI
provides those two ways:

| Flag | Source of (R, codebook) | Bit-identical to a Python `Quantizer(seed=S)`? |
|---|---|---|
| `--seed S` *(default RNG: numpy)* | Computed in Mojo from S via PCG64 + SeedSequence + Ziggurat + Householder QR + Lloyd-Max | **Yes**, at 1–4 bits. **8-bit:** see caveat below. |
| `--seed S --rng xoshiro` | Computed in Mojo from S via xoshiro256++ + Marsaglia + Householder QR + Lloyd-Max | **No.** Both Haar samples but from different Gaussian streams. Faster init, no Ziggurat tables. |
| `--params P.bin` | Loaded from a file written by `remex.save_params(quantizer, P)` | **Yes**, at all bit widths. |

`--seed` (numpy mode, the default) gives a self-contained Mojo workflow
with end-to-end byte parity vs Python: `remex encode X.npy --bits 4
--seed 42 -o out.pq` produces the same bytes as Python's
`save_pq(Quantizer(d, 4, seed=42).encode(X))`. This was the goal of
issue #40 and uses the new `src/rng_numpy.mojo` module.

`--seed --rng xoshiro` is the legacy fast path. Use it when you don't
need Python parity — startup is slightly faster (no 25 KB of Ziggurat
tables to populate).

`--params` remains the canonical bridge for any case where you want
guaranteed bit parity regardless of bit width or potential drift.

### 8-bit byte-parity caveat

At `--bits 8`, the Lloyd-Max codebook uses 256 levels and runs 300
iterations of refinement. Mojo's `std.math.erf` differs from libm's
`math.erf` (used by `scipy.stats.norm.cdf`) by ~1e-8 per call. Across
300 iterations × 256 levels these accumulate, shifting a handful of
codebook boundaries by ~1e-3. Net effect: ~0.1–0.2% of indices end up
1 level off. The encoded `.pq` is ~99.8% byte-identical but not
strictly so. For 1/2/3/4 bit, the codebook has fewer levels and far
more margin, so byte parity holds exactly.

If you need byte parity at 8-bit, use `--params`. (This is a Mojo
stdlib precision limitation, not a remex algorithm issue.)

## Tests

```bash
cd mojo
mojo run -I . tests/test_rng.mojo
mojo run -I . tests/test_rng_numpy.mojo   # NumPy-bit-identical RNG (issue #40)
mojo run -I . tests/test_rotation.mojo
mojo run -I . tests/test_codebook.mojo
mojo run -I . tests/test_packing.mojo

# Encode parity via --params (requires Python remex installed and fixtures generated):
python -c "
import numpy as np
from remex import Quantizer, save_pq, save_params
np.random.seed(0)
X = np.random.randn(50, 16).astype(np.float32)
np.save('/tmp/_parity_X.npy', X)
q = Quantizer(d=16, bits=4, seed=42)
save_params('/tmp/_parity.params', q)
save_pq('/tmp/_parity_ref.pq', q.encode(X))
"
mojo run -I . tests/test_encode.mojo

# Decode parity (full precision + coarse via nested codebook):
python -c "
import numpy as np
from remex import Quantizer, save_pq, save_params

np.random.seed(0)
n, d, bits = 80, 16, 4
coarse_precision = 2

X = np.random.randn(n, d).astype(np.float32)
q = Quantizer(d=d, bits=bits, seed=42)
save_params('/tmp/_decode.params', q)
cv = q.encode(X)
save_pq('/tmp/_decode.pq', cv)

np.save('/tmp/_decode_X.npy', X)
np.save('/tmp/_decode_full.npy', q.decode(cv).astype(np.float32))
np.save('/tmp/_decode_coarse.npy',
        q.decode(cv, precision=coarse_precision).astype(np.float32))
np.save('/tmp/_decode_meta.npy',
        np.array([[coarse_precision]], dtype=np.float32))
"
mojo run -I . tests/test_decode.mojo

# PackedVectors round-trip + at_precision parity:
python -c "
import numpy as np
from remex import Quantizer, PackedVectors

np.random.seed(0)
n, d, bits = 80, 16, 4
target_bits = 2

X = np.random.randn(n, d).astype(np.float32)
q = Quantizer(d=d, bits=bits, seed=42)
cv = q.encode(X)
packed = PackedVectors.from_compressed(cv)
np.save('/tmp/_pv_indices.npy', cv.indices.astype(np.float32))
np.save('/tmp/_pv_indices_at.npy',
        packed.at_precision(target_bits).unpack_rows(0, n).astype(np.float32))
np.save('/tmp/_pv_meta.npy',
        np.array([[n, d, bits, target_bits]], dtype=np.float32))
"
mojo run -I . tests/test_packed_vectors.mojo

# Encode parity via --seed (NumPy-compatible RNG path, issue #40):
python -c "
import numpy as np
from remex import Quantizer, save_pq
np.random.seed(0)
X = np.random.randn(50, 16).astype(np.float32)
np.save('/tmp/_seed_parity_X.npy', X)
q = Quantizer(d=16, bits=4, seed=42)
save_pq('/tmp/_seed_parity_ref.pq', q.encode(X))
"
mojo run -I . tests/test_encode_seed.mojo

# search_twostage parity (also requires Python remex installed):
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

meta = np.array([[k, candidates, coarse_precision, n_q]], dtype=np.float32)
np.save('/tmp/_twostage_meta.npy', meta)
np.save('/tmp/_twostage_expected_idx.npy', expected_idx)
np.save('/tmp/_twostage_expected_scores.npy', expected_scores)
"
mojo run -I . tests/test_search_twostage.mojo
```

## Benchmarks

```bash
cd mojo
python bench/compare.py --n 10000 --d 384 --bits 4 --queries 100 --k 10
```

See [`bench/RESULTS.md`](bench/RESULTS.md) for current numbers.

## GPU / MAX path (`--device gpu`)

**Status: scaffolding only.** The CLI flag, dispatch, test harness, and
bench drivers are wired, but the kernels in `src/gpu/encode.mojo` and
`src/gpu/adc.mojo` are stubs that raise `Error` until the real
MAX-graph or kernel-launch implementation lands. Tracked by
[issue #42](https://github.com/oaustegard/remex/issues/42).

### Build

The GPU build needs MAX with a CUDA-capable backend. CPU-only hosts
(M-series Macs, generic Linux without an NVIDIA GPU) can still build
and run the binaries — the GPU paths just refuse at runtime via
`is_gpu_available()`, which lets the test/bench drivers skip cleanly.

```bash
# Same Mojo install as the CPU build.
uv pip install --system --break-system-packages modular --no-deps
uv pip install --system --break-system-packages mojo max

# Build the CLI + GPU bench binaries (same flags as CPU).
cd mojo
mojo build -I . remex.mojo                     -o remex
mojo build -I . bench/bench_gpu_encode.mojo    -o bench/bench_gpu_encode
mojo build -I . bench/bench_gpu_search.mojo    -o bench/bench_gpu_search
```

### Run

```bash
# Encode + search on GPU (errors with a clear message until kernels land).
./remex encode corpus.npy --bits 4 --params P.bin --device gpu -o corpus.pq
./remex search corpus.pq query.npy --k 10 --params P.bin --device gpu --top 10
```

### Tests

```bash
# Skipped on CPU-only hosts; runs real parity checks on a GPU host.
mojo run -I . tests/test_gpu_encode.mojo
mojo run -I . tests/test_gpu_search.mojo
```

`test_gpu_encode.mojo` reuses the `/tmp/_parity_*` fixtures already
generated for `test_encode.mojo` (see § Tests below). `test_gpu_search`
is self-contained: it builds a synthetic corpus and asserts the GPU
top-k matches the CPU `adc_search` baseline (rtol=1e-5 on scores,
identical indices).

### Acceptance for the kernel implementation

Per issue #42:

- `remex encode --device gpu` produces packed indices byte-identical
  to the CPU path on the same input + `(R, codebook)`, modulo a
  documented FP-order tolerance for boundary-adjacent coordinates.
- `bench/RESULTS.md § Mojo port` gains a row for GPU encode + search
  with timings benchmarked against a CuPy/PyTorch baseline on the same
  GPU host (not against the Mojo CPU bench).

### Why this is its own follow-up

GPU work needs an NVIDIA host to validate. The kernel implementation
doesn't gate on this scaffolding: anyone with a supported GPU can pick
up `src/gpu/encode.mojo` and `src/gpu/adc.mojo`, replace the `raise
Error(...)` body, and the rest of the pipeline (CLI, tests, bench)
already works.

## Notes / known gaps

- **Encode hot loop is SIMD-vectorized + NB=8 row-blocked.** The per-row
  rotation matvec and the squared-norm reduction in `encode_batch` use a
  `simd_width_of[DType.float32]()`-wide FMA + horizontal reduce — see
  `_dot_f32`, `_dot_block_8`, and `_sumsq_f64` in `src/quantizer.mojo`.
  On top of that, `encode_batch` processes 8 rows of X at a time through
  `_dot_block_8`, which fuses 8 dot products against the same R[k, :]
  into eight SIMD accumulators sharing one R-load per j-step. R memory
  traffic drops 8× — the unblocked path reloaded the full ~d²·4 bytes
  of R per row, which doesn't fit in L2 at d=384. The combined effect
  brings Mojo encode from 179 µs/vec (pre-SIMD) → 21 µs/vec (PR #37)
  → 12 µs/vec (this) at d=384, **1.26× faster than NumPy's BLAS
  `X @ R.T`** on AVX-512 (issue #41).
- **`search_twostage` coarse top-k is heap-based.** Min-heap over the
  coarse-stage scores brings the selection from O(n·candidates) to
  O(n log k) — ~90× fewer comparisons at default n=10k, candidates=500
  (PR #51, closing issue #49). The coarse-stage inner loop itself is
  still scalar gather+arithmetic; tighter SIMD on that gather is the
  next bottleneck — landed in PR #52 (NB=8 row-block of the coarse-stage
  ADC scoring loop). Numbers in `bench/RESULTS.md` are post-#51 but
  pre-#52; a fresh bench run will pick up the additional speedup.
- **GPU kernels are stubbed.** `--device gpu` is wired through the CLI,
  tests, and bench drivers, but `src/gpu/encode.mojo` and
  `src/gpu/adc.mojo` raise until the MAX implementation lands — see
  the `## GPU / MAX path` section above and issue #42.
- **Seed parity** with NumPy's `default_rng` is now bit-identical at
  1–4 bits via `src/rng_numpy.mojo` (PCG64 + SeedSequence + Ziggurat,
  issue #40). 8-bit byte parity is blocked by Mojo `std.math.erf`
  precision drift in the 256-level Lloyd-Max iteration — see the 8-bit
  caveat above; use `--params` for guaranteed parity at 8-bit.
- **UnsafePointer field aliasing.** Several spots copy struct-owned
  buffers into a fresh local allocation before passing the pointer to
  another function. Direct `struct.field[i]` reads through an
  `UnsafePointer` field can return stale/garbage values for the first
  one or two indices on Mojo 0.26.2 in this container. The copies are
  flagged in comments where they happen.
