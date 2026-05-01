# remex Mojo port (`polarquant`)

A pure-Mojo port of the remex Quantizer's encode + ADC search path,
shipped as a standalone CLI binary.

Closes [issue #5](https://github.com/oaustegard/remex/issues/5).

## What's here

```
remex/mojo/
├── README.md                # This file
├── polarquant.mojo          # CLI entrypoint (encode + search)
├── src/
│   ├── mathx.mojo           # erf-based normal CDF / PDF
│   ├── rng.mojo             # xoshiro256++ + Marsaglia polar normal
│   ├── matrix.mojo          # Owning Float32 / Float64 matrices
│   ├── rotation.mojo        # Householder QR → Haar orthogonal matrix
│   ├── codebook.mojo        # Lloyd-Max iteration on N(0, 1/d)
│   ├── packing.mojo         # 1/2/3/4/8-bit pack and unpack
│   ├── npy.mojo             # .npy reader (float32, 2D, C-contiguous)
│   ├── pq_format.mojo       # .pq binary format read/write
│   ├── params_format.mojo   # .params dump (R + boundaries + centroids)
│   └── quantizer.mojo       # Quantizer struct: encode + ADC search
├── tests/
│   ├── test_rng.mojo
│   ├── test_rotation.mojo
│   ├── test_codebook.mojo
│   ├── test_packing.mojo
│   └── test_encode.mojo     # bit-identical encode parity vs Python
└── bench/
    ├── bench_encode.mojo
    ├── bench_search.mojo
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
cd remex/mojo
mojo build -I . polarquant.mojo            -o polarquant
mojo build -I . bench/bench_encode.mojo    -o bench/bench_encode
mojo build -I . bench/bench_search.mojo    -o bench/bench_search
```

## CLI usage

```bash
# Encode an .npy of float32 vectors → .pq
./polarquant encode corpus.npy --bits 4 --seed 42 -o corpus.pq

# Search a single (1, d) query against a .pq, top-k
./polarquant search corpus.pq query.npy --k 10 --seed 42 --top 10
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
| `--seed S` | Computed in Mojo from S via xoshiro256++ + Householder QR + Lloyd-Max | **No.** Mojo uses xoshiro256++; Python uses NumPy's PCG64 + `default_rng`. The rotations are different but both are valid Haar samples. |
| `--params P.bin` | Loaded from a file written by `remex.save_params(quantizer, P)` | **Yes.** Mojo's encoded `.pq` is byte-identical to Python's. |

`--params` is what the parity test (`tests/test_encode.mojo`) uses. Use
it when you want a Mojo-encoded index that the Python library can search
into and recover the same neighbors as if Python had encoded the corpus
itself. Use `--seed` for a fully self-contained Mojo workflow.

## Tests

```bash
cd remex/mojo
mojo run -I . tests/test_rng.mojo
mojo run -I . tests/test_rotation.mojo
mojo run -I . tests/test_codebook.mojo
mojo run -I . tests/test_packing.mojo

# Encode parity (requires Python remex installed and fixtures generated):
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
```

## Benchmarks

```bash
cd remex/mojo
python bench/compare.py --n 10000 --d 384 --bits 4 --queries 100 --k 10
```

See `bench/RESULTS.md` (in this PR) for current numbers.

## Notes / known gaps

- **Encode hot loop is SIMD-vectorized.** The per-row rotation matvec
  and the squared-norm reduction in `encode_batch` (and the q_rot
  matvec in `adc_search`) use a `simd_width_of[DType.float32]()`-wide
  FMA + horizontal reduce — see `_dot_f32` and `_sumsq_f32` in
  `src/quantizer.mojo`. On AVX-512 this brings Mojo encode from 179
  µs/vec to 21 µs/vec at d=384 (8.6x speedup), within 1.3x of NumPy's
  BLAS `X @ R.T`. Closing the remaining gap would mean tiling /
  blocking the matvec or batching multiple rows per call.
- **No Matryoshka / no `search_twostage`.** Out of scope for issue #5.
- **No GPU.** The `coding-mojo` skill notes Claude.ai containers are
  CPU-only; GPU work needs to be tested on a different host.
- **Seed parity** with NumPy's `default_rng` is not bit-identical (see
  the table above). Implementing PCG64 + SeedSequence + Ziggurat in
  Mojo to match NumPy exactly is a separate, substantial effort. The
  `--params` file path is the practical bridge.
- **UnsafePointer field aliasing.** Several spots copy struct-owned
  buffers into a fresh local allocation before passing the pointer to
  another function. Direct `struct.field[i]` reads through an
  `UnsafePointer` field can return stale/garbage values for the first
  one or two indices on Mojo 0.26.2 in this container. The copies are
  flagged in comments where they happen.
