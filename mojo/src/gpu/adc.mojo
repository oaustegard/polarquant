"""GPU ADC search kernel for `remex search --device gpu`.

Stub for issue #42. The CPU contract
(`src/quantizer.mojo::adc_search`) is the reference; this kernel must
return the same top-k indices and scores (rtol=1e-5) for the same query,
codes, norms, and `(R, codebook)`.

## Intended kernel shape

`adc_search` is three passes:

1. `q_rot = R @ query` — a single (d, d) × (d,) matvec. Trivially fused
   with the table build below.
2. Build `table[j, c] = q_rot[j] * centroids[c]` — a `(d, n_levels)`
   outer product.  At 4-bit and d=384 this is 384 × 16 × 4 = 24 KB
   — fits comfortably in shared / constant memory.
3. `score[i] = sum_j table[j, indices[i, j]] * norms[i]` — the gather +
   reduction that dominates runtime. Each row does `d` shared-memory
   loads keyed by a uint8 index.

Followed by top-k. For typical `k <= 100`, a per-block bitonic top-k or
a pair of `argpartition` + sort passes is fine. For large `k` use
`MAX ops.top_k` if it's available.

## Memory layout

- `indices` is `(n, d)` row-major uint8. Coalesced loads if threads in a
  warp index different rows at the same `j` — i.e. score in column-major
  blocks (transpose the access pattern, not the layout).
- `centroids` and `boundaries` are constant per query; copy once per
  search and reuse.
- For repeated queries against the same corpus, the device-side
  `indices` and `norms` buffers should outlive a single `gpu_adc_search`
  call. The current stub stages H2D each call; the real implementation
  should expose a pre-built corpus handle. Out of scope for the initial
  kernel — see follow-up notes in issue #42.
"""

from std.memory import UnsafePointer
from src.quantizer import Quantizer


def gpu_adc_search(q: Quantizer,
                   indices: UnsafePointer[UInt8, MutExternalOrigin],
                   norms: UnsafePointer[Float32, MutExternalOrigin],
                   n: Int,
                   query: UnsafePointer[Float32, MutExternalOrigin],
                   k: Int,
                   mut top_idx: UnsafePointer[Int, MutExternalOrigin],
                   mut top_scores: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """GPU mirror of `adc_search`. See `src/quantizer.mojo::adc_search`.

    Same input/output contract as the CPU path; host buffers are
    H2D-staged internally per call.

    Stub: raises until the MAX kernel lands. See issue #42.
    """
    raise Error(
        "gpu_adc_search: not yet implemented — "
        "see https://github.com/oaustegard/remex/issues/42"
    )
