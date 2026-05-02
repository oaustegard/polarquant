"""GPU encode kernel for `remex encode --device gpu`.

Stub for issue #42. The CPU contract (`src/quantizer.mojo::encode_batch`)
is the reference; this kernel must produce byte-identical packed indices
and identical norms (modulo documented FP-order tolerance) for the same
`(R, codebook)` and the same input.

## Intended kernel shape

`encode_batch` has two reductions per row that need GPU mapping:

1. Per-row norm: `nm = sqrt(sum_j X[i, j]^2)` — one reduction per row.
2. Rotation: `rotated[i, k] = sum_j R[k, j] * X[i, j] / nm` — a (n, d) ×
   (d, d) matvec, the dominant cost.

Followed by a per-coordinate `searchsorted` into `boundaries` (length
`2^bits - 1`) — a tiny per-element scan, embarrassingly parallel.

The natural mapping is:

- **Rotation**: blocked GEMM on device (`MAX ops.matmul` or a hand-rolled
  tile kernel) computing `X_rot = X @ R.T`, then per-row scaling by
  `1/nm`. Fuse the norm reduction into the same kernel pass to avoid an
  extra D2H/H2D round-trip.
- **Searchsorted + pack**: one thread per output index, binary search
  into the boundaries (which fit in shared memory: 15 floats at 4-bit).
  Pack inside the same kernel to write straight to the output `.pq`
  buffer.

## Numerics caveat

`encode_batch` parity test (`tests/test_encode.mojo`) currently asserts
**byte-identical** packed indices vs Python. A blocked GPU GEMM changes
reduction order, which can flip a coordinate that sits exactly on a
boundary. Acceptance for the GPU path should match issue #42's "byte
identical modulo documented FP-order tolerance" — expect a small number
of borderline coordinates to differ and document the tolerance.
"""

from std.memory import UnsafePointer
from src.quantizer import Quantizer


def gpu_encode_batch(q: Quantizer,
                     X: UnsafePointer[Float32, MutExternalOrigin],
                     n: Int,
                     mut indices_out: UnsafePointer[UInt8, MutExternalOrigin],
                     mut norms_out: UnsafePointer[Float32, MutExternalOrigin]) raises:
    """GPU mirror of `encode_batch`. See `src/quantizer.mojo::encode_batch`.

    Same input/output contract as the CPU path; the host buffers `X`,
    `indices_out`, `norms_out` are H2D-staged internally.

    Stub: raises until the MAX kernel lands. See issue #42.
    """
    raise Error(
        "gpu_encode_batch: not yet implemented — "
        "see https://github.com/oaustegard/remex/issues/42"
    )
