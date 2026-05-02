"""Build the cross-runtime fixture for `test_ivf.mojo`.

Run this once before the Mojo test:

    python remex/mojo/tests/build_ivf_fixture.py

Outputs (under /tmp/, all .npy float32 / .pq / .params):
  /tmp/_ivf.params              — Quantizer R + boundaries + centroids
  /tmp/_ivf.pq                  — Encoded corpus (PackedVectors-compatible)
  /tmp/_ivf_corpus.npy          — Original corpus (n, d) float32
  /tmp/_ivf_queries.npy         — Query batch (n_q, d) float32
  /tmp/_ivf_meta.npy            — (1, M) float32 metadata row
  /tmp/_ivf_lsh_hyperplanes.npy — (n_bits, d) float32 LSH hyperplanes
  /tmp/_ivf_lsh_cell_ids.npy    — (n,) cell IDs (LSH mode)
  /tmp/_ivf_rp_cell_ids.npy     — (n,) cell IDs (rotated_prefix mode)
  /tmp/_ivf_lsh_query_cells.npy — (n_q,) per-query cell IDs (LSH)
  /tmp/_ivf_rp_query_cells.npy  — (n_q,) per-query cell IDs (rotated_prefix)
  /tmp/_ivf_full_adc_idx.npy    — (n_q, k) Quantizer.adc top-k indices (ground truth)
  /tmp/_ivf_full_adc_scores.npy — (n_q, k) Quantizer.adc top-k scores
  /tmp/_ivf_full_adc_p1_idx.npy — (n_q, k) Quantizer.adc top-k indices @ precision=1
  /tmp/_ivf_full_2stage_idx.npy — (n_q, k) Quantizer.search_twostage indices
  /tmp/_ivf_full_2stage_scores.npy — (n_q, k) Quantizer.search_twostage scores

Metadata layout (all float32, all integers):
  [n, d, bits, n_bits, seed, n_q, k, candidates, coarse_precision]

The Mojo test loads these and asserts:
  - Mojo-built LSH hyperplanes == Python's (byte-identical, modulo
    rare libm tail rounding in the Ziggurat sampler).
  - Mojo-built cell_ids match Python's in *both* modes (byte-for-byte).
  - Per-query cell IDs match.
  - search_coarse with nprobe = n_cells reproduces adc_search exactly
    at precision=None and precision=1.
  - search_twostage with nprobe = n_cells reproduces Quantizer.search_twostage.
  - probe_cells produces the right Hamming-ranked order.
"""

from __future__ import annotations

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from remex import (  # noqa: E402
    IVFCoarseIndex,
    PackedVectors,
    Quantizer,
    save_params,
    save_pq,
)


# Fixture parameters. Kept small so the Mojo test runs in seconds.
N = 256
D = 16
BITS = 4
N_BITS = 4
SEED = 42
N_Q = 4
K = 10
CANDIDATES = 50
COARSE_PRECISION = 2
RNG_SEED = 0


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # Synthetic corpus + queries. Random Gaussian, no normalization needed
    # for IVF — `Quantizer.encode` divides by per-row norm internally.
    X = rng.standard_normal((N, D)).astype(np.float32)
    Q = rng.standard_normal((N_Q, D)).astype(np.float32)

    q = Quantizer(d=D, bits=BITS, seed=SEED)
    save_params("/tmp/_ivf.params", q)

    cv = q.encode(X)
    save_pq("/tmp/_ivf.pq", cv)

    np.save("/tmp/_ivf_corpus.npy", X)
    np.save("/tmp/_ivf_queries.npy", Q)

    meta = np.array(
        [[N, D, BITS, N_BITS, SEED, N_Q, K, CANDIDATES, COARSE_PRECISION]],
        dtype=np.float32,
    )
    np.save("/tmp/_ivf_meta.npy", meta)

    # Build IVF in both modes. Use PackedVectors as the Mojo side does.
    packed = PackedVectors.from_compressed(cv)

    ivf_lsh = IVFCoarseIndex(q, packed, n_bits=N_BITS, mode="lsh", seed=SEED)
    ivf_rp = IVFCoarseIndex(q, packed, n_bits=N_BITS, mode="rotated_prefix")

    # The Mojo `load_npy_2d_f32` requires a 2D shape — wrap 1D arrays as
    # (1, N) row-vectors so the loader doesn't reject them.
    np.save("/tmp/_ivf_lsh_hyperplanes.npy", ivf_lsh.hyperplanes.astype(np.float32))
    np.save(
        "/tmp/_ivf_lsh_cell_ids.npy",
        ivf_lsh.cell_ids.astype(np.float32).reshape(1, -1),
    )
    np.save(
        "/tmp/_ivf_rp_cell_ids.npy",
        ivf_rp.cell_ids.astype(np.float32).reshape(1, -1),
    )

    # Per-query cell IDs (one int per query, both modes), saved as (1, n_q).
    lsh_qc = np.zeros((1, N_Q), dtype=np.float32)
    rp_qc = np.zeros((1, N_Q), dtype=np.float32)
    for i in range(N_Q):
        lsh_qc[0, i] = ivf_lsh.query_cell(Q[i])
        rp_qc[0, i] = ivf_rp.query_cell(Q[i])
    np.save("/tmp/_ivf_lsh_query_cells.npy", lsh_qc)
    np.save("/tmp/_ivf_rp_query_cells.npy", rp_qc)

    # Ground-truth flat ADC results (full precision).
    full_idx = np.zeros((N_Q, K), dtype=np.float32)
    full_scores = np.zeros((N_Q, K), dtype=np.float32)
    for i in range(N_Q):
        ti, ts = q.search_adc(cv, Q[i], k=K)
        full_idx[i] = ti.astype(np.float32)
        full_scores[i] = ts
    np.save("/tmp/_ivf_full_adc_idx.npy", full_idx)
    np.save("/tmp/_ivf_full_adc_scores.npy", full_scores)

    # Ground-truth flat ADC at precision=1 (1-bit Matryoshka coarse).
    p1_idx = np.zeros((N_Q, K), dtype=np.float32)
    for i in range(N_Q):
        ti, _ = q.search_adc(cv, Q[i], k=K, precision=1)
        p1_idx[i] = ti.astype(np.float32)
    np.save("/tmp/_ivf_full_adc_p1_idx.npy", p1_idx)

    # Ground-truth two-stage results.
    ts_idx = np.zeros((N_Q, K), dtype=np.float32)
    ts_scores = np.zeros((N_Q, K), dtype=np.float32)
    for i in range(N_Q):
        ti, ts = q.search_twostage(
            cv,
            Q[i],
            k=K,
            candidates=CANDIDATES,
            coarse_precision=COARSE_PRECISION,
        )
        ts_idx[i] = ti.astype(np.float32)
        ts_scores[i] = ts
    np.save("/tmp/_ivf_full_2stage_idx.npy", ts_idx)
    np.save("/tmp/_ivf_full_2stage_scores.npy", ts_scores)

    print(
        f"[build_ivf_fixture] wrote /tmp/_ivf_* — "
        f"n={N} d={D} bits={BITS} n_bits={N_BITS} seed={SEED} "
        f"n_q={N_Q} k={K} candidates={CANDIDATES} "
        f"coarse_precision={COARSE_PRECISION}"
    )


if __name__ == "__main__":
    main()
