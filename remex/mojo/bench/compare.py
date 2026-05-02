"""Mojo vs NumPy benchmark for remex encode + ADC search + two-stage search.

Generates a synthetic standard-normal corpus, runs the same workload
through (a) the Python `remex.Quantizer` and (b) the Mojo binaries built
from `polarquant.mojo`. Prints a comparison table.

Usage:
    python bench/compare.py [--n 10000] [--d 384] [--bits 4]
                            [--queries 100] [--k 10]
                            [--candidates 500] [--coarse-precision K]

The Mojo binaries are expected at `bench/bench_encode`,
`bench/bench_search`, and `bench/bench_twostage` (build them with
`mojo build`).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
MOJO_DIR = THIS_DIR.parent
REPO_ROOT = MOJO_DIR.parents[1]


def _run_mojo(binary: str, args: list[str]) -> str:
    path = MOJO_DIR / "bench" / binary
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build with `mojo build -I {MOJO_DIR} "
            f"bench/{binary}.mojo -o bench/{binary}`"
        )
    out = subprocess.check_output([str(path), *args], cwd=MOJO_DIR, text=True)
    return out


def _parse_us_per_vec(out: str) -> float:
    m = re.search(r"([0-9.]+)\s+us / vector", out)
    if not m:
        raise RuntimeError(f"could not parse encode timing from: {out!r}")
    return float(m.group(1))


def _parse_ms_per_query(out: str) -> float:
    m = re.search(r"([0-9.]+)\s+ms / query", out)
    if not m:
        raise RuntimeError(f"could not parse search timing from: {out!r}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=500)
    ap.add_argument("--coarse-precision", type=int, default=None,
                    help="Coarse-stage precision for two-stage search "
                         "(default: max(1, bits - 2))")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    coarse_precision = (
        args.coarse_precision
        if args.coarse_precision is not None
        else max(1, args.bits - 2)
    )

    sys.path.insert(0, str(REPO_ROOT))
    from remex import Quantizer

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, args.d), dtype=np.float32)
    Qs = rng.standard_normal((args.queries, args.d), dtype=np.float32)

    # --- Python encode ---
    pq = Quantizer(d=args.d, bits=args.bits, seed=args.seed)
    # warmup
    pq.encode(X[:16])
    t0 = time.perf_counter()
    cv = pq.encode(X)
    py_encode_us_per_vec = (time.perf_counter() - t0) / args.n * 1e6

    # --- Python ADC search ---
    pq.search_adc(cv, Qs[0], k=args.k)  # warmup
    t0 = time.perf_counter()
    for q in Qs:
        pq.search_adc(cv, q, k=args.k)
    py_search_ms_per_q = (time.perf_counter() - t0) / args.queries * 1e3

    # --- Python two-stage search ---
    pq.search_twostage(
        cv, Qs[0], k=args.k,
        candidates=args.candidates, coarse_precision=coarse_precision,
    )  # warmup
    t0 = time.perf_counter()
    for q in Qs:
        pq.search_twostage(
            cv, q, k=args.k,
            candidates=args.candidates, coarse_precision=coarse_precision,
        )
    py_twostage_ms_per_q = (time.perf_counter() - t0) / args.queries * 1e3

    # --- Mojo encode ---
    out = _run_mojo("bench_encode", [
        "--n", str(args.n), "--d", str(args.d),
        "--bits", str(args.bits), "--seed", str(args.seed),
    ])
    mojo_encode_us_per_vec = _parse_us_per_vec(out)

    # --- Mojo search ---
    out = _run_mojo("bench_search", [
        "--n", str(args.n), "--d", str(args.d),
        "--bits", str(args.bits), "--queries", str(args.queries),
        "--k", str(args.k), "--seed", str(args.seed),
    ])
    mojo_search_ms_per_q = _parse_ms_per_query(out)

    # --- Mojo two-stage search ---
    out = _run_mojo("bench_twostage", [
        "--n", str(args.n), "--d", str(args.d),
        "--bits", str(args.bits), "--queries", str(args.queries),
        "--k", str(args.k), "--seed", str(args.seed),
        "--candidates", str(args.candidates),
        "--coarse-precision", str(coarse_precision),
    ])
    mojo_twostage_ms_per_q = _parse_ms_per_query(out)

    print()
    print(f"=== n={args.n}, d={args.d}, bits={args.bits}, "
          f"candidates={args.candidates}, coarse_precision={coarse_precision} ===")
    print(f"{'Stage':<14} {'NumPy':>14} {'Mojo':>14} {'Speedup':>10}")
    print("-" * 56)
    print(
        f"{'encode (us)':<14} "
        f"{py_encode_us_per_vec:>14.2f} {mojo_encode_us_per_vec:>14.2f} "
        f"{py_encode_us_per_vec/mojo_encode_us_per_vec:>9.2f}x"
    )
    print(
        f"{'search (ms)':<14} "
        f"{py_search_ms_per_q:>14.3f} {mojo_search_ms_per_q:>14.3f} "
        f"{py_search_ms_per_q/mojo_search_ms_per_q:>9.2f}x"
    )
    print(
        f"{'twostage (ms)':<14} "
        f"{py_twostage_ms_per_q:>14.3f} {mojo_twostage_ms_per_q:>14.3f} "
        f"{py_twostage_ms_per_q/mojo_twostage_ms_per_q:>9.2f}x"
    )
    print()
    print(
        "Note: Mojo (default --rng numpy) and NumPy share the RNG stack "
        "(PCG64 + SeedSequence + Ziggurat) since #40, so encoded indices "
        "are byte-identical at 1-4 bits. Pass --rng xoshiro to opt out."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
