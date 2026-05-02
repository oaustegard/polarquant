"""Tests for `IVFCoarseIndex` (Mojo port of `remex.ivf.IVFCoarseIndex`).

Cross-runtime parity test against a Python-built fixture. The Python
side builds an IVF index in both modes, runs `search_coarse` and
`search_twostage` at full nprobe, and dumps everything to /tmp. The
Mojo side rebuilds the index from the same params + corpus and asserts:

  - LSH hyperplanes byte-for-byte match Python's (modulo rare libm
    Ziggurat tail rounding — the only known cross-runtime drift).
  - cell_ids match in *both* modes.
  - Per-query cell IDs match in both modes.
  - search_coarse with `nprobe = n_cells` reproduces `Quantizer.adc_search`
    byte-identically (full precision) and at `precision=1`.
  - search_twostage with `nprobe = n_cells` reproduces
    `Quantizer.search_twostage`.
  - probe_cells multi-probe order: first `n_bits + 1` cells = q_cell
    plus all single-bit flips of it (Hamming distance 0, then 1).

Closes the Mojo half of issue #61.

Setup (run before this test):

    python remex/mojo/tests/build_ivf_fixture.py
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc, UnsafePointer
from src.codebook import Codebook, nested_codebooks_from
from src.ivf import (
    IVFCoarseIndex,
    MODE_LSH,
    MODE_ROTATED_PREFIX,
    build_ivf,
    candidate_count,
    candidate_indices,
    probe_cells,
    query_cell,
    search_coarse,
    search_twostage,
)
from src.matrix import Matrix
from src.npy import load_npy_2d_f32
from src.packed_vectors import PackedVectors, from_pq_bytes
from src.params_format import load_params
from src.pq_format import load_pq
from src.quantizer import Quantizer


def _abs(x: Float32) -> Float32:
    return -x if x < Float32(0.0) else x


# ---------------------------------------------------------------------------
# Fixture loader: load /tmp/_ivf_* once, return everything the tests need.
# ---------------------------------------------------------------------------


struct Fixture(Movable):
    """Container for the Python-built IVF fixture."""
    var meta_n: Int
    var meta_d: Int
    var meta_bits: Int
    var meta_n_bits: Int
    var meta_seed: UInt64
    var meta_n_q: Int
    var meta_k: Int
    var meta_candidates: Int
    var meta_coarse_precision: Int

    fn __init__(out self):
        self.meta_n = 0
        self.meta_d = 0
        self.meta_bits = 0
        self.meta_n_bits = 0
        self.meta_seed = UInt64(0)
        self.meta_n_q = 0
        self.meta_k = 0
        self.meta_candidates = 0
        self.meta_coarse_precision = 0


def _load_meta() raises -> Fixture:
    var meta = load_npy_2d_f32(String("/tmp/_ivf_meta.npy"))
    var f = Fixture()
    f.meta_n = Int(meta.get(0, 0))
    f.meta_d = Int(meta.get(0, 1))
    f.meta_bits = Int(meta.get(0, 2))
    f.meta_n_bits = Int(meta.get(0, 3))
    f.meta_seed = UInt64(Int(meta.get(0, 4)))
    f.meta_n_q = Int(meta.get(0, 5))
    f.meta_k = Int(meta.get(0, 6))
    f.meta_candidates = Int(meta.get(0, 7))
    f.meta_coarse_precision = Int(meta.get(0, 8))
    return f^


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _build_quantizer(d: Int, bits: Int, seed: UInt64) raises -> Quantizer:
    """Load (R, codebook) from the fixture's .params and wrap them."""
    var R = Matrix(d, d)
    var cb = Codebook(bits)
    load_params(String("/tmp/_ivf.params"), R, cb)
    return Quantizer(R^, cb^, d, bits, seed)


def _load_packed(d: Int, bits: Int) raises -> PackedVectors:
    """Load the corpus from the fixture's .pq into a PackedVectors.

    Copies `pq.packed_indices` and `pq.norms` into fresh buffers before
    handing them to `from_pq_bytes` — same field-aliasing workaround the
    other Mojo tests use (see `test_search_twostage.mojo` and the comment
    in `polarquant.mojo::cmd_search`). Without this, the first row of the
    resulting buffer comes back zeroed.
    """
    var pq = load_pq(String("/tmp/_ivf.pq"))
    if pq.d != d or pq.bits != bits:
        raise Error("_load_packed: .pq d/bits mismatch with metadata")
    var n = pq.n
    var nbytes = pq.packed_bytes
    var local_packed = alloc[UInt8](nbytes)
    for i in range(nbytes):
        local_packed[i] = pq.packed_indices[i]
    var local_norms = alloc[Float32](n)
    for i in range(n):
        local_norms[i] = pq.norms[i]
    var packed = from_pq_bytes(local_packed, nbytes, local_norms, n, d, bits)
    local_packed.free()
    local_norms.free()
    return packed^


def _copy_query(Q_npy_rows: Int, Q_npy_cols: Int, qi: Int,
                Q_data: UnsafePointer[Float32, MutExternalOrigin],
                d: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
    """Fresh d-element query buffer from row qi of an Npy2D."""
    var qbuf = alloc[Float32](d)
    for j in range(d):
        qbuf[j] = Q_data[qi * Q_npy_cols + j]
    return qbuf


# ---------------------------------------------------------------------------
# 1. Hyperplane parity (LSH only)
# ---------------------------------------------------------------------------


def test_lsh_hyperplanes_parity() raises:
    """Mojo-built LSH hyperplanes must equal Python's
    `np.random.default_rng(seed).standard_normal((n_bits, d))` byte-for-byte
    (modulo rare libm Ziggurat tail rounding).
    """
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_LSH, f.meta_seed)

    var hp = load_npy_2d_f32(String("/tmp/_ivf_lsh_hyperplanes.npy"))
    if hp.rows != f.meta_n_bits or hp.cols != f.meta_d:
        raise Error("test_lsh_hyperplanes_parity: hyperplane shape mismatch")

    var max_diff: Float32 = Float32(0.0)
    var n_diff: Int = 0
    for b in range(f.meta_n_bits):
        for k in range(f.meta_d):
            var got = ivf.hyperplanes[b * f.meta_d + k]
            var exp = hp.get(b, k)
            var diff = _abs(got - exp)
            if diff > max_diff:
                max_diff = diff
            if diff > Float32(0.0):
                n_diff += 1
    print("LSH hyperplane max diff:", max_diff,
          "  nonzero diffs:", n_diff,
          "/", f.meta_n_bits * f.meta_d)
    # libm Ziggurat tail can drift in the last ulp (~1e-7). Allow a tiny tol.
    assert_true(max_diff < Float32(1e-5))
    print("[test_lsh_hyperplanes_parity] ok")


# ---------------------------------------------------------------------------
# 2. Cell ID parity in both modes
# ---------------------------------------------------------------------------


def test_lsh_cell_ids_parity() raises:
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_LSH, f.meta_seed)

    var expected = load_npy_2d_f32(String("/tmp/_ivf_lsh_cell_ids.npy"))
    # 1D arrays come in as (1, n) in our loader (or (n, 1) — accept either).
    var n_expected = expected.rows * expected.cols
    if n_expected != f.meta_n:
        raise Error("test_lsh_cell_ids_parity: expected.size != n")

    var n_diff: Int = 0
    for i in range(f.meta_n):
        var got = Int(ivf.cell_ids[i])
        var exp_idx = i if expected.cols == 1 else i  # both layouts flatten the same
        var exp: Int
        if expected.rows == 1:
            exp = Int(expected.get(0, i))
        else:
            exp = Int(expected.get(i, 0))
        if got != exp:
            n_diff += 1
            if n_diff <= 5:
                print("  LSH cell_id mismatch row", i, ": got", got, "exp", exp)
    print("LSH cell_id mismatches:", n_diff, "/", f.meta_n)
    assert_equal(n_diff, 0)
    print("[test_lsh_cell_ids_parity] ok")


def test_rotated_prefix_cell_ids_parity() raises:
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_ROTATED_PREFIX, f.meta_seed)

    var expected = load_npy_2d_f32(String("/tmp/_ivf_rp_cell_ids.npy"))
    var n_expected = expected.rows * expected.cols
    if n_expected != f.meta_n:
        raise Error("test_rotated_prefix_cell_ids_parity: expected.size != n")

    var n_diff: Int = 0
    for i in range(f.meta_n):
        var got = Int(ivf.cell_ids[i])
        var exp: Int
        if expected.rows == 1:
            exp = Int(expected.get(0, i))
        else:
            exp = Int(expected.get(i, 0))
        if got != exp:
            n_diff += 1
            if n_diff <= 5:
                print("  rotated_prefix cell_id mismatch row", i,
                      ": got", got, "exp", exp)
    print("rotated_prefix cell_id mismatches:", n_diff, "/", f.meta_n)
    assert_equal(n_diff, 0)
    print("[test_rotated_prefix_cell_ids_parity] ok")


# ---------------------------------------------------------------------------
# 3. CSR layout invariant: every row in cell c has cell_ids[row] == c
# ---------------------------------------------------------------------------


def test_csr_layout_correct() raises:
    """sorted_idx[cell_offsets[c]:cell_offsets[c+1]] must contain exactly the
    rows whose cell_id == c, and the offsets must cover [0, n) exactly once.
    """
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_LSH, f.meta_seed)

    assert_equal(Int(ivf.cell_offsets[0]), 0)
    assert_equal(Int(ivf.cell_offsets[ivf.n_cells]), f.meta_n)

    # Counts non-decreasing
    for c in range(ivf.n_cells):
        assert_true(ivf.cell_offsets[c + 1] >= ivf.cell_offsets[c])

    # Every row in cell c has cell_ids[row] == c.
    for c in range(ivf.n_cells):
        var lo = ivf.cell_offsets[c]
        var hi = ivf.cell_offsets[c + 1]
        for j in range(lo, hi):
            var row = ivf.sorted_idx[j]
            assert_equal(Int(ivf.cell_ids[row]), c)
    print("[test_csr_layout_correct] ok")


# ---------------------------------------------------------------------------
# 4. Per-query cell ID parity
# ---------------------------------------------------------------------------


def test_query_cell_parity() raises:
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf_lsh = build_ivf(q, packed, f.meta_n_bits, MODE_LSH, f.meta_seed)
    var ivf_rp = build_ivf(q, packed, f.meta_n_bits, MODE_ROTATED_PREFIX, f.meta_seed)

    var Q = load_npy_2d_f32(String("/tmp/_ivf_queries.npy"))
    if Q.rows != f.meta_n_q or Q.cols != f.meta_d:
        raise Error("test_query_cell_parity: query shape mismatch")
    var lsh_qc = load_npy_2d_f32(String("/tmp/_ivf_lsh_query_cells.npy"))
    var rp_qc = load_npy_2d_f32(String("/tmp/_ivf_rp_query_cells.npy"))

    for qi in range(f.meta_n_q):
        var qbuf = _copy_query(Q.rows, Q.cols, qi, Q.data, f.meta_d)
        var got_lsh = query_cell(ivf_lsh, q, qbuf)
        var got_rp = query_cell(ivf_rp, q, qbuf)
        var exp_lsh = Int(lsh_qc.get(0, qi)) if lsh_qc.rows == 1 else Int(lsh_qc.get(qi, 0))
        var exp_rp = Int(rp_qc.get(0, qi)) if rp_qc.rows == 1 else Int(rp_qc.get(qi, 0))
        assert_equal(got_lsh, exp_lsh)
        assert_equal(got_rp, exp_rp)
        qbuf.free()
    print("[test_query_cell_parity] ok — n_q =", f.meta_n_q)


# ---------------------------------------------------------------------------
# 5. probe_cells Hamming ordering: q_cell + single-bit flips
# ---------------------------------------------------------------------------


def test_probe_cells_hamming_ordering() raises:
    """First `n_bits + 1` cells visited must be q_cell (HD=0) + every
    single-bit flip (HD=1), in ascending cell-ID within each Hamming bucket.
    """
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_LSH, f.meta_seed)

    var n_bits = f.meta_n_bits
    var q_cell = 5  # arbitrary; just needs to fit in n_bits
    if q_cell >= ivf.n_cells:
        q_cell = ivf.n_cells // 2
    var nprobe = 1 + n_bits

    var cells = alloc[Int](nprobe)
    var got_n = probe_cells(ivf, q_cell, nprobe, cells)
    assert_equal(got_n, nprobe)
    assert_equal(cells[0], q_cell)

    # Build the set of expected single-bit flips as a sorted list.
    var expected_flips = alloc[Int](n_bits)
    for b in range(n_bits):
        expected_flips[b] = q_cell ^ (1 << b)
    # Bubble-sort the flip list (n_bits <= 16 — trivially small).
    for i in range(n_bits):
        for j in range(0, n_bits - 1 - i):
            if expected_flips[j] > expected_flips[j + 1]:
                var t = expected_flips[j]
                expected_flips[j] = expected_flips[j + 1]
                expected_flips[j + 1] = t

    for b in range(n_bits):
        assert_equal(cells[1 + b], expected_flips[b])

    cells.free()
    expected_flips.free()
    print("[test_probe_cells_hamming_ordering] ok — n_bits =", n_bits)


# ---------------------------------------------------------------------------
# 6. Full nprobe == flat ADC scan
# ---------------------------------------------------------------------------


def _check_full_nprobe(precision: Int, expected_idx_path: String,
                       expected_scores_path: String) raises:
    """Run search_coarse at nprobe=n_cells and assert top-k matches the
    Python flat ADC reference for every query at the given `precision`
    (0 = full)."""
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    # Use rotated_prefix here so the cell assignment is deterministic from
    # the .params alone — no extra Mojo/Python LSH-RNG agreement needed.
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_ROTATED_PREFIX, f.meta_seed)
    var nested = nested_codebooks_from(q.cb, f.meta_d)

    var Q = load_npy_2d_f32(String("/tmp/_ivf_queries.npy"))
    var exp_idx = load_npy_2d_f32(expected_idx_path)
    var exp_scores: UnsafePointer[Float32, MutExternalOrigin]
    var exp_scores_rows: Int = 0
    var exp_scores_cols: Int = 0
    var exp_scores_loaded = False
    if len(expected_scores_path) > 0:
        var es = load_npy_2d_f32(expected_scores_path)
        exp_scores_rows = es.rows
        exp_scores_cols = es.cols
        exp_scores = alloc[Float32](exp_scores_rows * exp_scores_cols)
        for i in range(exp_scores_rows):
            for j in range(exp_scores_cols):
                exp_scores[i * exp_scores_cols + j] = es.get(i, j)
        exp_scores_loaded = True
    else:
        exp_scores = alloc[Float32](1)

    var top_idx = alloc[Int](f.meta_k)
    var top_scores = alloc[Float32](f.meta_k)

    var n_idx_mismatch: Int = 0
    var max_score_diff: Float32 = Float32(0.0)
    for qi in range(f.meta_n_q):
        var qbuf = _copy_query(Q.rows, Q.cols, qi, Q.data, f.meta_d)
        var n_returned = search_coarse(
            ivf, q, nested, packed, qbuf,
            f.meta_k, ivf.n_cells, precision,
            top_idx, top_scores,
        )
        assert_equal(n_returned, f.meta_k)
        for r in range(f.meta_k):
            var got_idx = top_idx[r]
            var exp = Int(exp_idx.get(qi, r))
            if got_idx != exp:
                n_idx_mismatch += 1
                if n_idx_mismatch <= 5:
                    print("  q", qi, "rank", r, " got", got_idx, " exp", exp)
            if exp_scores_loaded:
                var got_s = top_scores[r]
                var es = exp_scores[qi * exp_scores_cols + r]
                var diff = _abs(got_s - es)
                var ref_mag = _abs(es)
                if ref_mag > Float32(1.0):
                    diff = diff / ref_mag
                if diff > max_score_diff:
                    max_score_diff = diff
        qbuf.free()

    print("  precision =", precision,
          " idx mismatches:", n_idx_mismatch,
          " max score diff:", max_score_diff)
    assert_equal(n_idx_mismatch, 0)
    if exp_scores_loaded:
        assert_true(max_score_diff < Float32(1e-5))

    top_idx.free()
    top_scores.free()
    exp_scores.free()


def test_full_nprobe_matches_adc_search_full_precision() raises:
    _check_full_nprobe(
        0,
        String("/tmp/_ivf_full_adc_idx.npy"),
        String("/tmp/_ivf_full_adc_scores.npy"),
    )
    print("[test_full_nprobe_matches_adc_search_full_precision] ok")


def test_full_nprobe_matches_adc_search_precision_1() raises:
    _check_full_nprobe(
        1,
        String("/tmp/_ivf_full_adc_p1_idx.npy"),
        String(""),  # only check indices at p=1 (scores are tested at full)
    )
    print("[test_full_nprobe_matches_adc_search_precision_1] ok")


# ---------------------------------------------------------------------------
# 7. Two-stage parity at full nprobe
# ---------------------------------------------------------------------------


def test_full_nprobe_matches_search_twostage() raises:
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_ROTATED_PREFIX, f.meta_seed)
    var nested = nested_codebooks_from(q.cb, f.meta_d)

    var Q = load_npy_2d_f32(String("/tmp/_ivf_queries.npy"))
    var exp_idx = load_npy_2d_f32(String("/tmp/_ivf_full_2stage_idx.npy"))
    var exp_scores = load_npy_2d_f32(String("/tmp/_ivf_full_2stage_scores.npy"))

    var top_idx = alloc[Int](f.meta_k)
    var top_scores = alloc[Float32](f.meta_k)

    var n_idx_mismatch: Int = 0
    var max_score_diff: Float32 = Float32(0.0)
    for qi in range(f.meta_n_q):
        var qbuf = _copy_query(Q.rows, Q.cols, qi, Q.data, f.meta_d)
        var n_returned = search_twostage(
            ivf, q, nested, packed, qbuf,
            f.meta_k, f.meta_candidates, ivf.n_cells,
            f.meta_coarse_precision,
            top_idx, top_scores,
        )
        assert_equal(n_returned, f.meta_k)
        for r in range(f.meta_k):
            var got_idx = top_idx[r]
            var exp = Int(exp_idx.get(qi, r))
            if got_idx != exp:
                n_idx_mismatch += 1
                if n_idx_mismatch <= 5:
                    print("  q", qi, "rank", r, " got", got_idx, " exp", exp)
            var diff = _abs(top_scores[r] - exp_scores.get(qi, r))
            var ref_mag = _abs(exp_scores.get(qi, r))
            if ref_mag > Float32(1.0):
                diff = diff / ref_mag
            if diff > max_score_diff:
                max_score_diff = diff
        qbuf.free()

    print("  twostage idx mismatches:", n_idx_mismatch,
          " max score diff:", max_score_diff)
    assert_equal(n_idx_mismatch, 0)
    assert_true(max_score_diff < Float32(1e-5))
    top_idx.free()
    top_scores.free()
    print("[test_full_nprobe_matches_search_twostage] ok")


# ---------------------------------------------------------------------------
# 8. candidate_count sanity
# ---------------------------------------------------------------------------


def test_candidate_count_full_equals_n() raises:
    """At nprobe = n_cells, every corpus row is a candidate."""
    var f = _load_meta()
    var q = _build_quantizer(f.meta_d, f.meta_bits, f.meta_seed)
    var packed = _load_packed(f.meta_d, f.meta_bits)
    var ivf = build_ivf(q, packed, f.meta_n_bits, MODE_ROTATED_PREFIX, f.meta_seed)

    var Q = load_npy_2d_f32(String("/tmp/_ivf_queries.npy"))
    var qbuf = _copy_query(Q.rows, Q.cols, 0, Q.data, f.meta_d)
    var got = candidate_count(ivf, q, qbuf, ivf.n_cells)
    assert_equal(got, f.meta_n)
    qbuf.free()
    print("[test_candidate_count_full_equals_n] ok")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() raises:
    test_lsh_hyperplanes_parity()
    test_lsh_cell_ids_parity()
    test_rotated_prefix_cell_ids_parity()
    test_csr_layout_correct()
    test_query_cell_parity()
    test_probe_cells_hamming_ordering()
    test_full_nprobe_matches_adc_search_full_precision()
    test_full_nprobe_matches_adc_search_precision_1()
    test_full_nprobe_matches_search_twostage()
    test_candidate_count_full_equals_n()
    print("[test_ivf] all passed")
