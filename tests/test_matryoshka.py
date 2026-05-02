"""Tests for Matryoshka bit precision (nested codebooks + two-stage search)."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from remex import Quantizer
from remex.codebook import nested_codebooks, lloyd_max_codebook


class TestNestedCodebooks:

    def test_max_bits_unchanged(self):
        nested = nested_codebooks(128, 4)
        _, centroids = lloyd_max_codebook(128, 4)
        np.testing.assert_array_equal(nested[4], centroids)

    def test_all_levels_present(self):
        nested = nested_codebooks(128, 4)
        assert set(nested.keys()) == {1, 2, 3, 4}

    def test_centroid_count(self):
        nested = nested_codebooks(128, 4)
        for bits, centroids in nested.items():
            assert len(centroids) == 2**bits

    def test_centroids_near_optimal(self):
        d = 384
        nested = nested_codebooks(d, 4)
        for bits in [2, 3]:
            _, optimal = lloyd_max_codebook(d, bits)
            sigma = 1.0 / np.sqrt(d)
            max_dev = np.max(np.abs(nested[bits] - optimal)) / sigma
            assert max_dev < 0.15, f"{bits}-bit deviation {max_dev:.3f}s > 0.15s"

    def test_centroids_monotonic(self):
        nested = nested_codebooks(128, 4)
        for bits, centroids in nested.items():
            assert np.all(np.diff(centroids) > 0)

    def test_8bit_nesting(self):
        nested = nested_codebooks(384, 8)
        assert set(nested.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}
        assert len(nested[8]) == 256
        assert len(nested[1]) == 2


class TestPrecisionParameter:

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(42)
        d = 128
        pq = Quantizer(d=d, bits=4)
        corpus = rng.standard_normal((500, d)).astype(np.float32)
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        query = rng.standard_normal(d).astype(np.float32)
        query = query / np.linalg.norm(query)
        compressed = pq.encode(corpus)
        return pq, compressed, query, corpus

    def test_none_equals_full(self, setup):
        pq, compressed, query, _ = setup
        idx_none, scores_none = pq.search(compressed, query, k=10)
        idx_full, scores_full = pq.search(compressed, query, k=10, precision=pq.bits)
        np.testing.assert_array_equal(idx_none, idx_full)
        np.testing.assert_array_almost_equal(scores_none, scores_full)

    def test_lower_precision_different(self, setup):
        pq, compressed, query, _ = setup
        idx_4, _ = pq.search(compressed, query, k=10, precision=4)
        idx_2, _ = pq.search(compressed, query, k=10, precision=2)
        assert not np.array_equal(idx_4, idx_2)

    def test_precision_bounds(self, setup):
        pq, compressed, query, _ = setup
        with pytest.raises(ValueError):
            pq.search(compressed, query, precision=0)
        with pytest.raises(ValueError):
            pq.search(compressed, query, precision=pq.bits + 1)

    def test_mse_increases_with_lower_precision(self, setup):
        pq, _, _, corpus = setup
        mses = [pq.mse(corpus[:50], precision=p) for p in range(1, pq.bits + 1)]
        for i in range(len(mses) - 1):
            assert mses[i] >= mses[i + 1]

    def test_decode_precision(self, setup):
        pq, compressed, _, _ = setup
        dec_4 = pq.decode(compressed, precision=4)
        dec_2 = pq.decode(compressed, precision=2)
        uniq_4 = len(np.unique(np.round(dec_4[:, 0], 6)))
        uniq_2 = len(np.unique(np.round(dec_2[:, 0], 6)))
        assert uniq_2 <= uniq_4


class TestTwoStageSearch:

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(42)
        d = 128
        pq = Quantizer(d=d, bits=4)
        corpus = rng.standard_normal((1000, d)).astype(np.float32)
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        query = rng.standard_normal(d).astype(np.float32)
        query = query / np.linalg.norm(query)
        compressed = pq.encode(corpus)
        return pq, compressed, query

    def test_returns_correct_k(self, setup):
        pq, compressed, query = setup
        idx, scores = pq.search_twostage(compressed, query, k=10, candidates=200)
        assert len(idx) == 10
        assert len(scores) == 10

    def test_indices_in_range(self, setup):
        pq, compressed, query = setup
        idx, _ = pq.search_twostage(compressed, query, k=10, candidates=200)
        assert np.all(idx >= 0)
        assert np.all(idx < compressed.n)

    def test_scores_descending(self, setup):
        pq, compressed, query = setup
        _, scores = pq.search_twostage(compressed, query, k=10, candidates=200)
        assert np.all(np.diff(scores) <= 1e-7)

    def test_large_candidates_matches_full(self, setup):
        pq, compressed, query = setup
        idx_full, _ = pq.search(compressed, query, k=10)
        idx_ts, _ = pq.search_twostage(compressed, query, k=10, candidates=compressed.n)
        np.testing.assert_array_equal(idx_full, idx_ts)

    def test_default_coarse_precision(self, setup):
        pq, compressed, query = setup
        idx, scores = pq.search_twostage(compressed, query, k=10)
        assert len(idx) == 10


class TestPrecisionOneBit:
    """Coverage for precision=1 retrieval. The 1-bit Matryoshka extraction
    is special: the MSB of an n-bit Lloyd-Max index *is* the sign bit of the
    rotated coordinate, so 1-bit-from-n-bit must equal standalone 1-bit
    encoding of the same vectors (zero nesting penalty).
    """

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(42)
        d = 128
        n = 500
        corpus = rng.standard_normal((n, d)).astype(np.float32)
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        query = rng.standard_normal(d).astype(np.float32)
        query = query / np.linalg.norm(query)
        queries = rng.standard_normal((20, d)).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        return d, corpus, query, queries

    def test_search_precision_1(self, setup):
        d, corpus, query, _ = setup
        pq = Quantizer(d=d, bits=8)
        cv = pq.encode(corpus)
        idx, scores = pq.search(cv, query, k=10, precision=1)
        assert idx.shape == (10,)
        assert np.all(idx >= 0) and np.all(idx < cv.n)
        assert np.all(np.diff(scores) <= 1e-7)

    def test_search_adc_precision_1(self, setup):
        d, corpus, query, _ = setup
        pq = Quantizer(d=d, bits=8)
        cv = pq.encode(corpus)
        idx, scores = pq.search_adc(cv, query, k=10, precision=1)
        assert idx.shape == (10,)
        assert np.all(idx >= 0) and np.all(idx < cv.n)

    def test_search_batch_precision_1(self, setup):
        d, corpus, _, queries = setup
        pq = Quantizer(d=d, bits=8)
        cv = pq.encode(corpus)
        idx, scores = pq.search_batch(cv, queries, k=10, precision=1)
        assert idx.shape == (queries.shape[0], 10)
        assert np.all(idx >= 0) and np.all(idx < cv.n)

    def test_search_twostage_coarse_precision_1(self, setup):
        d, corpus, query, _ = setup
        pq = Quantizer(d=d, bits=8)
        cv = pq.encode(corpus)
        idx, scores = pq.search_twostage(
            cv, query, k=10, candidates=100, coarse_precision=1
        )
        assert idx.shape == (10,)
        assert np.all(idx >= 0) and np.all(idx < cv.n)

    def test_matryoshka_1bit_equals_standalone_1bit(self, setup):
        """MSB of an 8-bit Lloyd-Max code is the sign bit, which is exactly the
        1-bit code. So a search at precision=1 against an 8-bit encoding must
        return identical top-k as a 1-bit standalone encoding. This is the
        empirical 0% nesting penalty observed on SPECTER2.
        """
        d, corpus, _, queries = setup
        pq8 = Quantizer(d=d, bits=8, seed=7)
        pq1 = Quantizer(d=d, bits=1, seed=7)  # same seed → same rotation
        cv8 = pq8.encode(corpus)
        cv1 = pq1.encode(corpus)
        idx8, _ = pq8.search_batch(cv8, queries, k=10, precision=1)
        idx1, _ = pq1.search_batch(cv1, queries, k=10)
        np.testing.assert_array_equal(idx8, idx1)

    def test_decode_precision_1_coarser_than_full(self, setup):
        """Decoding at precision=1 must produce a coarser reconstruction than
        full precision. Verified via reconstruction MSE rather than per-coord
        unique counts: decode returns the inverse-rotated linear combination
        of centroids, so each output coordinate has many unique values even
        when only 2 centroids drove each rotated coordinate.
        """
        d, corpus, _, _ = setup
        pq = Quantizer(d=d, bits=8)
        cv = pq.encode(corpus)
        dec_1 = pq.decode(cv, precision=1)
        dec_8 = pq.decode(cv, precision=8)
        mse_1 = float(np.mean((corpus - dec_1) ** 2))
        mse_8 = float(np.mean((corpus - dec_8) ** 2))
        assert mse_1 > mse_8, f"1-bit MSE {mse_1:.6f} should exceed 8-bit MSE {mse_8:.6f}"


class TestSubset:

    def test_subset_correct(self):
        rng = np.random.default_rng(42)
        d = 64
        pq = Quantizer(d=d, bits=4)
        corpus = rng.standard_normal((100, d)).astype(np.float32)
        comp = pq.encode(corpus)
        idx = np.array([5, 10, 50])
        sub = comp.subset(idx)
        assert sub.n == 3
        np.testing.assert_array_equal(sub.indices[0], comp.indices[5])
        np.testing.assert_array_equal(sub.indices[2], comp.indices[50])
        np.testing.assert_array_almost_equal(sub.norms[1], comp.norms[10])
