"""Tests for PackedVectors (issue #16) and Arrow IPC format (issue #23).

Covers:
- PackedVectors creation, conversion, and round-trip correctness
- Memory savings vs CompressedVectors
- On-demand unpack_rows / unpack_at
- at_precision() Matryoshka derivation
- from_rows() database reconstruction
- ADC search and two-stage search with PackedVectors
- Arrow IPC save/load round-trip
- Arrow schema metadata preservation
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from remex import Quantizer, CompressedVectors, PackedVectors
from remex.packing import packed_nbytes


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def setup(rng):
    d = 128
    n = 2000
    pq = Quantizer(d=d, bits=4)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    query = rng.standard_normal(d).astype(np.float32)
    query /= np.linalg.norm(query)
    compressed = pq.encode(corpus)
    packed = PackedVectors.from_compressed(compressed)
    return pq, compressed, packed, query


@pytest.fixture
def setup_2bit(rng):
    d = 384
    n = 1000
    pq = Quantizer(d=d, bits=2)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    compressed = pq.encode(corpus)
    packed = PackedVectors.from_compressed(compressed)
    return pq, compressed, packed


# ------------------------------------------------------------------
# PackedVectors creation and conversion
# ------------------------------------------------------------------

class TestPackedVectorsCreation:

    def test_from_compressed_preserves_data(self, setup):
        _, comp, packed, _ = setup
        # Unpack all and compare to original indices
        unpacked = packed.unpack_rows(0, packed.n)
        np.testing.assert_array_equal(unpacked, comp.indices)

    def test_from_compressed_preserves_norms(self, setup):
        _, comp, packed, _ = setup
        np.testing.assert_array_equal(packed.norms, comp.norms)

    def test_attributes_match(self, setup):
        _, comp, packed, _ = setup
        assert packed.n == comp.n
        assert packed.d == comp.d
        assert packed.bits == comp.bits

    def test_to_compressed_roundtrip(self, setup):
        _, comp, packed, _ = setup
        comp2 = packed.to_compressed()
        np.testing.assert_array_equal(comp2.indices, comp.indices)
        np.testing.assert_array_equal(comp2.norms, comp.norms)
        assert comp2.d == comp.d
        assert comp2.bits == comp.bits

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
    def test_roundtrip_all_bits(self, rng, bits):
        d = 128
        pq = Quantizer(d=d, bits=bits)
        X = rng.standard_normal((100, d)).astype(np.float32)
        comp = pq.encode(X)
        packed = PackedVectors.from_compressed(comp)
        comp2 = packed.to_compressed()
        np.testing.assert_array_equal(comp2.indices, comp.indices)


# ------------------------------------------------------------------
# Memory savings
# ------------------------------------------------------------------

class TestPackedMemory:

    def test_packed_smaller_than_unpacked(self, setup):
        _, comp, packed, _ = setup
        # 4-bit: packed should be ~half of uint8 indices
        assert packed.nbytes < comp.nbytes_unpacked

    def test_2bit_compression(self, setup_2bit):
        _, comp, packed = setup_2bit
        # 2-bit: packed indices should be ~1/4 of uint8
        packed_idx_bytes = packed._packed.nbytes
        unpacked_idx_bytes = comp.indices.nbytes
        assert packed_idx_bytes < unpacked_idx_bytes * 0.35

    def test_resident_bytes_no_cache(self, setup):
        _, _, packed, _ = setup
        expected = packed._packed.nbytes + packed.norms.nbytes
        assert packed.resident_bytes == expected

    def test_compression_ratio(self, setup):
        _, _, packed, _ = setup
        # 4-bit d=128: ratio should be reasonable
        assert packed.compression_ratio > 5.0


# ------------------------------------------------------------------
# Row unpacking
# ------------------------------------------------------------------

class TestUnpacking:

    def test_unpack_rows_slice(self, setup):
        _, comp, packed, _ = setup
        chunk = packed.unpack_rows(10, 20)
        np.testing.assert_array_equal(chunk, comp.indices[10:20])

    def test_unpack_rows_full(self, setup):
        _, comp, packed, _ = setup
        all_rows = packed.unpack_rows(0, packed.n)
        np.testing.assert_array_equal(all_rows, comp.indices)

    def test_unpack_at_arbitrary(self, setup):
        _, comp, packed, _ = setup
        idx = np.array([0, 5, 100, 999])
        rows = packed.unpack_at(idx)
        np.testing.assert_array_equal(rows, comp.indices[idx])

    def test_unpack_at_single(self, setup):
        _, comp, packed, _ = setup
        row = packed.unpack_at(np.array([42]))
        np.testing.assert_array_equal(row, comp.indices[42:43])


# ------------------------------------------------------------------
# at_precision (Matryoshka)
# ------------------------------------------------------------------

class TestAtPrecision:

    def test_at_full_precision_returns_self(self, setup):
        _, _, packed, _ = setup
        same = packed.at_precision(packed.bits)
        assert same is packed

    def test_at_lower_precision_correct(self, setup):
        _, comp, packed, _ = setup
        packed_2 = packed.at_precision(2)
        # Should match right-shifting the original indices
        expected = comp.indices >> (comp.bits - 2)
        actual = packed_2.unpack_rows(0, packed_2.n)
        np.testing.assert_array_equal(actual, expected)

    def test_at_precision_validates_bounds(self, setup):
        _, _, packed, _ = setup
        with pytest.raises(ValueError):
            packed.at_precision(0)
        with pytest.raises(ValueError):
            packed.at_precision(packed.bits + 1)

    def test_at_precision_bits_attribute(self, setup):
        _, _, packed, _ = setup
        packed_2 = packed.at_precision(2)
        assert packed_2.bits == 2
        assert packed_2.n == packed.n
        assert packed_2.d == packed.d


# ------------------------------------------------------------------
# from_rows (database reconstruction)
# ------------------------------------------------------------------

class TestFromRows:

    def test_from_rows_bytes(self, setup):
        _, comp, packed, _ = setup
        # Extract rows as bytes
        rows = [packed._packed[i].tobytes() for i in range(packed.n)]
        reconstructed = PackedVectors.from_rows(rows, comp.norms, comp.d, comp.bits)
        np.testing.assert_array_equal(
            reconstructed.unpack_rows(0, reconstructed.n),
            comp.indices,
        )

    def test_from_rows_ndarray(self, setup):
        _, comp, packed, _ = setup
        rows = [packed._packed[i] for i in range(packed.n)]
        reconstructed = PackedVectors.from_rows(rows, comp.norms, comp.d, comp.bits)
        np.testing.assert_array_equal(
            reconstructed.unpack_rows(0, reconstructed.n),
            comp.indices,
        )


# ------------------------------------------------------------------
# subset
# ------------------------------------------------------------------

class TestPackedSubset:

    def test_subset(self, setup):
        _, comp, packed, _ = setup
        idx = np.array([0, 10, 50, 100])
        sub = packed.subset(idx)
        assert sub.n == len(idx)
        np.testing.assert_array_equal(
            sub.unpack_rows(0, sub.n),
            comp.indices[idx],
        )
        np.testing.assert_array_equal(sub.norms, comp.norms[idx])


# ------------------------------------------------------------------
# ADC search with PackedVectors
# ------------------------------------------------------------------

class TestPackedADCSearch:

    def test_adc_matches_compressed(self, setup):
        """PackedVectors ADC should match CompressedVectors ADC."""
        pq, comp, packed, query = setup
        idx_comp, scores_comp = pq.search_adc(comp, query, k=10)
        idx_packed, scores_packed = pq.search_adc(packed, query, k=10)
        np.testing.assert_array_equal(idx_comp, idx_packed)
        np.testing.assert_allclose(scores_comp, scores_packed, rtol=1e-5)

    def test_adc_precision_matches(self, setup):
        """PackedVectors ADC at reduced precision should match."""
        pq, comp, packed, query = setup
        idx_comp, scores_comp = pq.search_adc(comp, query, k=10, precision=2)
        idx_packed, scores_packed = pq.search_adc(packed, query, k=10, precision=2)
        np.testing.assert_array_equal(idx_comp, idx_packed)
        np.testing.assert_allclose(scores_comp, scores_packed, rtol=1e-5)

    def test_adc_chunk_sizes(self, setup):
        """Different chunk sizes should produce identical results."""
        pq, _, packed, query = setup
        idx_a, scores_a = pq.search_adc(packed, query, k=10, chunk_size=64)
        idx_b, scores_b = pq.search_adc(packed, query, k=10, chunk_size=4096)
        np.testing.assert_array_equal(idx_a, idx_b)
        np.testing.assert_allclose(scores_a, scores_b)

    def test_adc_scores_descending(self, setup):
        pq, _, packed, query = setup
        _, scores = pq.search_adc(packed, query, k=20)
        assert np.all(np.diff(scores) <= 1e-7)

    def test_search_raises_on_packed(self, setup):
        """Cached search() should raise TypeError for PackedVectors."""
        pq, _, packed, query = setup
        with pytest.raises(TypeError, match="PackedVectors"):
            pq.search(packed, query, k=10)


# ------------------------------------------------------------------
# Two-stage search with PackedVectors
# ------------------------------------------------------------------

class TestPackedTwoStage:

    def test_twostage_matches_compressed(self, setup):
        """Two-stage with PackedVectors should match CompressedVectors."""
        pq, comp, packed, query = setup
        idx_comp, scores_comp = pq.search_twostage(
            comp, query, k=10, candidates=200
        )
        idx_packed, scores_packed = pq.search_twostage(
            packed, query, k=10, candidates=200
        )
        np.testing.assert_array_equal(idx_comp, idx_packed)
        np.testing.assert_allclose(scores_comp, scores_packed, rtol=1e-5)

    def test_twostage_returns_correct_k(self, setup):
        pq, _, packed, query = setup
        idx, scores = pq.search_twostage(packed, query, k=10, candidates=200)
        assert len(idx) == 10
        assert len(scores) == 10

    def test_twostage_scores_descending(self, setup):
        pq, _, packed, query = setup
        _, scores = pq.search_twostage(packed, query, k=10, candidates=200)
        assert np.all(np.diff(scores) <= 1e-7)

    def test_twostage_large_candidates_matches_adc(self, setup):
        """With candidates=n, two-stage fine rerank should match full search."""
        pq, comp, packed, query = setup
        idx_full, _ = pq.search(comp, query, k=10)
        idx_ts, _ = pq.search_twostage(
            packed, query, k=10, candidates=packed.n
        )
        np.testing.assert_array_equal(idx_full, idx_ts)


# ------------------------------------------------------------------
# Serialization: .npz
# ------------------------------------------------------------------

class TestPackedSerialization:

    def test_save_load_roundtrip(self, setup, tmp_path):
        _, comp, packed, _ = setup
        path = str(tmp_path / "packed.npz")
        packed.save(path)
        loaded = PackedVectors.load(path)
        np.testing.assert_array_equal(
            loaded.unpack_rows(0, loaded.n),
            comp.indices,
        )
        np.testing.assert_array_equal(loaded.norms, comp.norms)
        assert loaded.d == comp.d
        assert loaded.bits == comp.bits
        assert loaded.n == comp.n


# ------------------------------------------------------------------
# Arrow IPC (Feather v2) — issue #23
# ------------------------------------------------------------------

def _has_pyarrow():
    try:
        import pyarrow
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_pyarrow(), reason="pyarrow not installed")
class TestArrowIPC:

    def test_packed_save_load_roundtrip(self, setup, tmp_path):
        _, comp, packed, _ = setup
        path = str(tmp_path / "test.arrow")
        packed.save_arrow(path, seed=42)
        loaded = PackedVectors.load_arrow(path)
        np.testing.assert_array_equal(
            loaded.unpack_rows(0, loaded.n),
            comp.indices,
        )
        np.testing.assert_allclose(loaded.norms, comp.norms)
        assert loaded.d == comp.d
        assert loaded.bits == comp.bits
        assert loaded.n == comp.n

    def test_compressed_save_load_roundtrip(self, setup, tmp_path):
        """CompressedVectors.save_arrow / load_arrow round-trip."""
        _, comp, _, _ = setup
        path = str(tmp_path / "test.arrow")
        comp.save_arrow(path, seed=42)
        loaded = CompressedVectors.load_arrow(path)
        np.testing.assert_array_equal(loaded.indices, comp.indices)
        np.testing.assert_allclose(loaded.norms, comp.norms)
        assert loaded.d == comp.d
        assert loaded.bits == comp.bits

    def test_schema_metadata(self, setup, tmp_path):
        """Arrow file should contain quantizer params in schema metadata."""
        import pyarrow.feather as feather
        _, _, packed, _ = setup
        path = str(tmp_path / "meta.arrow")
        packed.save_arrow(path, seed=42, partition="shard_0")
        table = feather.read_table(path)
        meta = table.schema.metadata
        assert int(meta[b"d"]) == packed.d
        assert int(meta[b"bits"]) == packed.bits
        assert int(meta[b"n"]) == packed.n
        assert int(meta[b"seed"]) == 42
        assert meta[b"partition"] == b"shard_0"

    def test_arrow_columns(self, setup, tmp_path):
        """Arrow file should have norms (float32) and packed_indices (FixedSizeBinary)."""
        import pyarrow as pa
        import pyarrow.feather as feather
        _, _, packed, _ = setup
        path = str(tmp_path / "cols.arrow")
        packed.save_arrow(path)
        table = feather.read_table(path)
        assert table.schema.field("norms").type == pa.float32()
        idx_type = table.schema.field("packed_indices").type
        assert isinstance(idx_type, pa.FixedSizeBinaryType)
        assert idx_type.byte_width == packed._row_bytes

    def test_memory_map_load(self, setup, tmp_path):
        """Loading with memory_map=True should produce correct results."""
        _, comp, packed, _ = setup
        path = str(tmp_path / "mmap.arrow")
        packed.save_arrow(path)
        loaded = PackedVectors.load_arrow(path, memory_map=True)
        np.testing.assert_array_equal(
            loaded.unpack_rows(0, loaded.n),
            comp.indices,
        )

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
    def test_arrow_roundtrip_all_bits(self, rng, bits, tmp_path):
        d = 128
        pq = Quantizer(d=d, bits=bits)
        X = rng.standard_normal((50, d)).astype(np.float32)
        comp = pq.encode(X)
        packed = PackedVectors.from_compressed(comp)

        path = str(tmp_path / f"test_{bits}bit.arrow")
        packed.save_arrow(path, seed=pq.seed)
        loaded = PackedVectors.load_arrow(path)

        np.testing.assert_array_equal(
            loaded.unpack_rows(0, loaded.n),
            comp.indices,
        )

    def test_arrow_import_error(self, setup, tmp_path, monkeypatch):
        """Should raise ImportError with helpful message when pyarrow missing."""
        _, _, packed, _ = setup
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("pyarrow", "pyarrow.feather"):
                raise ImportError("No module named 'pyarrow'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyarrow"):
            packed.save_arrow(str(tmp_path / "fail.arrow"))
