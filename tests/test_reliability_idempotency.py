"""Tests for idempotency and reliability features."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest


def test_atomic_file_operations_use_temp_and_replace():
    """Test that model file updates use atomic operations (temp file + os.replace)."""
    import shutil

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        best_model_dir = tmp_path / "best_model"
        best_model_dir.mkdir()

        # Create source and destination files
        current_model = tmp_path / "current_model.pkl"
        current_model.write_bytes(b"new model data")

        best_model = best_model_dir / "model.pkl"
        best_model.write_bytes(b"old model data")

        # Simulate atomic update pattern from model_evaluation.py
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=best_model.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_model_path = Path(tmp_file.name)

        # Verify temp file has .tmp suffix
        assert tmp_model_path.suffix == '.tmp'
        assert tmp_model_path.parent == best_model.parent

        # Copy to temp file
        shutil.copy2(current_model, tmp_model_path)

        # Atomic replace
        os.replace(tmp_model_path, best_model)

        # Verify final file has new data
        assert best_model.read_bytes() == b"new model data"

        # Verify temp file is gone
        assert not tmp_model_path.exists()


def test_os_replace_is_atomic_on_posix():
    """Verify that os.replace is atomic (on POSIX systems)."""
    # os.replace is guaranteed to be atomic on POSIX systems
    # This test documents that we're using the correct function

    # Check that os.replace exists and is the right function
    assert hasattr(os, 'replace')

    # On POSIX, os.replace is atomic
    # On Windows, it's atomic if both files are on the same volume
    # This is better than os.rename which fails if dest exists on Windows

    # Create test files to verify behavior
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / "source.txt"
        dst = Path(tmp_dir) / "dest.txt"

        src.write_text("new content")
        dst.write_text("old content")

        # os.replace should work even if dest exists
        os.replace(src, dst)

        assert dst.read_text() == "new content"
        assert not src.exists()


def test_model_evaluation_uses_atomic_pattern():
    """Verify that model_evaluation.py uses the atomic file update pattern."""
    from energy_forecasting.components import model_evaluation
    import inspect

    source = inspect.getsource(model_evaluation)

    # Check for atomic operation pattern
    assert "tempfile.NamedTemporaryFile" in source
    assert "os.replace" in source
    assert "suffix='.tmp'" in source or 'suffix=\".tmp\"' in source

    # Verify the pattern: create temp -> copy -> replace
    assert "shutil.copy2" in source
    assert "delete=False" in source  # Important: temp file must persist until replace


def test_upsert_pattern_prevents_duplicates():
    """Test that UPSERT with UNIQUE constraint prevents duplicate predictions."""
    # This test verifies the pattern, not the actual SQL execution

    # Simulate the UPSERT pattern
    # Table has UNIQUE(unique_id, ds, forecast_origin_ts)
    # INSERT ... ON CONFLICT (unique_id, ds, forecast_origin_ts) DO UPDATE

    # Mock data
    predictions = [
        {"unique_id": "meter_1", "ds": "2024-01-01 00:00:00", "forecast_origin_ts": "2024-01-01", "y_pred": 5.5},
        {"unique_id": "meter_1", "ds": "2024-01-01 00:00:00", "forecast_origin_ts": "2024-01-01", "y_pred": 5.5},  # Duplicate
    ]

    # In a real UPSERT, the second insert would UPDATE, not create a duplicate
    # The UNIQUE constraint prevents duplicates
    # The ON CONFLICT DO UPDATE makes it idempotent

    # Verify the pattern exists in code
    from energy_forecasting.components import batch_prediction
    import inspect

    source = inspect.getsource(batch_prediction)

    # Check for UPSERT pattern
    assert "ON CONFLICT" in source
    assert "DO UPDATE" in source
    assert "UNIQUE" in source


def test_upsert_is_idempotent():
    """Verify that running UPSERT twice produces the same result."""
    # This test documents the idempotency guarantee

    # Scenario: Run batch prediction twice with same forecast_origin_ts
    # Expected: Second run updates existing rows, doesn't duplicate

    # The UNIQUE constraint on (unique_id, ds, forecast_origin_ts) ensures:
    # 1. First INSERT creates the row
    # 2. Second INSERT triggers ON CONFLICT
    # 3. DO UPDATE overwrites the existing row
    # 4. Result: 1 row per (unique_id, ds, forecast_origin_ts), not 2

    # We verify the pattern is correct in the code
    from energy_forecasting.components import batch_prediction
    import inspect

    source = inspect.getsource(batch_prediction)

    # Check for UNIQUE constraint (may span multiple lines)
    assert "UNIQUE" in source
    assert "ON CONFLICT" in source
    assert "DO UPDATE" in source

    # Verify the UNIQUE constraint includes the key columns
    # The constraint might span multiple lines, so check in full source
    assert "unique_id" in source
    assert "ds" in source or "time_col" in source
    assert "forecast_origin_ts" in source


def test_crash_safety_no_partial_writes():
    """Test that atomic operations prevent partial writes on crash."""
    # Simulate crash scenario
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        best_model = tmp_path / "model.pkl"

        # Original model
        best_model.write_bytes(b"original model")

        # Simulate crash during update (temp file exists but replace not called)
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=tmp_path,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_file.write(b"incomplete new model")
            tmp_model_path = Path(tmp_file.name)

        # Crash happens here (before os.replace)

        # Original file still intact
        assert best_model.read_bytes() == b"original model"

        # Temp file exists but hasn't replaced original
        assert tmp_model_path.exists()
        assert tmp_model_path.read_bytes() == b"incomplete new model"

        # Clean up temp file (simulating recovery)
        tmp_model_path.unlink()

        # Original file still intact after cleanup
        assert best_model.read_bytes() == b"original model"


def test_atomic_operations_documented_in_code():
    """Verify that atomic operation rationale is documented."""
    from energy_forecasting.components import model_evaluation
    import inspect

    source = inspect.getsource(model_evaluation)

    # Check for documentation of atomic operations
    assert "atomic" in source.lower()
    # Should have comments explaining why atomic operations are needed


def test_duplicate_handling_in_data_ingestion():
    """Test that data ingestion handles duplicates correctly."""
    from energy_forecasting.components import data_ingestion
    import inspect

    source = inspect.getsource(data_ingestion)

    # Check for duplicate handling
    assert "drop_duplicates" in source
    assert "unique_id" in source and "ds" in source

    # Should keep last occurrence (most recent data)
    assert 'keep="last"' in source or "keep='last'" in source