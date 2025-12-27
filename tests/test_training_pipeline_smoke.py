"""
Smoke test for training pipeline - checks instantiation without actually running.

This ensures:
- No import errors
- Configuration loads correctly
- Pipeline can be created
- All components are properly wired
"""
from __future__ import annotations

import pytest


def test_training_pipeline_can_instantiate(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify TrainPipeline can be instantiated without errors.

    This is a lightweight smoke test that catches:
    - Import errors
    - Configuration issues
    - Missing dependencies
    - Component wiring problems

    Does NOT actually run the pipeline (no DB/file operations).
    """
    # Set required env vars (even though we won't use DB)
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "test_db")
    monkeypatch.setenv("DB_USER", "test_user")

    # Import and instantiate (this is what we're testing)
    from energy_forecasting.pipline.training_pipeline import TrainPipeline

    pipeline = TrainPipeline()

    # Verify pipeline has expected attributes
    assert hasattr(pipeline, 'run_pipeline'), "Pipeline should have run_pipeline method"
    assert pipeline is not None, "Pipeline should be created"

    print("✅ TrainPipeline instantiated successfully")
