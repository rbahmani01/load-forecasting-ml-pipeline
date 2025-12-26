"""Tests for model evaluation and selection logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_forecasting.components.model_evaluation import ModelEvaluation
from energy_forecasting.entity.config_entity import ModelEvaluationConfig, TrainingPipelineConfig
from energy_forecasting.entity.artifact_entity import ModelTrainerArtifact


def test_load_rmse_prefers_test_metrics(tmp_path: Path):
    """Test that _load_rmse prefers rmse_test over rmse."""
    config = ModelEvaluationConfig(training_pipeline_config=TrainingPipelineConfig())
    evaluator = ModelEvaluation(
        config=config,
        model_trainer_artifact=ModelTrainerArtifact(
            model_path=tmp_path / "model.pkl",
            metrics_path=tmp_path / "metrics.json",
            test_data_path=tmp_path / "test.csv",
            rmse=5.0,
            mae=3.0,
        ),
    )

    # Create metrics with both rmse and rmse_test
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({
            "rmse": 5.0,
            "rmse_test": 2.5,
            "mae": 3.0,
        })
    )

    rmse = evaluator._load_rmse(metrics_path)

    # Should prefer rmse_test
    assert rmse == 2.5


def test_load_rmse_falls_back_to_rmse(tmp_path: Path):
    """Test that _load_rmse falls back to rmse when rmse_test missing."""
    config = ModelEvaluationConfig(training_pipeline_config=TrainingPipelineConfig())
    evaluator = ModelEvaluation(
        config=config,
        model_trainer_artifact=ModelTrainerArtifact(
            model_path=tmp_path / "model.pkl",
            metrics_path=tmp_path / "metrics.json",
            test_data_path=tmp_path / "test.csv",
            rmse=3.5,
            mae=2.0,
        ),
    )

    # Create metrics without rmse_test
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({
            "rmse": 3.5,
            "mae": 2.0,
        })
    )

    rmse = evaluator._load_rmse(metrics_path)

    # Should use rmse
    assert rmse == 3.5