"""Tests for data quality validation in DataIngestion component."""
from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.components.data_ingestion import DataIngestion
from energy_forecasting.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from energy_forecasting.exception import EnergyException


def test_validate_frequency_passes_for_valid_hourly_data():
    """Test that validation passes for properly formatted hourly data."""
    config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    ingestion = DataIngestion(data_ingestion_config=config)

    # Create valid hourly data
    df = pd.DataFrame({
        "unique_id": ["series_1"] * 24,
        "ds": pd.date_range("2024-01-01", periods=24, freq="h"),
        "y": range(24),
    })

    # Should not raise
    validated_df = ingestion._validate_raw_data(df)
    assert len(validated_df) == 24


def test_validate_frequency_fails_for_large_gaps():
    """Test that validation fails when gaps exceed 24 hours."""
    config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    ingestion = DataIngestion(data_ingestion_config=config)

    # Create data with a 48-hour gap (exceeds 24-hour threshold)
    timestamps = list(pd.date_range("2024-01-01", periods=10, freq="h"))
    # Add gap: jump 48 hours ahead
    timestamps += list(pd.date_range("2024-01-03", periods=10, freq="h"))

    df = pd.DataFrame({
        "unique_id": ["series_1"] * 20,
        "ds": timestamps,
        "y": range(20),
    })

    with pytest.raises(EnergyException, match="gaps exceeding 24 hours"):
        ingestion._validate_raw_data(df)


def test_validate_raw_data_removes_duplicates():
    """Test that duplicate (unique_id, ds) pairs are removed."""
    config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    ingestion = DataIngestion(data_ingestion_config=config)

    # Create data with duplicates
    df = pd.DataFrame({
        "unique_id": ["series_1"] * 5,
        "ds": pd.date_range("2024-01-01", periods=3, freq="h").tolist() +
             pd.date_range("2024-01-01", periods=2, freq="h").tolist(),  # 2 duplicates
        "y": [1, 2, 3, 999, 888],  # Last values should be kept
    })

    validated_df = ingestion._validate_raw_data(df)

    # Should keep only 3 unique timestamps (last occurrence)
    assert len(validated_df) == 3
    # Check that last values were kept
    assert validated_df.loc[validated_df["ds"] == "2024-01-01", "y"].values[0] == 999


def test_validate_raw_data_fails_on_missing_required_columns():
    """Test that validation fails when required columns are missing."""
    config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    ingestion = DataIngestion(data_ingestion_config=config)

    # Missing 'y' column
    df = pd.DataFrame({
        "unique_id": ["series_1"] * 10,
        "ds": pd.date_range("2024-01-01", periods=10, freq="h"),
    })

    with pytest.raises(EnergyException, match="Missing required columns"):
        ingestion._validate_raw_data(df)


def test_validate_raw_data_fails_on_null_values():
    """Test that validation fails when required columns have null values."""
    config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    ingestion = DataIngestion(data_ingestion_config=config)

    df = pd.DataFrame({
        "unique_id": ["series_1"] * 10,
        "ds": pd.date_range("2024-01-01", periods=10, freq="h"),
        "y": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],  # One null
    })

    with pytest.raises(EnergyException, match="Found null values"):
        ingestion._validate_raw_data(df)


def test_validate_raw_data_fails_on_empty_dataframe():
    """Test that validation fails on empty dataframe."""
    config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    ingestion = DataIngestion(data_ingestion_config=config)

    df = pd.DataFrame(columns=["unique_id", "ds", "y"])

    with pytest.raises(EnergyException, match="DataFrame is empty"):
        ingestion._validate_raw_data(df)


def test_validate_frequency_handles_multiple_series():
    """Test that frequency validation works across multiple time series."""
    config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    ingestion = DataIngestion(data_ingestion_config=config)

    # Create two valid series
    df = pd.DataFrame({
        "unique_id": ["series_1"] * 24 + ["series_2"] * 24,
        "ds": list(pd.date_range("2024-01-01", periods=24, freq="h")) * 2,
        "y": list(range(24)) * 2,
    })

    # Should not raise
    validated_df = ingestion._validate_raw_data(df)
    assert validated_df["unique_id"].nunique() == 2