from __future__ import annotations

from typing import Optional

import pandas as pd

from energy_forecasting.entity.config_entity import (
    DataIngestionConfig,
    DatabaseConfig,
)
from energy_forecasting.entity.artifact_entity import DataIngestionArtifact
from energy_forecasting.data_access.postgres_loader import (
    load_recent_table_from_db,
)
from energy_forecasting.exception import EnergyException
from energy_forecasting.logger import logger


class DataIngestion:
    """
    Data ingestion component (DB-only).

    Always:
      - reads history from Postgres using a DatabaseConfig
      - writes a raw CSV snapshot into the artifacts directory
    """

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig,
        database_config: Optional[DatabaseConfig] = None,
    ) -> None:
        self.config = data_ingestion_config
        self.database_config = database_config

    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """
        Validate raw data for required columns, data types, and data quality.
        
        Raises:
            EnergyException: If validation fails
        """
        if df.empty:
            raise EnergyException("DataFrame is empty")
        
        # Check for required columns
        required_cols = ["ds", "y", "unique_id"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise EnergyException(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Check for null values in critical columns
        nulls_per_col = df[required_cols].isnull().sum()
        if nulls_per_col.any():
            null_info = nulls_per_col[nulls_per_col > 0].to_dict()
            raise EnergyException(
                f"Found null values in required columns: {null_info}. "
                f"Cannot proceed with missing data in: {list(null_info.keys())}"
            )
        
        # Check data types
        try:
            # ds should be datetime-like
            df["ds"] = pd.to_datetime(df["ds"])
        except Exception as e:
            raise EnergyException(
                f"Column 'ds' must be datetime-convertible, but got: {e}"
            )
        
        try:
            # y should be numeric
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            if df["y"].isnull().any():
                raise ValueError("Some 'y' values are not numeric")
        except Exception as e:
            raise EnergyException(
                f"Column 'y' must be numeric, but got: {e}"
            )
        
        # unique_id should be string-like (not all null)
        if df["unique_id"].isnull().all():
            raise EnergyException("Column 'unique_id' is entirely null")
        
        # Check for duplicate (unique_id, ds) pairs
        duplicates = df[["unique_id", "ds"]].duplicated().sum()
        if duplicates > 0:
            logger.warning(
                "Found %d duplicate (unique_id, ds) pairs. "
                "Keeping last occurrence of each pair.",
                duplicates
            )
            df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")

        # Validate hourly frequency and detect gaps
        self._validate_frequency_and_gaps(df)

        # Log data quality summary
        logger.info(
            "Data validation passed: %d rows, %d series, "
            "ds range: %s to %s, y range: %.2f to %.2f",
            len(df),
            df["unique_id"].nunique(),
            df["ds"].min(),
            df["ds"].max(),
            df["y"].min(),
            df["y"].max(),
        )

        return df

    def _validate_frequency_and_gaps(self, df: pd.DataFrame, freq: str = "H", max_gap_hours: int = 24) -> None:
        """
        Validate that data has expected frequency and no large gaps.

        Args:
            df: DataFrame with 'unique_id' and 'ds' columns
            freq: Expected frequency (default: 'H' for hourly)
            max_gap_hours: Maximum acceptable gap in hours (default: 24)

        Raises:
            EnergyException: If validation fails
        """
        series_ids = df["unique_id"].unique()
        total_gaps = 0
        max_gap_found = pd.Timedelta(0)
        problematic_series = []

        for series_id in series_ids:
            series_data = df[df["unique_id"] == series_id].sort_values("ds")
            timestamps = series_data["ds"]

            if len(timestamps) < 2:
                logger.warning(
                    "Series '%s' has only %d timestamp(s), skipping frequency check",
                    series_id,
                    len(timestamps)
                )
                continue

            # Calculate time differences between consecutive timestamps
            time_diffs = timestamps.diff().dropna()

            # Expected frequency for hourly data
            expected_diff = pd.Timedelta(hours=1)
            tolerance = pd.Timedelta(minutes=5)  # Allow 5-minute tolerance

            # Find gaps (differences > expected + tolerance)
            gaps = time_diffs[time_diffs > (expected_diff + tolerance)]

            if len(gaps) > 0:
                total_gaps += len(gaps)
                series_max_gap = gaps.max()

                if series_max_gap > max_gap_found:
                    max_gap_found = series_max_gap

                # Check if any gap exceeds threshold
                max_gap_threshold = pd.Timedelta(hours=max_gap_hours)
                if series_max_gap > max_gap_threshold:
                    problematic_series.append((series_id, series_max_gap))
                    logger.warning(
                        "Series '%s' has large gap: %s (max gap threshold: %d hours)",
                        series_id,
                        series_max_gap,
                        max_gap_hours
                    )

        # Log summary
        if total_gaps > 0:
            logger.warning(
                "Found %d gaps across %d series. Max gap: %s",
                total_gaps,
                len(series_ids),
                max_gap_found
            )
        else:
            logger.info(
                "Frequency validation passed: all %d series have consistent hourly data",
                len(series_ids)
            )

        # Fail if there are critical gaps
        if problematic_series:
            error_msg = (
                f"Found {len(problematic_series)} series with gaps exceeding {max_gap_hours} hours:\n"
            )
            for series_id, gap in problematic_series[:5]:  # Show first 5
                error_msg += f"  - {series_id}: {gap}\n"
            if len(problematic_series) > 5:
                error_msg += f"  ... and {len(problematic_series) - 5} more\n"
            error_msg += (
                "Large gaps in time series can degrade model quality. "
                "Please ensure data is continuous or adjust max_gap_hours threshold."
            )
            raise EnergyException(error_msg)

    # ------------------------------------------------------------------
    def _ingest_from_db(self) -> DataIngestionArtifact:
        """
        Load recent history from Postgres and save it as raw_energy_data.csv
        under the current artifacts run directory.
        """
        if self.database_config is None:
            raise EnergyException(
                "DataIngestion requires a DatabaseConfig in DB-only mode, "
                "but no DatabaseConfig was provided."
            )

        df = load_recent_table_from_db(self.database_config)

        if df.empty:
            raise EnergyException(
                "DB ingestion returned an empty dataframe; nothing to ingest."
            )

        # Validate raw data quality
        df = self._validate_raw_data(df)
        if "ts" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"ts": "timestamp"})

        if "hour" not in df.columns and "timestamp" in df.columns:
            # derive hour of day from timestamp
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        # Ensure output directory exists
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.config.raw_data_path, index=False)

        logger.info(
            "DB ingestion: wrote %d rows to %s",
            len(df),
            self.config.raw_data_path,
        )

        return DataIngestionArtifact(
            raw_data_path=self.config.raw_data_path,
            n_rows=len(df),
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting DataIngestion in DB-only mode.")
        try:
            artifact = self._ingest_from_db()

            logger.info(
                "DataIngestion completed: path=%s, rows=%d",
                artifact.raw_data_path,
                artifact.n_rows,
            )
            return artifact
        except Exception as e:
            raise EnergyException(e) from e
