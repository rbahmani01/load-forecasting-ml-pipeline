from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

from energy_forecasting.entity.config_entity import ModelEvaluationConfig
from energy_forecasting.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from energy_forecasting.logger import logger


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


class ModelEvaluation:
    """
    Compares the current model with the best (production) model so far,
    using *test data* when available.

    Logic:
        1) Compute test RMSE/MAE for current model on test.csv.
           - Store in metrics.json as rmse_test / mae_test.
        2) If a best model exists:
           - Try to compute test RMSE/MAE for it on the same test.csv.
           - Store in its metrics.json as rmse_test / mae_test (if possible).
        3) For comparison:
           - _load_rmse() prefers rmse_test if present, else rmse.
        4) If current RMSE < best RMSE → accept and overwrite best.
           Else → keep existing best.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        self.config = config
        self.model_trainer_artifact = model_trainer_artifact

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _load_rmse(self, metrics_path: Path) -> float:
        """
        Prefer rmse_test if available, otherwise fall back to rmse.
        """
        with metrics_path.open("r") as f:
            metrics = json.load(f)

        if "rmse_test" in metrics:
            return float(metrics["rmse_test"])
        return float(metrics["rmse"])

    def _compute_test_metrics(
        self,
        model_path: Path,
        metrics_path: Path,
        test_path: Path,
    ) -> Tuple[float, float]:
        """
        Compute RMSE/MAE on the test.csv for a saved MLForecast model.

        Assumes:
          - model_path points to a joblib'd MLForecast object
          - metrics.json contains 'mlforecast_config' detailing:
                unique_id_col, time_col, target_col,
                dynamic_features, static_features, ...
          - test_path points to test.csv with those columns.
        """
        import joblib

        with metrics_path.open("r") as f:
            metrics = json.load(f)

        mlf_cfg = metrics.get("mlforecast_config")
        h_cfg = mlf_cfg.get("h", None)
        if h_cfg is None:
            raise ValueError("mlforecast_config must contain 'h' (forecast horizon).")
        h_eval = int(h_cfg)

        id_col = mlf_cfg["unique_id_col"]
        time_col = mlf_cfg["time_col"]
        target_col = mlf_cfg["target_col"]
        dyn_cols = mlf_cfg.get("dynamic_features") or []
        static_cols = mlf_cfg.get("static_features") or []

        logger.info(
            "Evaluating model at %s on test data %s (id=%s, time=%s, target=%s)",
            model_path,
            test_path,
            id_col,
            time_col,
            target_col,
        )

        test_df = pd.read_csv(test_path)

        # parse timestamps consistently with trainer
        if time_col not in test_df.columns:
            raise ValueError(f"Test data missing time column '{time_col}'")

        test_df[time_col] = pd.to_datetime(
            test_df[time_col],
            utc=True,
            errors="coerce",
        )
        na_mask = test_df[time_col].isna()
        if na_mask.any():
            n_bad = int(na_mask.sum())
            logger.warning(
                "Dropping %d rows with invalid '%s' timestamps during evaluation.",
                n_bad,
                time_col,
            )
            test_df = test_df.loc[~na_mask].copy()

        # make timestamps naive (no timezone)
        test_df[time_col] = test_df[time_col].dt.tz_convert(None)

        required_cols = [id_col, time_col, target_col] + dyn_cols + static_cols
        missing = [c for c in required_cols if c not in test_df.columns]
        if missing:
            raise ValueError(
                f"Test data at {test_path} is missing columns: {missing}"
            )

        df_test = test_df[required_cols].copy()

        # load model (MLForecast)
        model = joblib.load(model_path)

        df_test = test_df[required_cols].copy()

        # sort for consistency
        df_test = df_test.sort_values([id_col, time_col])

        # 1) use horizon from config
        h_eval = int(mlf_cfg.get("h", 24))

        # 2) get the expected future index from the model
        future_index = model.make_future_dataframe(h=h_eval)

        # future_index should contain [id_col, time_col] for each id × horizon
        # 3) merge exogenous features from test onto that future index
        future_X = future_index.merge(
            df_test[[id_col, time_col] + dyn_cols + static_cols],
            on=[id_col, time_col],
            how="left",
        )

        # Check for missing exogenous values - fail evaluation if any are missing
        # to avoid biasing metrics by removing hard-to-predict timestamps
        missing_rows = future_X[dyn_cols + static_cols].isna().any(axis=1)
        n_missing = int(missing_rows.sum())
        if n_missing > 0:
            missing_info = future_X.loc[missing_rows, [id_col, time_col]].head(10)
            raise ValueError(
                f"Found {n_missing} rows with missing exogenous features in evaluation. "
                f"Cannot evaluate model fairly by dropping these rows as it would bias metrics. "
                f"First few missing rows:\n{missing_info}\n"
                f"Please ensure all exogenous features are available for the entire test window."
            )

        # 4) predict with that X_df
        preds = model.predict(h=h_eval, X_df=future_X)


        # Find prediction column: all non-id/time columns
        pred_cols = [c for c in preds.columns if c not in (id_col, time_col)]
        if not pred_cols:
            raise ValueError(
                f"No prediction columns found in preds. Got columns: {list(preds.columns)}"
            )
        pred_col = pred_cols[0]

        merged = df_test.merge(
            preds[[id_col, time_col, pred_col]],
            on=[id_col, time_col],
            how="inner",
        )

        if merged.empty:
            raise ValueError("No overlapping rows between test_df and predictions.")

        # -------------------------------------------------
        # NEW: save test data + predictions to CSV
        # -------------------------------------------------
        # Example filename: metrics_test_predictions.csv in same folder as metrics.json
        preds_out_path = metrics_path.with_name(metrics_path.stem + "_test_predictions.csv")
        merged.to_csv(preds_out_path, index=False)
        logger.info("Saved test targets + predictions to %s", preds_out_path)

        # -------------------------------------------------
        # Compute metrics
        # -------------------------------------------------
        y_true = merged[target_col].values
        y_pred = merged[pred_col].values

        rmse_val = _rmse(y_true, y_pred)
        mae_val = float(mean_absolute_error(y_true, y_pred))

        logger.info(
            "Computed test metrics for %s -> RMSE=%.4f, MAE=%.4f",
            model_path,
            rmse_val,
            mae_val,
        )
        return rmse_val, mae_val

    def _register_model_with_mlflow(
        self,
        model_path: Path,
        metrics_path: Path,
        model_name: str = "energy-load-forecaster",
        stage: str = "Staging",
    ) -> Optional[int]:
        """Register model with MLflow for version control and audit trail."""
        try:
            with open(metrics_path, "r") as f:
                all_metrics = json.load(f)
            
            # Log only numeric metrics (skip dicts, lists, etc.)
            numeric_metrics = {
                k: v for k, v in all_metrics.items()
                if isinstance(v, (int, float))
            }
            mlflow.log_metrics(numeric_metrics)
            
            mlflow.sklearn.log_model(
                joblib.load(model_path),
                artifact_path="model",
                registered_model_name=model_name,
            )
            
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest_version = max(v.version for v in versions)
                client.transition_model_version_stage(
                    name=model_name,
                    version=str(latest_version),
                    stage=stage,
                    archive_existing_versions=False,
                )
                logger.info(
                    "Registered model %s version %s in stage %s",
                    model_name,
                    latest_version,
                    stage,
                )
                return int(latest_version)
        except Exception as e:
            logger.warning(
                "Failed to register model with MLflow: %s. Continuing with local model management.",
                e,
            )
        return None

    # ---------------------------------------------------------
    # Main
    # ---------------------------------------------------------
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logger.info("Starting ModelEvaluation.")

        best_dir = self.config.best_model_dir
        best_dir.mkdir(parents=True, exist_ok=True)

        current_model_path = self.model_trainer_artifact.model_path
        current_metrics_path = self.model_trainer_artifact.metrics_path
        test_data_path = self.model_trainer_artifact.test_data_path

        best_model_path = self.config.best_model_path
        best_metrics_path = self.config.best_metrics_path

        # -----------------------------------------------------
        # 1) Compute test metrics for current model
        # -----------------------------------------------------
        try:
            curr_rmse_test, curr_mae_test = self._compute_test_metrics(
                current_model_path,
                current_metrics_path,
                test_data_path,
            )
            # update current metrics.json with test metrics
            with current_metrics_path.open("r") as f:
                curr_metrics = json.load(f)
            curr_metrics["rmse_test"] = curr_rmse_test
            curr_metrics["mae_test"] = curr_mae_test
            current_metrics_path.write_text(
                json.dumps(curr_metrics, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(
                "Failed to compute test metrics for current model: %s", e
            )

        # -----------------------------------------------------
        # 2) Compute test metrics for best model (if exists)
        # -----------------------------------------------------
        if best_model_path.exists() and best_metrics_path.exists():
            try:
                best_rmse_test, best_mae_test = self._compute_test_metrics(
                    best_model_path,
                    best_metrics_path,
                    test_data_path,
                )
                with best_metrics_path.open("r") as f:
                    best_metrics = json.load(f)
                best_metrics["rmse_test"] = best_rmse_test
                best_metrics["mae_test"] = best_mae_test
                best_metrics_path.write_text(
                    json.dumps(best_metrics, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.info(
                    "Skipping test metrics for existing best model (incompatible with current test window): %s",
                    e,
                )

        # -----------------------------------------------------
        # 3) Load RMSE values for comparison
        #     (rmse_test if present, else rmse)
        # -----------------------------------------------------
        current_rmse = self._load_rmse(current_metrics_path)
        logger.info("Current model evaluation RMSE: %.4f", current_rmse)

        best_rmse: Optional[float] = None
        if best_metrics_path.exists():
            try:
                best_rmse = self._load_rmse(best_metrics_path)
                logger.info("Existing best model evaluation RMSE: %.4f", best_rmse)
            except Exception as e:
                logger.warning(
                    "Failed to read existing best metrics from %s: %s",
                    best_metrics_path,
                    e,
                )

        # -----------------------------------------------------
        # 4) Decide whether to accept the current model
        # -----------------------------------------------------
        if best_rmse is None:
            logger.info("No best model found yet. Accepting current model as best.")
            is_accepted = True
        else:
            if current_rmse < best_rmse:
                logger.info(
                    "Current model is better (RMSE %.4f < %.4f). Accepting.",
                    current_rmse,
                    best_rmse,
                )
                is_accepted = True
            else:
                logger.info(
                    "Current model is worse (RMSE %.4f >= %.4f). Rejecting.",
                    current_rmse,
                    best_rmse,
                )
                is_accepted = False

        # -----------------------------------------------------
        # 5) Copy artifacts if accepted (using atomic operations)
        # -----------------------------------------------------
        if is_accepted:
            # Use atomic rename to prevent corruption if process crashes
            # Copy to temp file, then atomically rename to final destination
            import tempfile
            import os

            # Atomic update for model file
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=best_model_path.parent,
                delete=False,
                suffix='.tmp'
            ) as tmp_model:
                tmp_model_path = Path(tmp_model.name)
            shutil.copy2(current_model_path, tmp_model_path)
            os.replace(tmp_model_path, best_model_path)

            # Atomic update for metrics file
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=best_metrics_path.parent,
                delete=False,
                suffix='.tmp'
            ) as tmp_metrics:
                tmp_metrics_path = Path(tmp_metrics.name)
            shutil.copy2(current_metrics_path, tmp_metrics_path)
            os.replace(tmp_metrics_path, best_metrics_path)

            logger.info(
                "Best model updated atomically: model=%s, metrics=%s",
                best_model_path,
                best_metrics_path,
            )
            
            # Register with MLflow for version control and audit trail
            mlflow_version = self._register_model_with_mlflow(
                model_path=best_model_path,
                metrics_path=best_metrics_path,
                model_name="energy-load-forecaster",
                stage="Staging",
            )
            if mlflow_version:
                logger.info(
                    "Model version %s registered in MLflow (Staging stage)",
                    mlflow_version,
                )
            
            best_model_path_out: Optional[Path] = best_model_path
            best_metrics_path_out: Optional[Path] = best_metrics_path
        else:
            best_model_path_out = best_model_path if best_model_path.exists() else None
            best_metrics_path_out = (
                best_metrics_path if best_metrics_path.exists() else None
            )

        return ModelEvaluationArtifact(
            is_model_accepted=is_accepted,
            best_model_path=best_model_path_out,
            best_metrics_path=best_metrics_path_out,
            current_model_path=current_model_path,
            current_metrics_path=current_metrics_path,
        )
