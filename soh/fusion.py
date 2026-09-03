"""Equal-weight fusion of repeated SOH diagnoses across SOC windows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_fusion(predictions: pd.DataFrame, *, rated_capacity_ah: float = 35.0
                        ) -> pd.DataFrame:
    required = {"battery_id", "target_soc", "prediction", "true_soh"}
    if not required <= set(predictions):
        raise KeyError(f"Missing fusion columns: {sorted(required-set(predictions))}")
    if predictions.duplicated(["battery_id", "target_soc"]).any():
        raise ValueError("Duplicate battery_id + target_soc predictions")
    if not predictions.groupby("battery_id").true_soh.nunique().eq(1).all():
        raise ValueError("true_soh must be constant per battery")
    fused = predictions.groupby("battery_id", sort=True).agg(
        true_soh=("true_soh", "first"), prediction=("prediction", "mean"),
        window_count=("target_soc", "nunique")).reset_index()
    fused["error_pp"] = 100 * (fused.prediction - fused.true_soh)
    fused["q_available_true_ah"] = rated_capacity_ah * fused.true_soh
    fused["q_available_pred_ah"] = rated_capacity_ah * fused.prediction
    fused["capacity_error_ah"] = fused.q_available_pred_ah - fused.q_available_true_ah
    if not np.isfinite(fused.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError("Non-finite fused result")
    return fused
