"""Primary U1-U21 -> Random Forest -> SOH model."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from . import FEATURE_COLUMNS
except ImportError:  # direct script execution
    from __init__ import FEATURE_COLUMNS


def build_random_forest(seed: int = 0) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=20, max_depth=64,
                                 min_samples_leaf=1, bootstrap=False,
                                 random_state=seed, n_jobs=1)


def fit_random_forest(features, soh, *, seed: int = 0) -> RandomForestRegressor:
    x = np.asarray(features, dtype=float); y = np.asarray(soh, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_COLUMNS) or len(x) != len(y):
        raise ValueError("RF expects aligned N×21 features and N SOH labels")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("RF input contains NaN or infinity")
    return build_random_forest(seed).fit(x, y)


def predict_soh(model: RandomForestRegressor, features) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("RF prediction expects N×21 features")
    prediction = model.predict(x)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite SOH prediction")
    return prediction


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true; denominator = np.sum((y_true - y_true.mean()) ** 2)
    return {"mae_pp": float(100 * np.mean(np.abs(error))),
            "rmse_pp": float(100 * np.sqrt(np.mean(error**2))),
            "mape_percent": float(100 * np.mean(np.abs(error / y_true))),
            "maxae_pp": float(100 * np.max(np.abs(error))),
            "bias_pp": float(100 * np.mean(error)),
            "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else float("nan")}
