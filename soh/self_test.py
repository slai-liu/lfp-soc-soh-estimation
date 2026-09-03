"""Data-free self-test for loading invariants, RF prediction, and fusion."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import FEATURE_COLUMNS
    from .data import read_lfp_features, validate_lfp_features
    from .run import run
except ImportError:
    from __init__ import FEATURE_COLUMNS
    from data import read_lfp_features, validate_lfp_features
    from run import run


def main() -> None:
    rng = np.random.default_rng(0); rows = []
    for battery in range(12):
        soh = 0.76 + 0.018 * battery
        for soc in (.05, .10, .25, .30, .50):
            features = 3.15 + .45 * soc + .12 * soh + rng.normal(0, .001, 21)
            rows.append({"battery_id": f"B{battery:02d}", "SOC": soc, "SOH": soh,
                         **dict(zip(FEATURE_COLUMNS, features))})
    data = pd.DataFrame(rows); validate_lfp_features(data)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic_lfp_features.csv"
        data.to_csv(path, index=False)
        data = read_lfp_features(path)
    predictions, metrics, artifacts = run(data, [5, 25, 50], [10, 30])
    assert len(predictions) == 24 and len(metrics) == 2
    assert len(artifacts["fused"]) == 12
    assert np.isfinite(predictions.prediction).all()
    assert np.allclose(artifacts["fused"].q_available_pred_ah,
                       35 * artifacts["fused"].prediction)
    assert artifacts["summary"]["main_model"] == "U1-U21 -> Random Forest -> SOH"
    assert artifacts["summary"]["cvae_optional"] is False
    print("soh self_test: ok")


if __name__ == "__main__":
    main()
