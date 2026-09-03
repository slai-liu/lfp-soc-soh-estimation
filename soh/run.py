"""Single entry point for LFP short-pulse SOH estimation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import FEATURE_COLUMNS
    from .data import read_lfp_features
    from .fusion import equal_weight_fusion
    from .rf import fit_random_forest, predict_soh, regression_metrics
except ImportError:  # direct script execution
    from __init__ import FEATURE_COLUMNS
    from data import read_lfp_features
    from fusion import equal_weight_fusion
    from rf import fit_random_forest, predict_soh, regression_metrics


def as_fraction(values: list[float]) -> list[float]:
    return [value / 100.0 if value > 1 else value for value in values]


def run(data: pd.DataFrame, train_soc: list[float], target_soc: list[float], *,
        seed: int = 0, rated_capacity_ah: float = 35.0,
        use_cvae: bool = False, cvae_epochs: int = 50) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_soc, target_soc = as_fraction(train_soc), as_fraction(target_soc)
    if set(train_soc) & set(target_soc):
        raise ValueError("Training and target SOC sets must be disjoint")
    train = data[np.isclose(data.SOC.to_numpy()[:, None], train_soc).any(axis=1)].copy()
    if train.empty:
        raise ValueError("No rows match the requested training SOC values")
    prediction_frames = []; per_soc_metrics = []
    for target in target_soc:
        test = data[np.isclose(data.SOC, target)].copy()
        if test.empty:
            raise ValueError(f"No rows match target SOC {target:g}")
        if use_cvae:
            try:
                from .cvae import generate_auxiliary_features
            except ImportError:
                from cvae import generate_auxiliary_features
            rf_train = generate_auxiliary_features(train, target, epochs=cvae_epochs, seed=seed)
            method = "optional_cvae_generated_rf"
        else:
            rf_train = train
            method = "u1_u21_random_forest"
        model = fit_random_forest(rf_train[FEATURE_COLUMNS], rf_train.SOH, seed=seed)
        prediction = predict_soh(model, test[FEATURE_COLUMNS])
        frame = pd.DataFrame({"battery_id": test.battery_id.to_numpy(),
                              "target_soc": target,
                              "true_soh": test.SOH.to_numpy(float),
                              "prediction": prediction, "method": method})
        frame["q_available_true_ah"] = rated_capacity_ah * frame.true_soh
        frame["q_available_pred_ah"] = rated_capacity_ah * frame.prediction
        frame["capacity_error_ah"] = frame.q_available_pred_ah - frame.q_available_true_ah
        prediction_frames.append(frame)
        per_soc_metrics.append({"target_soc": target, "method": method,
                                **regression_metrics(frame.true_soh, frame.prediction)})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fused = equal_weight_fusion(predictions, rated_capacity_ah=rated_capacity_ah)
    summary = {"method": predictions.method.iloc[0],
               "main_model": "U1-U21 -> Random Forest -> SOH",
               "cvae_optional": bool(use_cvae),
               "train_soc": train_soc, "target_soc": target_soc,
               "single_window_pooled": regression_metrics(predictions.true_soh, predictions.prediction),
               "equal_weight_fusion": regression_metrics(fused.true_soh, fused.prediction)}
    return predictions, pd.DataFrame(per_soc_metrics), {"summary": summary, "fused": fused}


def main() -> int:
    parser = argparse.ArgumentParser(description="LFP U1-U21 Random Forest SOH estimator")
    parser.add_argument("--data", type=Path, required=True, help="Processed LFP CSV or XLSX")
    parser.add_argument("--sheet", default="SOC ALL", help="XLSX worksheet name")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--train-soc", type=float, nargs="+", default=[5, 25, 50])
    parser.add_argument("--target-soc", type=float, nargs="+", default=[10, 15, 20, 30, 35, 40, 45])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rated-capacity-ah", type=float, default=35.0)
    parser.add_argument("--use-cvae", action="store_true", help="Enable optional CVAE-generated RF training data")
    parser.add_argument("--cvae-epochs", type=int, default=50)
    args = parser.parse_args()
    data = read_lfp_features(args.data, sheet_name=args.sheet)
    predictions, metrics, artifacts = run(data, args.train_soc, args.target_soc,
        seed=args.seed, rated_capacity_ah=args.rated_capacity_ah,
        use_cvae=args.use_cvae, cvae_epochs=args.cvae_epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_by_soc.csv", index=False)
    artifacts["fused"].to_csv(args.output_dir / "fused_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(artifacts["summary"], indent=2))
    print(json.dumps(artifacts["summary"], indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
