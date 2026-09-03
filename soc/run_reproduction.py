"""Single command for the RFORC-LSTM SOC reproduction."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from a123_data import load_segments, select_profile, validate_segments, write_data_audit
from plot_results import make_all_figures
from rforc import identify_parameters, shield_voltage
from training import hybrid_features, predict, raw_features, train_model


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = 100 * (predicted.astype(float) - actual.astype(float))
    return {"rmse_percent": float(np.sqrt(np.mean(error**2))),
            "maxe_percent": float(np.max(np.abs(error))),
            "mae_percent": float(np.mean(np.abs(error))),
            "bias_percent": float(np.mean(error))}


def feature_lookups(segments, parameters):
    raw, hybrid, diagnostics = {}, {}, []
    for segment in segments:
        key = (segment.profile, segment.ambient_temperature_c)
        ocvn, polarization = shield_voltage(segment, parameters[segment.ambient_temperature_c])
        raw[key] = raw_features(segment); hybrid[key] = hybrid_features(segment, ocvn)
        if segment.profile == "DST":
            diagnostics.append(pd.DataFrame({"temperature_c": segment.ambient_temperature_c,
                "time_s": segment.time_s, "soc": segment.soc,
                "terminal_voltage_v": segment.voltage_v,
                "polarization_voltage_v": polarization, "ocvn_v": ocvn}))
    return raw, hybrid, pd.concat(diagnostics, ignore_index=True)


def main() -> int:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="RFORC-LSTM A123 SOC reproduction")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=project / "results")
    parser.add_argument("--config", type=Path, default=project / "config.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, help="Debug override of the configured epoch count")
    parser.add_argument("--paper-metrics", type=Path,
                        default=project / "paper_reference_metrics.csv",
                        help="Optional paper metric CSV; absent files are skipped")
    args = parser.parse_args(); started = time.perf_counter()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        if args.epochs < 1: raise ValueError("--epochs must be positive")
        config["lstm"]["epochs"] = args.epochs
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    segments = load_segments(args.data_dir.resolve(), config); validate_segments(segments, config)
    write_data_audit(segments, output / "data_audit.csv")
    train = select_profile(segments, config["train_profile"])
    validation = select_profile(segments, config["validation_profile"])
    test = select_profile(segments, config["test_profile"])
    parameters, pso_rows = {}, []
    for segment in train:
        parameter, history = identify_parameters(segment, config["pso"], seed=int(config["seed"]))
        parameters[segment.ambient_temperature_c] = parameter; pso_rows.extend(history)
    pd.DataFrame(pso_rows).to_csv(output / "pso_history.csv", index=False)
    pd.DataFrame([{"temperature_c": t, **p.to_dict()} for t, p in parameters.items()]).to_csv(
        output / "rforc_parameters.csv", index=False)
    raw, hybrid, diagnostics = feature_lookups(segments, parameters)
    diagnostics.to_csv(output / "voltage_shielding_diagnostics.csv", index=False)

    models, scalers = {}, {}
    for name, lookup in (("lstm_rnn", raw), ("rforc_lstm", hybrid)):
        model, scaler, _, _ = train_model(method=name, train_segments=train,
            validation_segments=validation, feature_lookup=lookup, config=config,
            device=device, output_dir=output)
        models[name], scalers[name] = model, scaler

    predictions, metric_rows = [], []
    for segment in test:
        key = (segment.profile, segment.ambient_temperature_c)
        values = {name: predict(models[name], segment, lookup[key], scalers[name],
                                config["lstm"], device)
                  for name, lookup in (("lstm_rnn", raw), ("rforc_lstm", hybrid))}
        predictions.append(pd.DataFrame({"temperature_c": segment.ambient_temperature_c,
            "profile": segment.profile, "time_s": segment.time_s, "true_soc": segment.soc,
            "lstm_rnn_soc": values["lstm_rnn"], "rforc_lstm_soc": values["rforc_lstm"]}))
        for name, display in (("lstm_rnn", "LSTM-RNN"), ("rforc_lstm", "RFORC-LSTM")):
            metric_rows.append({"scope": "temperature", "temperature_c": segment.ambient_temperature_c,
                                "method": display, "samples": len(segment.soc),
                                **metrics(segment.soc, values[name])})
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(output / "predictions_us06.csv", index=False)
    for column, display in (("lstm_rnn_soc", "LSTM-RNN"), ("rforc_lstm_soc", "RFORC-LSTM")):
        metric_rows.append({"scope": "total", "temperature_c": np.nan, "method": display,
                            "samples": len(prediction_frame),
                            **metrics(prediction_frame.true_soc, prediction_frame[column])})
    pd.DataFrame(metric_rows).to_csv(output / "metrics_us06.csv", index=False)
    paper_plot = make_all_figures(output, args.paper_metrics)
    summary = {"status": "completed", "device": str(device),
               "paper_comparison_generated": paper_plot,
               "runtime_seconds": time.perf_counter() - started}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
