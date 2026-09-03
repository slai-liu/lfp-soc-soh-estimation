"""Small data-free smoke test for the SOC model chain."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from a123_data import A123Segment
from model import SocLSTM
from plot_results import make_all_figures
from rforc import RFORCParameters, shield_voltage
from training import FeatureScaler, make_windows


def main() -> None:
    samples = 120
    segment = A123Segment(
        profile="DST", ambient_temperature_c=25,
        time_s=np.arange(samples, dtype=np.float64),
        voltage_v=np.linspace(3.5, 3.0, samples, dtype=np.float32),
        current_a=np.full(samples, -1.0, dtype=np.float32),
        measured_temperature_c=np.full(samples, 25.0, dtype=np.float32),
        soc=np.linspace(1.0, 0.0, samples, dtype=np.float32),
        usable_capacity_ah=1.0, archive="synthetic.zip",
        workbook="synthetic.xlsx", source_rows=samples)
    ocvn, polarization = shield_voltage(segment, RFORCParameters(0.15, 0.03, 1000.0))
    assert ocvn.shape == polarization.shape == (samples,)
    features = np.column_stack([ocvn, segment.current_a, segment.measured_temperature_c])
    scaler = FeatureScaler.fit([features])
    windows, targets = make_windows(scaler.transform(features), segment.soc, 50)
    assert windows.shape == (samples, 50, 3) and targets.shape == (samples, 1)
    prediction = SocLSTM()(windows[:8])
    assert prediction.shape == (8, 1)
    assert torch.all((prediction >= 0.0) & (prediction <= 1.0))
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        x = np.arange(20, dtype=float)
        pd.DataFrame({"temperature_c": 25, "profile": "US06", "time_s": x,
                      "true_soc": np.linspace(1, 0, 20),
                      "lstm_rnn_soc": np.linspace(.99, .01, 20),
                      "rforc_lstm_soc": np.linspace(1, 0, 20)}).to_csv(
                          output / "predictions_us06.csv", index=False)
        pd.DataFrame({"scope": ["total", "total"], "temperature_c": [np.nan, np.nan],
                      "method": ["LSTM-RNN", "RFORC-LSTM"],
                      "rmse_percent": [1.0, .8]}).to_csv(
                          output / "metrics_us06.csv", index=False)
        pd.DataFrame({"soc": np.linspace(1, 0, 20),
                      "terminal_voltage_v": np.linspace(3.5, 3.0, 20),
                      "ocvn_v": np.linspace(3.45, 3.05, 20)}).to_csv(
                          output / "voltage_shielding_diagnostics.csv", index=False)
        for method in ("lstm_rnn", "rforc_lstm"):
            pd.DataFrame({"method": method, "epoch": [1, 2],
                          "validation_mse": [.1, .05]}).to_csv(
                              output / f"training_history_{method}.csv", index=False)
        assert make_all_figures(output, output / "paper_reference_metrics.csv") is False
        assert (output / "figures" / "soc_estimation_us06.png").is_file()
        assert not (output / "figures" / "paper_vs_reproduction.png").exists()
    print("soc self_test: ok")


if __name__ == "__main__":
    main()
