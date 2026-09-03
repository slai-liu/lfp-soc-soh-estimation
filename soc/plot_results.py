"""Result plots for the SOC reproduction; paper comparison is optional."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"truth": "#222222", "baseline": "#4C78A8", "hybrid": "#E45756",
          "paper": "#888888", "ocvn": "#2A9D8F"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def plot_soc_trajectories(predictions: pd.DataFrame, output_dir: Path) -> None:
    temperatures = sorted(predictions.temperature_c.astype(int).unique())
    fig, axes = plt.subplots(len(temperatures), 1, figsize=(8, 1.8 * len(temperatures)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, temperature in zip(axes, temperatures):
        data = predictions[predictions.temperature_c == temperature]
        ax.plot(data.time_s, 100 * data.true_soc, color=COLORS["truth"], label="True")
        ax.plot(data.time_s, 100 * data.lstm_rnn_soc, color=COLORS["baseline"], label="LSTM-RNN")
        ax.plot(data.time_s, 100 * data.rforc_lstm_soc, color=COLORS["hybrid"], label="RFORC-LSTM")
        ax.set(ylabel="SOC (%)", title=f"US06, {temperature} °C")
    axes[-1].set_xlabel("Time (s)"); axes[0].legend(ncol=3)
    _save(fig, output_dir / "soc_estimation_us06")


def plot_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> None:
    total = metrics[metrics.scope == "total"]
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(total)); ax.bar(x, total.rmse_percent,
        color=[COLORS["baseline"], COLORS["hybrid"]])
    ax.set_xticks(x, total.method); ax.set_ylabel("RMSE (%)")
    _save(fig, output_dir / "error_comparison")


def plot_paper_comparison(metrics: pd.DataFrame, paper: pd.DataFrame, output_dir: Path) -> None:
    observed = metrics[metrics.scope == "temperature"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, method in zip(axes, ("LSTM-RNN", "RFORC-LSTM")):
        a = observed[observed.method == method].sort_values("temperature_c")
        b = paper[paper.method == method].sort_values("temperature_c")
        ax.plot(b.temperature_c, b.rmse_percent, "o-", color=COLORS["paper"], label="Paper")
        ax.plot(a.temperature_c, a.rmse_percent, "s-",
                color=COLORS["baseline"] if method == "LSTM-RNN" else COLORS["hybrid"],
                label="Reproduction")
        ax.set(title=method, xlabel="Temperature (°C)", ylabel="RMSE (%)"); ax.legend()
    _save(fig, output_dir / "paper_vs_reproduction")


def plot_voltage_shielding(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sample = diagnostics.iloc[::20]
    ax.scatter(100 * sample.soc, sample.terminal_voltage_v, s=3, alpha=.25, label="Terminal voltage")
    ax.scatter(100 * sample.soc, sample.ocvn_v, s=3, alpha=.4, color=COLORS["ocvn"], label="OCVN")
    ax.set(xlabel="SOC (%)", ylabel="Voltage (V)"); ax.legend()
    _save(fig, output_dir / "voltage_shielding")


def plot_training_history(history: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    for method, data in history.groupby("method"):
        ax.plot(data.epoch, data.validation_mse, label=method)
    ax.set(xlabel="Epoch", ylabel="Validation MSE", yscale="log"); ax.legend()
    _save(fig, output_dir / "training_convergence")


def make_all_figures(results_dir: Path, paper_metrics_path: Path | None = None) -> bool:
    """Build core plots and return whether the optional paper plot was built."""
    output_dir = results_dir / "figures"
    predictions = pd.read_csv(results_dir / "predictions_us06.csv")
    metrics = pd.read_csv(results_dir / "metrics_us06.csv")
    diagnostics = pd.read_csv(results_dir / "voltage_shielding_diagnostics.csv")
    history = pd.concat([pd.read_csv(results_dir / "training_history_lstm_rnn.csv"),
                         pd.read_csv(results_dir / "training_history_rforc_lstm.csv")], ignore_index=True)
    plot_soc_trajectories(predictions, output_dir)
    plot_metric_bars(metrics, output_dir)
    plot_voltage_shielding(diagnostics, output_dir)
    plot_training_history(history, output_dir)
    if paper_metrics_path is not None and paper_metrics_path.is_file():
        plot_paper_comparison(metrics, pd.read_csv(paper_metrics_path), output_dir)
        return True
    print("paper_reference_metrics.csv not found; skipping optional paper-comparison plot")
    return False
