"""Deterministic preprocessing, training, and inference for SOC LSTMs."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from a123_data import A123Segment
from model import SocLSTM


@dataclass
class FeatureScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, features: list[np.ndarray]) -> "FeatureScaler":
        combined = np.concatenate(features, axis=0)
        return cls(combined.min(axis=0), combined.max(axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        denominator = np.maximum(self.maximum - self.minimum, 1e-8)
        return (2.0 * (values - self.minimum) / denominator - 1.0).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {"feature_order": ["voltage_or_ocvn_v", "current_a", "temperature_c"],
                "minimum": self.minimum.tolist(), "maximum": self.maximum.tolist(),
                "target_range": [-1.0, 1.0], "fit_scope": "DST training data only"}


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def raw_features(segment: A123Segment) -> np.ndarray:
    return np.column_stack([segment.voltage_v, segment.current_a,
                            segment.measured_temperature_c]).astype(np.float32)


def hybrid_features(segment: A123Segment, ocvn_v: np.ndarray) -> np.ndarray:
    return np.column_stack([ocvn_v, segment.current_a,
                            segment.measured_temperature_c]).astype(np.float32)


def make_windows(features: np.ndarray, soc: np.ndarray, time_steps: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = np.repeat(features[:1], time_steps - 1, axis=0)
    padded = np.concatenate([prefix, features], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=time_steps, axis=0).transpose(0, 2, 1)
    return (torch.from_numpy(windows.copy()),
            torch.from_numpy(soc.reshape(-1, 1).astype(np.float32, copy=True)))


def _tensors(segments: list[A123Segment], lookup: dict[tuple[str, int], np.ndarray],
             scaler: FeatureScaler, time_steps: int, device: torch.device
             ) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [make_windows(scaler.transform(lookup[(s.profile, s.ambient_temperature_c)]),
                          s.soc, time_steps) for s in segments]
    return torch.cat([x for x, _ in pairs]).to(device), torch.cat([y for _, y in pairs]).to(device)


def _mse(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
         batch_size: int) -> float:
    model.eval(); total = 0.0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            total += float(nn.functional.mse_loss(model(inputs[start:start+batch_size]),
                                                   targets[start:start+batch_size], reduction="sum"))
    return total / len(inputs)


def train_model(*, method: str, train_segments: list[A123Segment],
                validation_segments: list[A123Segment],
                feature_lookup: dict[tuple[str, int], np.ndarray],
                config: dict[str, object], device: torch.device,
                output_dir: Path) -> tuple[SocLSTM, FeatureScaler, pd.DataFrame, dict[str, object]]:
    cfg = dict(config["lstm"]); seed = int(config["seed"]); set_reproducible_seed(seed)
    scaler = FeatureScaler.fit([feature_lookup[(s.profile, s.ambient_temperature_c)]
                                for s in train_segments])
    train_x, train_y = _tensors(train_segments, feature_lookup, scaler,
                                int(cfg["time_steps"]), device)
    val_x, val_y = _tensors(validation_segments, feature_lookup, scaler,
                            int(cfg["time_steps"]), device)
    model = SocLSTM(int(cfg["input_features"]), int(cfg["hidden_size"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]),
                                 weight_decay=float(cfg["weight_decay"]))
    criterion = nn.MSELoss(); best_mse = float("inf"); best_epoch = 0; best_state = None
    history = []; generator = torch.Generator(device=device.type).manual_seed(seed)
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train(); total = 0.0
        order = torch.randperm(len(train_x), device=device, generator=generator)
        for start in range(0, len(train_x), batch_size):
            index = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x.index_select(0, index)), train_y.index_select(0, index))
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(index)
        train_mse = total / len(train_x)
        val_mse = _mse(model, val_x, val_y, int(cfg["evaluation_batch_size"]))
        history.append({"method": method, "epoch": epoch,
                        "training_mse": train_mse, "validation_mse": val_mse})
        if val_mse < best_mse:
            best_mse, best_epoch, best_state = val_mse, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(f"{method} epoch {epoch}/{cfg['epochs']} train={train_mse:.7f} val={val_mse:.7f}")
    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best_state); output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history); frame.to_csv(output_dir / f"training_history_{method}.csv", index=False)
    checkpoint = {"method": method, "state_dict": best_state, "best_epoch": best_epoch,
                  "best_validation_mse": best_mse, "scaler": scaler.to_dict(),
                  "lstm_config": cfg, "seed": seed}
    torch.save(checkpoint, output_dir / f"best_model_{method}.pt")
    return model, scaler, frame, checkpoint


def predict(model: SocLSTM, segment: A123Segment, features: np.ndarray,
            scaler: FeatureScaler, lstm_config: dict[str, object],
            device: torch.device) -> np.ndarray:
    inputs, targets = make_windows(scaler.transform(features), segment.soc,
                                   int(lstm_config["time_steps"]))
    loader = DataLoader(TensorDataset(inputs, targets),
                        batch_size=int(lstm_config["evaluation_batch_size"]), shuffle=False)
    outputs = []; model.eval()
    with torch.inference_mode():
        for batch, _ in loader:
            outputs.append(model(batch.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(outputs)
