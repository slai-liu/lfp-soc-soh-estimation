"""Reverse first-order RC voltage shielding and deterministic PSO fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from a123_data import A123Segment


@dataclass(frozen=True)
class RFORCParameters:
    r0_ohm: float
    r1_ohm: float
    c1_f: float

    def to_dict(self) -> dict[str, float]:
        return {"r0_ohm": self.r0_ohm, "r1_ohm": self.r1_ohm, "c1_f": self.c1_f}


def polarization_voltage(current_a: np.ndarray, sample_interval_s: float,
                         r1_ohm: float, c1_f: float) -> np.ndarray:
    tau = max(r1_ohm * c1_f, 1e-9)
    decay = float(np.exp(-sample_interval_s / tau))
    return lfilter([(1.0 - decay) * r1_ohm], [1.0, -decay], current_a).astype(np.float64)


def shield_voltage(segment: A123Segment, parameters: RFORCParameters) -> tuple[np.ndarray, np.ndarray]:
    polarization = polarization_voltage(segment.current_a.astype(np.float64),
                                          segment.sample_interval_s,
                                          parameters.r1_ohm, parameters.c1_f)
    ocvn = segment.voltage_v.astype(np.float64) - parameters.r0_ohm * segment.current_a - polarization
    return ocvn.astype(np.float32), polarization.astype(np.float32)


def _swarm_objective(segment: A123Segment, positions: np.ndarray, mask: np.ndarray,
                     design: np.ndarray, pseudoinverse: np.ndarray) -> np.ndarray:
    current = segment.current_a.astype(np.float64)
    voltage = segment.voltage_v.astype(np.float64)
    values = np.empty((len(positions), int(mask.sum())), dtype=np.float64)
    for index, (r0, r1, c1) in enumerate(positions):
        tau = max(float(r1 * c1), 1e-9)
        decay = float(np.exp(-segment.sample_interval_s / tau))
        polarization = lfilter([(1.0 - decay) * float(r1)], [1.0, -decay], current)
        values[index] = (voltage - r0 * current - polarization)[mask]
    coefficients = pseudoinverse @ values.T
    residuals = values.T - design @ coefficients
    return np.mean(np.square(residuals), axis=0)


def identify_parameters(segment: A123Segment, pso_config: dict[str, object], *, seed: int
                        ) -> tuple[RFORCParameters, list[dict[str, float | int]]]:
    names = ["r0_ohm", "r1_ohm", "c1_f"]
    position_bounds = pso_config["position_bounds"]
    velocity_bounds = pso_config["velocity_bounds"]
    lower = np.asarray([position_bounds[name][0] for name in names], dtype=float)
    upper = np.asarray([position_bounds[name][1] for name in names], dtype=float)
    v_lower = np.asarray([velocity_bounds[name][0] for name in names], dtype=float)
    v_upper = np.asarray([velocity_bounds[name][1] for name in names], dtype=float)
    rng = np.random.default_rng(seed + segment.ambient_temperature_c * 101)
    positions = rng.uniform(lower, upper, (int(pso_config["particles"]), 3))
    velocities = rng.uniform(v_lower, v_upper, positions.shape)
    low, high = map(float, pso_config["soc_fit_range"])
    mask = (segment.soc >= low) & (segment.soc <= high)
    design = np.vander(segment.soc[mask].astype(float), int(pso_config["polynomial_degree"]) + 1)
    pseudoinverse = np.linalg.pinv(design)
    scores = _swarm_objective(segment, positions, mask, design, pseudoinverse)
    personal_positions, personal_scores = positions.copy(), scores.copy()
    best = int(np.argmin(scores)); global_position = positions[best].copy()
    global_score = float(scores[best]); history = []
    iterations = int(pso_config["iterations"])
    for iteration in range(1, iterations + 1):
        fraction = (iteration - 1) / max(iterations - 1, 1)
        inertia = float(pso_config["inertia_start"]) + fraction * (
            float(pso_config["inertia_end"]) - float(pso_config["inertia_start"]))
        velocities = (inertia * velocities
                      + float(pso_config["cognitive"]) * rng.random(positions.shape) * (personal_positions - positions)
                      + float(pso_config["social"]) * rng.random(positions.shape) * (global_position - positions))
        velocities = np.clip(velocities, v_lower, v_upper)
        positions = np.clip(positions + velocities, lower, upper)
        scores = _swarm_objective(segment, positions, mask, design, pseudoinverse)
        improved = scores < personal_scores
        personal_scores[improved] = scores[improved]
        personal_positions[improved] = positions[improved]
        best = int(np.argmin(personal_scores))
        if personal_scores[best] < global_score:
            global_score = float(personal_scores[best]); global_position = personal_positions[best].copy()
        history.append({"temperature_c": segment.ambient_temperature_c,
                        "iteration": iteration, "inertia": inertia,
                        "best_mse_v2": global_score})
    return RFORCParameters(*map(float, global_position)), history
