# SOC reproduction assumptions and scope

The implementation follows the equations, PSO search space, LSTM architecture,
data split, normalization and metrics described by Chen et al. (2025), DOI
`10.1016/j.est.2025.115888`.

## Specified by the paper

- CALCE A123 LiFePO4 data at 0, 10, 20, 25, 30, 40 and 50 °C.
- DST for identification/training, FUDS for validation and US06 for testing.
- Per-temperature RFORC parameters and sixth-order OCVN–SOC polynomial fitting.
- PSO with 30 particles, 50 iterations, `c1=c2=2` and inertia 0.9 to 0.4.
- One-layer, 30-unit LSTM with 50-step windows, batch size 64 and 300 epochs.
- Voltage/OCVN, current and temperature inputs; min–max scaling to `[-1, 1]`.

## Necessary disclosed choices

- Cell `A1-007` is used because its drive-cycle durations match the published axes;
  the article does not identify the cell.
- SOC is calculated by Coulomb counting and normalized by each complete trajectory's
  measured net usable capacity.
- Adam uses learning rate `1e-3`, no weight decay and seed `20250803`.
- Scaling is fitted on DST only. The best epoch is selected on FUDS within the
  configured 300-epoch budget.
- Polarization voltage starts at zero at each isolated drive-cycle segment.

Exact numerical equality is not expected because the article does not provide its
original implementation, trained weights, cell identifier or all training details.
