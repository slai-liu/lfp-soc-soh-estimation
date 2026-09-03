# SOC estimation: RFORC-LSTM reproduction

This module implements the reverse first-order RC (RFORC) voltage-shielding and
LSTM SOC-estimation workflow described by Chen et al. for A123 LiFePO4 cells.

## Method

1. Load the CALCE A123 DST, FUDS and US06 segments from seven temperature ZIPs.
2. Calculate reference SOC by Coulomb counting each complete discharge trajectory.
3. Identify `R0`, `R1` and `C1` by deterministic PSO on DST.
4. Train a terminal-voltage LSTM-RNN and an OCVN-based RFORC-LSTM.
5. Select checkpoints on FUDS and evaluate complete US06 trajectories.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r soc/requirements.txt
python soc/self_test.py
```

## Run

Place the seven required archives in `data/soc/` as documented in
[`../data/README.md`](../data/README.md), then run from the repository root:

```bash
python soc/run_reproduction.py --data-dir data/soc --device auto
```

The paper-aligned configuration uses two 300-epoch trainings. For a pipeline
smoke run only, an explicit override is available:

```bash
python soc/run_reproduction.py --data-dir data/soc --device cpu --epochs 1
```

Artifacts are written to `soc/results/`, which is ignored by Git.

## Optional paper comparison

The public repository intentionally does not include `paper_reference_metrics.csv`.
If a user supplies a compatible file with columns `method`, `temperature_c` and
`rmse_percent`, the paper-comparison plot is produced automatically. If the file
is absent, only that plot is skipped; data loading, training, testing and every
other plot continue normally.

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for disclosed reconstruction choices.

## Reference

J. Chen et al., “A novel state of charge estimation method for LiFePO4 battery
based on combined modeling of physical model and machine learning model,”
*Journal of Energy Storage* 115 (2025) 115888.
<https://doi.org/10.1016/j.est.2025.115888>
