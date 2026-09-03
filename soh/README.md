# SOH estimation from short dynamic pulses

The primary estimator is explicitly:

```text
U1-U21 -> Random Forest -> SOH
```

It reads processed LFP pulse features, trains a deterministic Random Forest,
predicts SOH at target SOC windows, computes current available capacity, and
performs equal-weight fusion across multiple SOC diagnoses.

## Install and self-test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r soh/requirements.txt
python soh/self_test.py
```

## Run the primary RF path

Place a processed table at `data/soh/lfp_pulse_features.csv` as documented in
[`../data/README.md`](../data/README.md), then run:

```bash
python soh/run.py --data data/soh/lfp_pulse_features.csv \
  --train-soc 5 25 50 --target-soc 10 15 20 30 35 40 45
```

Outputs are written under `soh/results/` and are ignored by Git. Available
capacity uses `Q_available = rated_capacity_ah × SOH`; the default rated capacity
is 35 Ah and can be changed with `--rated-capacity-ah`.

## Optional CVAE auxiliary generation

CVAE is **optional** and is not part of the default import or execution path.
Install its additional dependencies and enable it explicitly:

```bash
pip install -r soh/requirements-cvae.txt
python soh/run.py --data data/soh/lfp_pulse_features.csv --use-cvae
```

The CVAE fits feature and condition scalers only on supplied training rows and
uses training-battery SOH labels to create target-SOC features. It does not change
the scientific status of the Random Forest as the primary online estimator.

## Boundary

The default SOC split reproduces a cross-SOC setting in which battery identities
may occur in both training and target SOC rows. It is not proof of zero-calibration
generalization to completely unseen batteries. Temperature, pack-level and
vehicle-level validation are not included in this public code package.

## Reference

S. Tao et al., “Generative learning assisted state-of-health estimation for
sustainable battery recycling with random retirement conditions,” *Nature
Communications* 15, 10154 (2024).
<https://doi.org/10.1038/s41467-024-54454-0>
