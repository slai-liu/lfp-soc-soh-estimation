# LFP SOC and SOH Estimation

This repository provides compact, reproducible reference implementations for state-of-charge (SOC) and state-of-health (SOH) estimation of lithium iron phosphate (LFP) cells under dynamic operating conditions. It packages two independent research modules while deliberately excluding datasets, trained models, predictions, and experiment artifacts.

No license is included yet. The team is reviewing the appropriate release terms; until a license is added, normal copyright restrictions apply.

## Modules

### `soc/`: hybrid physical model and LSTM

The SOC workflow combines a reverse first-order RC model, particle-swarm parameter identification, feature shielding, and an LSTM estimator. Its public entry point is [`soc/run_reproduction.py`](soc/run_reproduction.py).

### `soh/`: short-pulse voltage features and Random Forest

The primary SOH route is:

```text
U1-U21 -> Random Forest -> SOH
```

It supports equal-weight fusion of predictions from multiple SOC diagnostic windows and converts SOH to currently available capacity. A conditional variational autoencoder (CVAE) feature generator is provided only as an optional auxiliary path; it is not the primary estimator. The public entry point is [`soh/run.py`](soh/run.py).

## Data are not included

This repository contains no original or processed battery measurements. Obtain the data from the applicable official or author-provided source and comply with its terms. See [`data/README.md`](data/README.md) for the exact expected filenames, schemas, and directory layout.

For the SOC module, the expected A123 archives are distributed through the [CALCE Battery Research Data page](https://calce.umd.edu/battery-data). Place the seven archives under `data/soc/` without renaming them.

For the SOH module, prepare a CSV or Excel workbook containing `battery_id`, `SOC`, `SOH`, and `U1` through `U21`, then place it under `data/soh/`. The public code does not download or reconstruct the source measurements.

## Installation

Python 3.11 is recommended. The modules have separate dependency files so the optional TensorFlow stack is not required for the primary SOH workflow.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r soc/requirements.txt
python -m pip install -r soh/requirements.txt
```

For optional CVAE generation, also install:

```bash
python -m pip install -r soh/requirements-cvae.txt
```

Run the synthetic checks before using external data:

```bash
python soc/self_test.py
python soh/self_test.py
```

## Running SOC estimation

The default configuration expects the seven A123 archives under `data/soc/` and writes newly generated artifacts to the ignored `soc/results/` directory.

```bash
python soc/run_reproduction.py \
  --data-dir data/soc \
  --config soc/config.json \
  --output-dir soc/results
```

The optional paper-comparison figure is produced only when a compatible reference CSV is explicitly available:

```bash
python soc/run_reproduction.py \
  --data-dir data/soc \
  --paper-metrics /path/to/paper_reference_metrics.csv
```

`paper_reference_metrics.csv` is intentionally not distributed. If it is absent, only that comparison plot is skipped; training, testing, and the other plots continue normally.

## Running SOH estimation

The default example uses the Case 3 SOC split: measured features at 5%, 25%, and 50% SOC train the Random Forest, which predicts at the remaining listed target windows.

```bash
python soh/run.py \
  --data data/soh/lfp_pulse_features.csv \
  --output-dir soh/results
```

To enable the optional CVAE-generated-feature route:

```bash
python soh/run.py \
  --data data/soh/lfp_pulse_features.csv \
  --output-dir soh/results-cvae \
  --use-cvae
```

The CVAE route requires TensorFlow/Keras and is explicitly auxiliary. It must use only training-cell labels and train-fitted scalers in strict battery-level evaluation.

## Scientific scope and limitations

- The repository targets LFP chemistry only. Results must not be mixed with NMC, LMO, or other chemistries.
- The included SOC code targets the documented A123 dynamic-drive-cycle data layout. Reproduction can vary with dependency versions and hardware.
- The primary SOH interface assumes already processed 21-dimensional short-pulse voltage features. It does not extract U1-U21 from raw high-rate waveforms.
- A cross-SOC split can contain measurements from the same cell in training and testing; it must not be described as generalization to completely unseen cells.
- Unknown-cell, zero-individual-calibration performance requires a battery-ID-isolated protocol. The simple public entry point is a runnable reference, not evidence of universal deployment performance.
- No claims are made here for multiple temperatures, battery packs, complete vehicle tests, raw-sensor noise, or the full LFP life cycle.
- Equal-weight multi-window fusion can reduce complementary SOC-dependent errors, but it cannot be assumed to eliminate systematic bias.
- Available capacity is reported as `rated_capacity_ah * SOH`; users must supply the correct rated capacity for their cell.

## Related publications

1. J. Chen et al., “A novel state of charge estimation method for LiFePO4 battery based on combined modeling of physical model and machine learning model,” *Journal of Energy Storage*, 115, 115888, 2025. [https://doi.org/10.1016/j.est.2025.115888](https://doi.org/10.1016/j.est.2025.115888)
2. S. Tao et al., “Generative learning assisted state-of-health estimation for sustainable battery recycling with random retirement conditions,” *Nature Communications*, 15, 10154, 2024. [https://doi.org/10.1038/s41467-024-54454-0](https://doi.org/10.1038/s41467-024-54454-0)

## Team

- 刘佳俊 (Jiajun Liu)
- 刘开畅
- 姜京浩
- 赵鲁豪
- 贾子寒
