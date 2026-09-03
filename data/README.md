# Data layout

No battery data are distributed in this repository. Create the directories shown below locally; they are ignored by Git.

```text
data/
├── soc/
│   ├── A123_DST-US06-FUDS-0.zip
│   ├── A123_DST-US06-FUDS-10.zip
│   ├── A123_DST-US06-FUDS-20.zip
│   ├── A123_DST-US06-FUDS-25.zip
│   ├── A123_DST-US06-FUDS-30.zip
│   ├── A123_DST-US06-FUDS-40.zip
│   └── A123_DST-US06-FUDS-50.zip
└── soh/
    └── lfp_pulse_features.csv
```

## SOC archives

Obtain the A123 LFP dynamic-drive-cycle archives from the [CALCE Battery Research Data page](https://calce.umd.edu/battery-data) and retain the filenames exactly as listed above. The loader reads the spreadsheet files directly from each ZIP archive, so extraction is not required.

The module expects the source worksheets and columns documented by the upstream dataset. It does not silently substitute synthetic data when files are missing.

## SOH processed feature table

Prepare a CSV or Excel file with one row per battery and diagnostic SOC window. At minimum it must contain:

| Column | Meaning |
|---|---|
| `battery_id` | Stable cell identifier used to prevent cross-cell leakage |
| `SOC` | Diagnostic state of charge, either as a fraction in `[0, 1]` or percentage in `[0, 100]` |
| `SOH` | Capacity-based state of health as a fraction |
| `U1` ... `U21` | Twenty-one processed transient-voltage features, in volts |

All required values must be finite. Each `(battery_id, SOC)` pair must be unique, and SOH must be constant across SOC rows for the same battery. Additional columns are ignored.

The expected default path is:

```text
data/soh/lfp_pulse_features.csv
```

Excel input (`.xlsx`) is also accepted through the `--data` argument. These are processed features, not raw 100 Hz voltage waveforms; this release does not claim to reproduce the upstream feature-extraction stage.

Users are responsible for obtaining data under the upstream provider's terms and for verifying that redistribution or derived-data use is permitted.
