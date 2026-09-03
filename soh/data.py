"""Read and validate processed LFP U1-U21 pulse-feature tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import FEATURE_COLUMNS
except ImportError:  # direct script execution
    from __init__ import FEATURE_COLUMNS


def read_lfp_features(path: Path, *, sheet_name: str = "SOC ALL") -> pd.DataFrame:
    """Return canonical columns battery_id, SOC, SOH and U1-U21.

    CSV and XLSX inputs are supported. SOC is normalized to a fraction in [0, 1]
    when the source uses percent values. The file itself is never modified.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xlsm"}:
        data = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError("Expected a .csv, .xlsx, or .xlsm processed feature file")
    aliases = {}
    if "battery_id" not in data and "ID" in data:
        aliases["ID"] = "battery_id"
    if "SOH" not in data and "soh" in data:
        aliases["soh"] = "SOH"
    if "SOC" not in data and "soc" in data:
        aliases["soc"] = "SOC"
    data = data.rename(columns=aliases)
    required = ["battery_id", "SOC", "SOH", *FEATURE_COLUMNS]
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    result = data.loc[:, required].copy()
    result["battery_id"] = result.battery_id.astype(str)
    numeric = ["SOC", "SOH", *FEATURE_COLUMNS]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if float(result.SOC.max()) > 1.0:
        result["SOC"] = result.SOC / 100.0
    validate_lfp_features(result)
    return result.sort_values(["battery_id", "SOC"]).reset_index(drop=True)


def validate_lfp_features(data: pd.DataFrame) -> None:
    required = ["battery_id", "SOC", "SOH", *FEATURE_COLUMNS]
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if not np.isfinite(data[["SOC", "SOH", *FEATURE_COLUMNS]].to_numpy(float)).all():
        raise ValueError("Feature table contains NaN or infinity")
    if not data.SOC.between(0, 1).all() or not data.SOH.between(0, 1.2).all():
        raise ValueError("SOC/SOH values are outside expected fractional ranges")
    if data.duplicated(["battery_id", "SOC"]).any():
        raise ValueError("Duplicate battery_id + SOC rows")
    if not data.groupby("battery_id").SOH.nunique().eq(1).all():
        raise ValueError("SOH must be constant across SOC for each battery_id")
