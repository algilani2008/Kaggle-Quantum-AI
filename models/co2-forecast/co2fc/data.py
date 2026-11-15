"""Download and parse the NOAA Mauna Loa monthly CO2 record.

The CSV is the mirror kept by the Frictionless Data "datasets" organisation on
GitHub (https://github.com/datasets/co2-ppm, ODC-PDDL-1.0), which copies the
NOAA Global Monitoring Laboratory file ``co2_mm_mlo.txt``.

Note: the mirror's header row has fewer names than the data has columns. The
columns are read by position, following NOAA's own layout:

    0 month (YYYY-MM)   1 decimal date   2 monthly mean (ppm)
    3 de-seasonalised mean (ppm)   4 number of measurement days
    5 std. dev. of days   6 uncertainty of monthly mean

A ``days`` value below 0 means NOAA filled that month by interpolation.
"""
from __future__ import annotations

import io
import os
import urllib.request

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasets/co2-ppm/main/data/co2-mm-mlo.csv"
COLUMNS = ["month", "decimal_date", "co2", "co2_deseason", "days", "sdev", "unc"]


def parse_csv(text: str) -> pd.DataFrame:
    """Parse the mirror CSV text into a clean monthly frame.

    Returns a DataFrame indexed by a monthly ``PeriodIndex`` with columns
    ``co2``, ``co2_deseason`` and ``interpolated`` (bool).
    """
    raw = pd.read_csv(io.StringIO(text), header=None, skiprows=1, names=COLUMNS,
                      usecols=range(len(COLUMNS)))
    raw["month"] = pd.PeriodIndex(raw["month"].astype(str), freq="M")
    raw = raw.set_index("month").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw[(raw["co2"] > 0)]  # NOAA uses -99.99 for missing in older files
    out = pd.DataFrame({
        "co2": raw["co2"].astype(float),
        "co2_deseason": raw["co2_deseason"].astype(float),
        "interpolated": raw["days"].astype(float) < 0,
    })
    full = pd.period_range(out.index.min(), out.index.max(), freq="M")
    if len(full) != len(out):
        raise ValueError(f"series has gaps: {len(full) - len(out)} missing months")
    return out


def load(path: str | None = None, cache: str | None = None) -> pd.DataFrame:
    """Load the series from ``path`` if given, else download it.

    If ``cache`` is a file path, the downloaded text is saved there and reused.
    """
    if path:
        with open(path, encoding="utf-8") as fh:
            return parse_csv(fh.read())
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return parse_csv(fh.read())
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        text = resp.read().decode("utf-8")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            fh.write(text)
    return parse_csv(text)
