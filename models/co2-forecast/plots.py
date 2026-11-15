"""Charts for the results/ folder (called by train.py)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"actual": "#1f1f1f", "harmonic_trend": "#2a6fdb",
          "seasonal_naive_drift": "#e07b00", "seasonal_naive": "#9a9a9a"}
LABELS = {"harmonic_trend": "Harmonic trend (learned)",
          "seasonal_naive_drift": "Same month last year + drift",
          "seasonal_naive": "Same month last year"}


def _ts(months) -> pd.DatetimeIndex:
    return pd.PeriodIndex(months, freq="M").to_timestamp()


def make_plots(y: pd.Series, tp: pd.DataFrame, fc: pd.DataFrame, report: dict, out: str) -> None:
    # 1) test window: actual vs each model
    fig, ax = plt.subplots(figsize=(9, 4.8))
    hist = y[y.index >= pd.Period("2019-01", "M")]
    hist = hist[hist.index <= pd.Period(report["split"]["validation"][1], "M")]
    ax.plot(hist.index.to_timestamp(), hist.values, color=COLORS["actual"], lw=1.2, alpha=.5)
    t = _ts(tp["month"])
    ax.fill_between(t, tp["harmonic_trend_lo90"], tp["harmonic_trend_hi90"],
                    color=COLORS["harmonic_trend"], alpha=.15, label="90% interval (learned)")
    for n in ["seasonal_naive", "seasonal_naive_drift", "harmonic_trend"]:
        mae = report["test"][n]["mae"]
        ax.plot(t, tp[n], color=COLORS[n], lw=1.6, label=f"{LABELS[n]}  (MAE {mae:.2f})")
    ax.plot(t, tp["actual"], color=COLORS["actual"], lw=2, label="Actual (NOAA)")
    ax.axvline(t[0], color="#666", ls=":", lw=1)
    ax.set_title("Test: one forecast made at end of 2022 vs what was measured")
    ax.set_ylabel("CO2 at Mauna Loa (ppm)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "test_forecast.png"), dpi=130)
    plt.close(fig)

    # 2) error by horizon year
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    for i, n in enumerate(["seasonal_naive", "seasonal_naive_drift", "harmonic_trend"]):
        rows = report["test_mae_by_horizon_year"][n]
        xs = [r["horizon_year"] + (i - 1) * width for r in rows]
        ax.bar(xs, [r["mae"] for r in rows], width, color=COLORS[n], label=LABELS[n])
    rows = report["test_mae_by_horizon_year"]["harmonic_trend"]
    ax.set_xticks([r["horizon_year"] for r in rows],
                  [f"Year {r['horizon_year']}\n({r['months']} months)" for r in rows])
    ax.set_xlabel("Months after the forecast date (end of 2022)")
    ax.set_ylabel("Mean absolute error (ppm)")
    ax.set_title("Test error by forecast year")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_by_horizon.png"), dpi=130)
    plt.close(fig)

    # 3) next 24 months
    fig, ax = plt.subplots(figsize=(9, 4.8))
    recent = y[y.index >= pd.Period("2020-01", "M")]
    ax.plot(recent.index.to_timestamp(), recent.values, color=COLORS["actual"], lw=1.8,
            label="Measured (NOAA)")
    t = _ts(fc["month"])
    ax.fill_between(t, fc["lo90"], fc["hi90"], color=COLORS["harmonic_trend"], alpha=.18,
                    label="90% interval")
    ax.plot(t, fc["forecast"], color=COLORS["harmonic_trend"], lw=2, ls="--", label="Forecast")
    ax.set_title(f"Next {len(fc)} months at Mauna Loa (forecast, not a measurement)")
    ax.set_ylabel("CO2 (ppm)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forecast_next24.png"), dpi=130)
    plt.close(fig)
