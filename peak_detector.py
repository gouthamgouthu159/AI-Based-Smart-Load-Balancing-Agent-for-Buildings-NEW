"""
Peak Load Detection Module
Identifies peak electricity demand periods and detects abnormal load spikes
using statistical methods and rolling-window analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class PeakEvent:
    timestamp: str
    total_kw: float
    z_score: float
    severity: str          # "normal" | "elevated" | "high" | "critical"
    systems_contributing: List[str] = field(default_factory=list)


def _severity_label(z: float) -> str:
    if z < 1.5:
        return "normal"
    elif z < 2.0:
        return "elevated"
    elif z < 2.5:
        return "high"
    else:
        return "critical"


def detect_peaks(
    total_series: pd.DataFrame,
    df_systems: pd.DataFrame,
    z_threshold: float = 1.5,
    rolling_window: int = 3,
) -> List[PeakEvent]:
    """
    Detect peak load periods using Z-score analysis.

    Parameters
    ----------
    total_series   : DataFrame with columns [timestamp, total_kw]
    df_systems     : Raw per-system DataFrame from load_monitor
    z_threshold    : Minimum Z-score to flag as a peak (default 1.5)
    rolling_window : Hours for rolling mean/std baseline

    Returns
    -------
    List of PeakEvent objects (sorted by total_kw descending)
    """
    ts = total_series.copy().sort_values("timestamp").reset_index(drop=True)

    # Rolling stats for contextual baseline
    ts["rolling_mean"] = (
        ts["total_kw"].rolling(window=rolling_window, min_periods=1).mean()
    )
    ts["rolling_std"] = (
        ts["total_kw"].rolling(window=rolling_window, min_periods=1).std().fillna(1)
    )

    # Global stats for Z-score
    global_mean = ts["total_kw"].mean()
    global_std  = ts["total_kw"].std() or 1.0

    ts["z_score"] = ((ts["total_kw"] - global_mean) / global_std).round(3)

    peaks: List[PeakEvent] = []
    peak_rows = ts[ts["z_score"] >= z_threshold]

    for _, row in peak_rows.iterrows():
        # Which systems contributed most at this timestamp?
        ts_val = row["timestamp"]
        slice_ = df_systems[df_systems["timestamp"] == ts_val].copy()
        if not slice_.empty:
            slice_ = slice_.sort_values("load_kw", ascending=False)
            top_systems = slice_["system"].head(3).tolist()
        else:
            top_systems = []

        peaks.append(
            PeakEvent(
                timestamp=str(ts_val),
                total_kw=round(row["total_kw"], 2),
                z_score=row["z_score"],
                severity=_severity_label(row["z_score"]),
                systems_contributing=top_systems,
            )
        )

    peaks.sort(key=lambda p: p.total_kw, reverse=True)
    return peaks


def get_peak_hours(total_series: pd.DataFrame, top_n: int = 5) -> List[dict]:
    """
    Return the top-N highest-load hours with contextual info.
    """
    ts = total_series.copy()
    ts["hour"] = pd.to_datetime(ts["timestamp"]).dt.hour
    ts_sorted = ts.sort_values("total_kw", ascending=False).head(top_n)

    result = []
    for _, row in ts_sorted.iterrows():
        result.append({
            "timestamp": str(row["timestamp"]),
            "hour": int(row["hour"]),
            "total_kw": round(row["total_kw"], 2),
            "time_label": _time_label(int(row["hour"])),
        })
    return result


def get_demand_statistics(total_series: pd.DataFrame) -> dict:
    """
    Return key demand statistics: peak, average, baseline, peak-to-average ratio.
    """
    vals = total_series["total_kw"]
    peak      = vals.max()
    average   = vals.mean()
    baseline  = vals.quantile(0.10)   # 10th percentile = baseline
    par       = peak / average if average > 0 else 1.0

    return {
        "peak_kw":            round(peak, 2),
        "average_kw":         round(average, 2),
        "baseline_kw":        round(baseline, 2),
        "peak_to_avg_ratio":  round(par, 3),
        "std_kw":             round(float(vals.std()), 2),
        "total_hours":        len(vals),
    }


def _time_label(hour: int) -> str:
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 9:
        return "Morning Ramp-Up"
    elif 9 <= hour < 12:
        return "Morning Peak"
    elif 12 <= hour < 14:
        return "Lunch Period"
    elif 14 <= hour < 18:
        return "Afternoon Peak"
    elif 18 <= hour < 21:
        return "Evening Wind-Down"
    else:
        return "Late Night"
