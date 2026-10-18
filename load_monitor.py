"""
Load Monitoring and Analysis Module
Simulates and analyzes electricity usage data from building systems:
HVAC, Lighting, Elevators, and Other Appliances.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


BUILDING_SYSTEMS = ["HVAC", "Lighting", "Elevators", "Appliances", "Server Room"]

# Typical kW draw ranges per system
SYSTEM_PROFILES = {
    "HVAC":        {"base": 45, "peak_factor": 2.2, "off_factor": 0.3},
    "Lighting":    {"base": 20, "peak_factor": 1.5, "off_factor": 0.1},
    "Elevators":   {"base": 10, "peak_factor": 2.0, "off_factor": 0.05},
    "Appliances":  {"base": 15, "peak_factor": 1.8, "off_factor": 0.4},
    "Server Room": {"base": 25, "peak_factor": 1.2, "off_factor": 0.9},
}


def _hour_multiplier(hour: int) -> float:
    """Return a usage multiplier based on hour of day (0-23)."""
    if 0 <= hour < 6:
        return 0.25   # Night – very low
    elif 6 <= hour < 9:
        return 0.75   # Morning ramp-up
    elif 9 <= hour < 12:
        return 1.0    # Morning peak
    elif 12 <= hour < 14:
        return 0.85   # Lunch dip
    elif 14 <= hour < 18:
        return 1.0    # Afternoon peak
    elif 18 <= hour < 21:
        return 0.65   # Evening wind-down
    else:
        return 0.30   # Late night


def generate_hourly_data(hours: int = 24, seed: int = None) -> pd.DataFrame:
    """
    Generate simulated hourly energy consumption data for all building systems.

    Parameters
    ----------
    hours : int   – Number of hours to generate (default 24)
    seed  : int   – Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: timestamp, system, load_kw, hour
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    records = []

    for h in range(hours):
        ts = now - timedelta(hours=hours - h - 1)
        hour = ts.hour
        mult = _hour_multiplier(hour)

        for system, profile in SYSTEM_PROFILES.items():
            base = profile["base"]
            if mult >= 0.9:
                load = base * profile["peak_factor"]
            elif mult <= 0.3:
                load = base * profile["off_factor"]
            else:
                load = base * mult

            # Add realistic noise
            noise = np.random.normal(0, load * 0.08)
            load = max(0.5, load + noise)

            # Occasional random spike (5 % chance)
            if random.random() < 0.05:
                load *= random.uniform(1.4, 2.0)

            records.append({
                "timestamp": ts,
                "system": system,
                "load_kw": round(load, 2),
                "hour": hour,
            })

    return pd.DataFrame(records)


def get_system_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-system statistics from the hourly data.

    Returns
    -------
    pd.DataFrame with columns: system, avg_kw, max_kw, min_kw, total_kwh
    """
    summary = (
        df.groupby("system")["load_kw"]
        .agg(avg_kw="mean", max_kw="max", min_kw="min", total_kwh="sum")
        .reset_index()
    )
    summary["avg_kw"]    = summary["avg_kw"].round(2)
    summary["max_kw"]    = summary["max_kw"].round(2)
    summary["min_kw"]    = summary["min_kw"].round(2)
    summary["total_kwh"] = summary["total_kwh"].round(2)
    return summary


def get_total_load_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return hourly total building load by summing all systems.

    Returns
    -------
    pd.DataFrame with columns: timestamp, total_kw
    """
    total = (
        df.groupby("timestamp")["load_kw"]
        .sum()
        .reset_index()
        .rename(columns={"load_kw": "total_kw"})
    )
    total["total_kw"] = total["total_kw"].round(2)
    return total


def compute_load_factor(total_series: pd.DataFrame) -> float:
    """
    Load factor = average load / peak load  (0–1, higher is better).
    """
    avg  = total_series["total_kw"].mean()
    peak = total_series["total_kw"].max()
    return round(avg / peak, 4) if peak > 0 else 0.0
