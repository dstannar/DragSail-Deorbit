# geomagnetic predictions

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
import numpy as np
import requests


Scenario = Literal["quiet", "nominal", "active"]


@dataclass
class SpaceWeatherConfig:
    scenario: Scenario = "nominal"
    timeout_s: float = 30.0


def _get_json(url: str, timeout_s: float):
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def fetch_noaa_45day_forecast(timeout_s: float = 30.0) -> pd.DataFrame:
    """
    Daily UTC DataFrame indexed by date with columns:
      - ap
      - f107
    """
    url = "https://services.swpc.noaa.gov/json/45-day-forecast.json"
    payload = _get_json(url, timeout_s)

    if "data" not in payload:
        raise ValueError("NOAA 45-day forecast JSON missing 'data' field.")

    df = pd.DataFrame(payload["data"])

    required = {"time", "metric", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected NOAA 45-day forecast JSON columns: {list(df.columns)}"
        )

    df["time"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    df["metric"] = df["metric"].astype(str).str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    out = (
        df.pivot_table(index="time", columns="metric", values="value", aggfunc="first")
        .rename_axis(columns=None)
        .sort_index()
    )

    missing = {"ap", "f107"} - set(out.columns)
    if missing:
        raise ValueError(f"NOAA 45-day forecast missing columns: {missing}")

    return out[["ap", "f107"]].astype(float)


def fetch_noaa_predicted_solar_cycle(timeout_s: float = 30.0) -> pd.DataFrame:
    """
    Returns a monthly UTC dataframe indexed by month-start date with columns:
      - predicted_ssn
      - predicted_f107

    Source: NOAA SWPC predicted solar cycle JSON.
    """
    url = "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json"
    data = _get_json(url, timeout_s=timeout_s)

    records = []
    for row in data:
        # Current NOAA field names include "time-tag" and predicted F10.7 values.
        t = row.get("time-tag")
        if t is None:
            continue

        # Prefer central forecast if available, otherwise fall back to any predicted f10.7 field.
        f107 = (
            row.get("predicted_f10.7")
            or row.get("predicted_f107")
            or row.get("f10.7")
            or row.get("f107")
        )
        if f107 is None:
            # Some datasets expose percentile bands but not a single central value.
            # Use midpoint of high/low central bands if needed.
            hi = row.get("high_f10.7") or row.get("high_f107")
            lo = row.get("low_f10.7") or row.get("low_f107")
            if hi is not None and lo is not None:
                f107 = 0.5 * (float(hi) + float(lo))
            else:
                continue

        ssn = row.get("predicted_ssn")
        ts = pd.to_datetime(t, utc=True).normalize()
        records.append(
            {
                "date": ts,
                "predicted_ssn": float(ssn) if ssn is not None else np.nan,
                "predicted_f107": float(f107),
            }
        )

    if not records:
        raise ValueError("Could not parse NOAA predicted solar cycle JSON.")

    return pd.DataFrame.from_records(records).set_index("date").sort_index()


def ap_scenario_value(scenario: Scenario) -> float:
    """
    Scenario-based daily Ap values beyond the official short-term forecast horizon.
    """
    if scenario == "quiet":
        return 4.0
    if scenario == "active":
        return 15.0
    return 8.0  # nominal


def build_space_weather_profile(
    start: str | pd.Timestamp,
    years: int = 5,
    config: Optional[SpaceWeatherConfig] = None,
) -> pd.DataFrame:
    """
    Build a daily space weather profile out to `years`.

    Output columns:
      - ap
      - f107
      - f107a

    Strategy:
      - Use NOAA 45-day forecast for ap and f107 where available
      - Extend f107 using NOAA predicted solar cycle monthly values
      - Extend ap using scenario assumption
      - Compute f107a as 81-day rolling mean
    """
    if config is None:
        config = SpaceWeatherConfig()

    start_ts = pd.to_datetime(start, utc=True).normalize()
    end_ts = start_ts + pd.DateOffset(years=years)
    daily_index = pd.date_range(start=start_ts, end=end_ts, freq="D", tz="UTC")

    df = pd.DataFrame(index=daily_index, data={"ap": np.nan, "f107": np.nan})

    forecast_45 = fetch_noaa_45day_forecast(timeout_s=config.timeout_s)
    solar_cycle = fetch_noaa_predicted_solar_cycle(timeout_s=config.timeout_s)

    # Fill short-range forecast directly
    overlap = df.index.intersection(forecast_45.index)
    df.loc[overlap, "ap"] = forecast_45.loc[overlap, "ap"]
    df.loc[overlap, "f107"] = forecast_45.loc[overlap, "f107"]

    # Extend F10.7 from monthly NOAA predicted solar-cycle values
    monthly = solar_cycle[["predicted_f107"]].copy()

    # Reindex monthly prediction over the daily range and time interpolate
    daily_f107_ext = (
        monthly.reindex(monthly.index.union(df.index))
        .sort_index()
        .interpolate(method="time")
        .reindex(df.index)
        .rename(columns={"predicted_f107": "f107_ext"})
    )

    df["f107"] = df["f107"].fillna(daily_f107_ext["f107_ext"])

    # Extend Ap by scenario after forecast horizon
    df["ap"] = df["ap"].fillna(ap_scenario_value(config.scenario))

    # Fill any remaining f107 gaps conservatively with scenario-friendly defaults
    if config.scenario == "quiet":
        df["f107"] = df["f107"].fillna(90.0)
    elif config.scenario == "active":
        df["f107"] = df["f107"].fillna(170.0)
    else:
        df["f107"] = df["f107"].fillna(130.0)

    # 81-day average, commonly used by MSIS family models
    df["f107a"] = df["f107"].rolling(81, min_periods=1).mean()

    return df


def lookup_space_weather(
    profile: pd.DataFrame,
    when: str | pd.Timestamp,
) -> tuple[float, float, float]:
    """
    Return (f107a, f107, ap) for a given day.
    """
    ts = pd.to_datetime(when, utc=True).normalize()
    row = profile.loc[ts]
    return float(row["f107a"]), float(row["f107"]), float(row["ap"])


if __name__ == "__main__":
    profile = build_space_weather_profile(
        start="2026-04-29",
        years=5,
        config=SpaceWeatherConfig(scenario="nominal"),
    )

    print(profile.head(10))
    print()
    print(profile.tail(10))

    f107a, f107, ap = lookup_space_weather(profile, "2026-05-10")
    print()
    print(f"Example lookup for 2026-05-10: f107a={f107a:.1f}, f107={f107:.1f}, ap={ap:.1f}")