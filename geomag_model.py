from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import io
import re

import numpy as np
import pandas as pd
import requests


@dataclass
class SpaceWeatherConfig:
    timeout_s: float = 30.0
    nasa_forecast_url: str | None = None  # optional override


def _get_json(url: str, timeout_s: float):
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _get_text(url: str, timeout_s: float) -> str:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


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


def _discover_latest_nasa_forecast_url(timeout_s: float = 30.0) -> str:
    """
    Find the latest NASA Marshall forecast text file from the forecast page.

    We prefer the F10.7 table link because NASA's text table currently contains
    both F10.7 and Ap percentile forecasts in the same file.
    """
    page_url = "https://www.nasa.gov/solar-cycle-progression-and-forecast/"
    html = _get_text(page_url, timeout_s)

    # Typical current pattern:
    #   /wp-content/uploads/2026/04/apr2026f10-prd.txt
    # The page may also append query params. We ignore them.
    matches = re.findall(
        r'https://www\.nasa\.gov/wp-content/uploads/\d{4}/\d{2}/[a-z]{3}\d{4}f10-prd\.txt(?:\?[^"\']*)?',
        html,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[0]

    # Fallback: relative URL form
    rel_matches = re.findall(
        r'/wp-content/uploads/\d{4}/\d{2}/[a-z]{3}\d{4}f10-prd\.txt(?:\?[^"\']*)?',
        html,
        flags=re.IGNORECASE,
    )
    if rel_matches:
        return "https://www.nasa.gov" + rel_matches[0]

    raise ValueError("Could not discover latest NASA Marshall forecast text URL.")


def fetch_nasa_monthly_forecast(
    timeout_s: float = 30.0,
    url: str | None = None,
) -> pd.DataFrame:
    """
    Returns a monthly UTC DataFrame indexed by month-start with columns:
      - predicted_f107
      - predicted_ap

    Uses the NASA Marshall Solar Cycle forecast text table, which currently
    includes both F10.7 and Ap percentile forecasts. We use the 50th-percentile
    values.
    """
    if url is None:
        url = _discover_latest_nasa_forecast_url(timeout_s=timeout_s)

    text = _get_text(url, timeout_s)

    records: list[dict[str, float | pd.Timestamp]] = []

    # Example row shape in the NASA table:
    # 2026.0003 JAN 150.6 137.9 129.0 17.4 16.2 14.3
    #
    # Columns:
    # decimal_year, month_name, f107_95, f107_50, f107_5, ap_95, ap_50, ap_5
    row_re = re.compile(
        r"^\s*"
        r"(?P<decimal_year>\d+\.\d+)\s+"
        r"(?P<month>[A-Z]{3})\s+"
        r"(?P<f107_95>-?\d+(?:\.\d+)?)\s+"
        r"(?P<f107_50>-?\d+(?:\.\d+)?)\s+"
        r"(?P<f107_5>-?\d+(?:\.\d+)?)\s+"
        r"(?P<ap_95>-?\d+(?:\.\d+)?)\s+"
        r"(?P<ap_50>-?\d+(?:\.\d+)?)\s+"
        r"(?P<ap_5>-?\d+(?:\.\d+)?)\s*$"
    )

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    for line in text.splitlines():
        m = row_re.match(line)
        if not m:
            continue

        decimal_year = float(m.group("decimal_year"))
        year = int(decimal_year)
        month = month_map[m.group("month").upper()]

        ts = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        records.append(
            {
                "date": ts,
                "predicted_f107": float(m.group("f107_50")),
                "predicted_ap": float(m.group("ap_50")),
            }
        )

    if not records:
        raise ValueError("Could not parse NASA monthly forecast text.")

    out = (
        pd.DataFrame.from_records(records)
        .drop_duplicates(subset="date")
        .set_index("date")
        .sort_index()
    )

    return out[["predicted_f107", "predicted_ap"]]


def fetch_noaa_predicted_solar_cycle(timeout_s: float = 30.0) -> pd.DataFrame:
    """
    Optional fallback source for monthly F10.7 only.

    Returns a monthly UTC dataframe indexed by month-start date with columns:
      - predicted_ssn
      - predicted_f107
    """
    url = "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json"
    data = _get_json(url, timeout_s=timeout_s)

    records = []
    for row in data:
        t = row.get("time-tag")
        if t is None:
            continue

        f107 = (
            row.get("predicted_f10.7")
            or row.get("predicted_f107")
            or row.get("f10.7")
            or row.get("f107")
        )
        if f107 is None:
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
      - Extend ap and f107 from NASA monthly forecast
      - Fall back to NOAA solar-cycle monthly f107 if NASA forecast fetch fails
      - Compute f107a as 81-day rolling mean
    """
    if config is None:
        config = SpaceWeatherConfig()

    start_ts = pd.to_datetime(start, utc=True).normalize()
    end_ts = start_ts + pd.DateOffset(years=years)
    daily_index = pd.date_range(start=start_ts, end=end_ts, freq="D", tz="UTC")

    df = pd.DataFrame(index=daily_index, data={"ap": np.nan, "f107": np.nan})

    forecast_45 = fetch_noaa_45day_forecast(timeout_s=config.timeout_s)

    # Fill short-range daily NOAA forecast directly
    overlap = df.index.intersection(forecast_45.index)
    df.loc[overlap, "ap"] = forecast_45.loc[overlap, "ap"]
    df.loc[overlap, "f107"] = forecast_45.loc[overlap, "f107"]

    # Preferred long-range source: NASA monthly F10.7 + Ap forecast
    nasa_monthly = None
    try:
        nasa_monthly = fetch_nasa_monthly_forecast(
            timeout_s=config.timeout_s,
            url=config.nasa_forecast_url,
        )
    except Exception:
        nasa_monthly = None

    if nasa_monthly is not None:
        daily_ext = (
            nasa_monthly.reindex(nasa_monthly.index.union(df.index))
            .sort_index()
            .interpolate(method="time")
            .reindex(df.index)
        )

        df["f107"] = df["f107"].fillna(daily_ext["predicted_f107"])
        df["ap"] = df["ap"].fillna(daily_ext["predicted_ap"])

    # Secondary fallback for F10.7 only if NASA is unavailable
    if df["f107"].isna().any():
        solar_cycle = fetch_noaa_predicted_solar_cycle(timeout_s=config.timeout_s)
        daily_f107_ext = (
            solar_cycle[["predicted_f107"]]
            .reindex(solar_cycle.index.union(df.index))
            .sort_index()
            .interpolate(method="time")
            .reindex(df.index)
        )
        df["f107"] = df["f107"].fillna(daily_f107_ext["predicted_f107"])

    # Conservative hard fallback only if external sources fail
    df["f107"] = df["f107"].fillna(130.0)
    df["ap"] = df["ap"].fillna(8.0)

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
        config=SpaceWeatherConfig(),
    )

    print(profile.head(10))
    print()
    print(profile.tail(10))

    f107a, f107, ap = lookup_space_weather(profile, "2026-05-10")
    print()
    print(f"Example lookup for 2026-05-10: f107a={f107a:.1f}, f107={f107:.1f}, ap={ap:.1f}")
