"""
Cloud Function: scrape_weather
Pulls current + 7-day forecast weather for NYC from Open-Meteo (free, no API key).
Runs every 6 hours.
Stores: scrapes/weather/YYYY-MM-DD_HH.jsonl

Key insight: we want FORECAST data, not just current conditions.
Poor forecasts for upcoming weekends predict reduced tourism.
"""

import json
import os
import requests
from datetime import datetime, timezone
from google.cloud import storage

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name")

# NYC coordinates (Manhattan center)
NYC_LAT = 40.7589
NYC_LON = -73.9851

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather() -> list:
    """
    Fetches current conditions + 7-day daily forecast for NYC.
    Returns a list of records (one per forecast day).
    """
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "current": [
            "temperature_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weather_code",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "sunshine_duration",
        ],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",
        "forecast_days": 7,
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    now_utc = datetime.now(timezone.utc).isoformat()

    # Parse current conditions into a record
    current = data.get("current", {})
    current_record = {
        "record_type": "current",
        "forecast_date": data["daily"]["time"][0],  # today
        "days_ahead": 0,
        "timestamp_utc": now_utc,
        "temp_f": current.get("temperature_2m"),
        "precipitation_mm": current.get("precipitation"),
        "weather_code": current.get("weather_code"),
        "wind_mph": current.get("wind_speed_10m"),
        # Derived: is it "bad" weather? WMO codes 51+ = rain/snow/storm
        "is_bad_weather": int(current.get("weather_code", 0) >= 51),
        "temp_max_f": None,
        "temp_min_f": None,
        "precip_prob_pct": None,
        "sunshine_seconds": None,
    }

    # Parse daily forecasts
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    forecast_records = []
    for i, date in enumerate(dates):
        weather_code = daily["weather_code"][i] if daily.get("weather_code") else None
        record = {
            "record_type": "forecast",
            "forecast_date": date,
            "days_ahead": i,
            "timestamp_utc": now_utc,
            "temp_f": None,  # no current temp for future days
            "precipitation_mm": daily["precipitation_sum"][i] if daily.get("precipitation_sum") else None,
            "weather_code": weather_code,
            "wind_mph": daily["wind_speed_10m_max"][i] if daily.get("wind_speed_10m_max") else None,
            "is_bad_weather": int(weather_code >= 51) if weather_code is not None else None,
            "temp_max_f": daily["temperature_2m_max"][i] if daily.get("temperature_2m_max") else None,
            "temp_min_f": daily["temperature_2m_min"][i] if daily.get("temperature_2m_min") else None,
            "precip_prob_pct": daily["precipitation_probability_max"][i] if daily.get("precipitation_probability_max") else None,
            "sunshine_seconds": daily["sunshine_duration"][i] if daily.get("sunshine_duration") else None,
        }
        forecast_records.append(record)

    return [current_record] + forecast_records


def scrape_weather(request):
    """HTTP Cloud Function entry point."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    records = fetch_weather()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H")
    blob_path = f"scrapes/weather/{date_str}_{hour_str}.jsonl"

    blob = bucket.blob(blob_path)
    jsonl_content = "\n".join(json.dumps(r) for r in records)
    blob.upload_from_string(jsonl_content, content_type="application/jsonl")

    print(f"[Weather] Wrote {len(records)} records to gs://{BUCKET_NAME}/{blob_path}")
    return f"OK: {len(records)} weather records", 200
