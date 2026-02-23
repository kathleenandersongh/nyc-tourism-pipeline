"""
Cloud Function: materialize
Reads all raw scrape JSON-L files from the last 24 hours,
constructs a daily feature row, and appends it to listings_master.csv.

Runs once per day at :30 (after all daily scrapers have completed).
Output: preds/listings_master.csv (same pattern as class GCP pipeline)

Target variable: nyc_tourism_index
  = weighted combination of:
    - total_arrivals_jfk_lga_ewr (primary signal, most reliable)
    - nyc_travel_trends_avg (Google Trends mean across travel keywords)
    - total_events_tonight (Ticketmaster events for today)
  All normalized to [0,1] then weighted average.
  This gives a composite 0-100 index reflecting tourism pressure.
"""

import json
import os
import io
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name")

# Feature weights for the tourism index target variable
# Adjust these after analyzing — flights are the most reliable signal
WEIGHTS = {
    "flights": 0.50,
    "trends": 0.30,
    "events": 0.20,
}

# Historical reference ranges for normalization (update after a few weeks of data)
# These are rough estimates — the model learns to normalize internally too
FLIGHT_RANGE = (0, 400)       # total arrivals across 3 airports per hour × 24
TRENDS_RANGE = (0, 100)       # Google Trends is already 0-100
EVENTS_RANGE = (0, 50)        # events per day in NYC


def load_jsonl_from_gcs(client: storage.Client, prefix: str, date_str: str) -> list:
    """Load all JSON-L blobs matching a date prefix."""
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))
    records = []
    for blob in blobs:
        if date_str in blob.name:
            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def normalize(value: Optional[float], min_val: float, max_val: float) -> Optional[float]:
    """Clip and normalize a value to [0, 1]."""
    if value is None:
        return None
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def build_feature_row(date_str: str, client: storage.Client) -> dict:
    """
    Aggregates all raw scrape data for a given date into a single feature row.
    Returns a flat dict ready for CSV appending.
    """
    row = {"date": date_str}

    # ── FLIGHTS ──────────────────────────────────────────────────────────────
    flight_records = load_jsonl_from_gcs(client, "scrapes/opensky/", date_str)
    if flight_records:
        total_arrivals = sum(
            r.get("arrivals_last_hour", 0) or 0
            for r in flight_records
        )
        intl_arrivals = sum(
            r.get("international_arrivals_last_hour", 0) or 0
            for r in flight_records
        )
        jfk_arrivals = sum(
            r.get("arrivals_last_hour", 0) or 0
            for r in flight_records if r.get("airport") == "JFK"
        )
        row["total_arrivals"] = total_arrivals
        row["international_arrivals"] = intl_arrivals
        row["jfk_arrivals"] = jfk_arrivals
        row["intl_share"] = round(intl_arrivals / total_arrivals, 3) if total_arrivals > 0 else None
        flights_norm = normalize(total_arrivals, *FLIGHT_RANGE)
    else:
        row.update({"total_arrivals": None, "international_arrivals": None,
                    "jfk_arrivals": None, "intl_share": None})
        flights_norm = None

    # ── GOOGLE TRENDS ─────────────────────────────────────────────────────────
    trends_records = load_jsonl_from_gcs(client, "scrapes/trends/", date_str)
    if trends_records:
        scores = [r.get("interest_score") for r in trends_records if r.get("interest_score") is not None]
        avg_trends = sum(scores) / len(scores) if scores else None

        # Separate travel intent keywords for targeted features
        flight_kws = [r for r in trends_records if "flight" in r.get("keyword", "").lower()]
        hotel_kws = [r for r in trends_records if "hotel" in r.get("keyword", "").lower()]

        row["trends_avg_score"] = round(avg_trends, 1) if avg_trends else None
        row["trends_flight_intent"] = round(
            sum(r["interest_score"] for r in flight_kws) / len(flight_kws), 1
        ) if flight_kws else None
        row["trends_hotel_intent"] = round(
            sum(r["interest_score"] for r in hotel_kws) / len(hotel_kws), 1
        ) if hotel_kws else None
        trends_norm = normalize(avg_trends, *TRENDS_RANGE)
    else:
        row.update({"trends_avg_score": None, "trends_flight_intent": None, "trends_hotel_intent": None})
        trends_norm = None

    # ── WEATHER ───────────────────────────────────────────────────────────────
    weather_records = load_jsonl_from_gcs(client, "scrapes/weather/", date_str)
    if weather_records:
        # Get today's current conditions (record_type=current, days_ahead=0)
        today_weather = [r for r in weather_records if r.get("record_type") == "current"]
        if today_weather:
            w = today_weather[-1]  # most recent pull
            row["temp_f"] = w.get("temp_f")
            row["precipitation_mm"] = w.get("precipitation_mm")
            row["is_bad_weather"] = w.get("is_bad_weather")
            row["wind_mph"] = w.get("wind_mph")

        # Get 3-day forecast outlook (for predictive features)
        forecast = [r for r in weather_records if r.get("record_type") == "forecast" and r.get("days_ahead") in [1, 2, 3]]
        if forecast:
            row["avg_precip_prob_3d"] = round(
                sum(r.get("precip_prob_pct", 0) or 0 for r in forecast) / len(forecast), 1
            )
            row["bad_weather_days_3d"] = sum(r.get("is_bad_weather", 0) or 0 for r in forecast)
        else:
            row["avg_precip_prob_3d"] = None
            row["bad_weather_days_3d"] = None
    else:
        row.update({
            "temp_f": None, "precipitation_mm": None, "is_bad_weather": None,
            "wind_mph": None, "avg_precip_prob_3d": None, "bad_weather_days_3d": None,
        })

    # ── TICKETMASTER ──────────────────────────────────────────────────────────
    event_records = load_jsonl_from_gcs(client, "scrapes/ticketmaster/", date_str)
    if event_records:
        # Events happening today or tomorrow
        today_events = [r for r in event_records if r.get("event_date") == date_str]
        tomorrow = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        tmrw_events = [r for r in event_records if r.get("event_date") == tomorrow]

        today_total = sum(r.get("total_events", 0) or 0 for r in today_events)
        tmrw_total = sum(r.get("total_events", 0) or 0 for r in tmrw_events)

        row["events_today"] = today_total
        row["events_tomorrow"] = tmrw_total
        row["music_events_today"] = sum(r.get("music_events", 0) or 0 for r in today_events)
        row["sports_events_today"] = sum(r.get("sports_events", 0) or 0 for r in today_events)
        row["avg_ticket_price_today"] = (
            round(sum(r.get("avg_min_ticket_price", 0) or 0 for r in today_events) / len(today_events), 2)
            if today_events else None
        )
        events_norm = normalize(today_total, *EVENTS_RANGE)
    else:
        row.update({
            "events_today": None, "events_tomorrow": None,
            "music_events_today": None, "sports_events_today": None,
            "avg_ticket_price_today": None,
        })
        events_norm = None

    # ── CALENDAR FEATURES ─────────────────────────────────────────────────────
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    row["day_of_week"] = dt.weekday()          # 0=Mon, 6=Sun
    row["is_weekend"] = int(dt.weekday() >= 5)
    row["month"] = dt.month
    row["day_of_year"] = dt.timetuple().tm_yday
    row["week_of_year"] = dt.isocalendar()[1]

    # ── TARGET VARIABLE: nyc_tourism_index ────────────────────────────────────
    # Weighted composite of available signals (all normalized to [0,1] × 100)
    components = []
    component_weights = []

    if flights_norm is not None:
        components.append(flights_norm * WEIGHTS["flights"])
        component_weights.append(WEIGHTS["flights"])
    if trends_norm is not None:
        components.append(trends_norm * WEIGHTS["trends"])
        component_weights.append(WEIGHTS["trends"])
    if events_norm is not None:
        components.append(events_norm * WEIGHTS["events"])
        component_weights.append(WEIGHTS["events"])

    if component_weights:
        # Re-normalize weights for missing components
        total_weight = sum(component_weights)
        tourism_index = sum(components) / total_weight * 100
        row["nyc_tourism_index"] = round(tourism_index, 2)
    else:
        row["nyc_tourism_index"] = None  # will be dropped during model training

    row["data_completeness"] = len(component_weights) / 3  # 0.33, 0.67, or 1.0

    return row


def load_master_csv(client: storage.Client) -> pd.DataFrame:
    """Load the existing master CSV or return empty DataFrame."""
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("preds/listings_master.csv")
    try:
        content = blob.download_as_text()
        return pd.read_csv(io.StringIO(content))
    except Exception:
        return pd.DataFrame()


def save_master_csv(client: storage.Client, df: pd.DataFrame):
    """Upload the master CSV back to GCS."""
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("preds/listings_master.csv")
    csv_content = df.to_csv(index=False)
    blob.upload_from_string(csv_content, content_type="text/csv")


def materialize(request):
    """HTTP Cloud Function entry point."""
    client = storage.Client()

    # Default: materialize yesterday's data (all scrapers have completed by then)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    # Allow override via request body: {"date": "2026-02-01"}
    try:
        body = request.get_json(silent=True) or {}
        date_str = body.get("date", yesterday)
    except Exception:
        date_str = yesterday

    print(f"[Materialize] Building feature row for {date_str}")
    new_row = build_feature_row(date_str, client)

    # Load existing master, deduplicate, append
    master_df = load_master_csv(client)
    new_df = pd.DataFrame([new_row])

    if not master_df.empty and "date" in master_df.columns:
        # Remove any existing row for this date (idempotent)
        master_df = master_df[master_df["date"] != date_str]
        master_df = pd.concat([master_df, new_df], ignore_index=True)
        master_df = master_df.sort_values("date").reset_index(drop=True)
    else:
        master_df = new_df

    save_master_csv(client, master_df)

    print(f"[Materialize] Master CSV now has {len(master_df)} rows")
    print(f"[Materialize] Tourism index for {date_str}: {new_row.get('nyc_tourism_index')}")
    return f"OK: materialized {date_str}, index={new_row.get('nyc_tourism_index')}", 200
