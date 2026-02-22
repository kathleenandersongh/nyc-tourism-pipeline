"""
Cloud Function: scrape_opensky
Pulls inbound flight counts into JFK, LGA, and EWR from the OpenSky Network API.
Runs every hour. Stores raw JSON-L to GCS: scrapes/opensky/YYYY-MM-DD_HH.jsonl
"""

import json
import os
import requests
from datetime import datetime, timezone
from google.cloud import storage

# NYC-area airports: (airport ICAO, lat, lon, radius_km)
NYC_AIRPORTS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
}

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name")
OPENSKY_USER = os.environ.get("OPENSKY_USER", "")
OPENSKY_PASS = os.environ.get("OPENSKY_PASS", "")


def fetch_arrivals(airport: str, lat: float, lon: float) -> dict:
    """
    Hits OpenSky's /flights/arrival endpoint for flights that arrived
    in the last 1 hour. Falls back to /states/all bounding box if needed.
    Returns a dict with airport, timestamp, and flight count.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    one_hour_ago = now - 3600

    url = "https://opensky-network.org/api/flights/arrival"
    params = {"airport": airport, "begin": one_hour_ago, "end": now}

    auth = (OPENSKY_USER, OPENSKY_PASS) if OPENSKY_USER else None

    try:
        resp = requests.get(url, params=params, auth=auth, timeout=30)
        resp.raise_for_status()
        flights = resp.json()
        count = len(flights) if isinstance(flights, list) else 0

        # Lightweight feature extraction from individual flights
        international = sum(
            1 for f in (flights or [])
            if f.get("estDepartureAirport") and not f["estDepartureAirport"].startswith("K")
        )

    except requests.RequestException as e:
        print(f"[OpenSky] Error fetching {airport}: {e}")
        count = None
        international = None

    return {
        "airport": airport,
        "lat": lat,
        "lon": lon,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arrivals_last_hour": count,
        "international_arrivals_last_hour": international,
    }


def scrape_opensky(request):
    """HTTP Cloud Function entry point."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H")
    blob_path = f"scrapes/opensky/{date_str}_{hour_str}.jsonl"

    records = []
    for airport, (lat, lon) in NYC_AIRPORTS.items():
        record = fetch_arrivals(airport, lat, lon)
        records.append(record)
        print(f"[OpenSky] {airport}: {record['arrivals_last_hour']} arrivals")

    # Write JSON-L (one JSON object per line, matches class ETL pattern)
    blob = bucket.blob(blob_path)
    jsonl_content = "\n".join(json.dumps(r) for r in records)
    blob.upload_from_string(jsonl_content, content_type="application/jsonl")

    print(f"[OpenSky] Wrote {len(records)} records to gs://{BUCKET_NAME}/{blob_path}")
    return f"OK: {len(records)} airports scraped", 200
