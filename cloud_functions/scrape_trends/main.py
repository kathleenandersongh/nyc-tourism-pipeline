"""
Cloud Function: scrape_trends
Pulls Google Trends interest scores for NYC travel-related keywords.
Runs every 6 hours (Trends data updates slowly; over-polling returns cached data).
Stores: scrapes/trends/YYYY-MM-DD_HH.jsonl
"""

import json
import os
from datetime import datetime, timezone, timedelta
from google.cloud import storage
from pytrends.request import TrendReq

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name")

# Keywords reflecting travel intent — capturing different stages of the funnel
KEYWORD_GROUPS = [
    # Intent: planning a trip
    ["NYC trip", "New York vacation", "visit New York", "NYC travel", "New York tourism"],
    # Intent: booking flights
    ["flights to NYC", "flights to New York", "cheap flights JFK", "flights to LaGuardia"],
    # Intent: booking hotels
    ["NYC hotels", "hotels New York City", "Manhattan hotel deals"],
    # Intent: things to do
    ["things to do NYC", "NYC attractions", "Broadway tickets", "NYC events"],
]

GEO = "US"  # Nationwide US — reflects outbound US tourism intent toward NYC


def fetch_trends_group(pytrends: TrendReq, keywords: list, timeframe: str) -> list:
    """Fetch interest over time for a keyword group and return flattened records."""
    try:
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=GEO, gprop="")
        df = pytrends.interest_over_time()
        if df.empty:
            return []

        # Take only the most recent row (last data point in the timeframe)
        latest = df.iloc[-1]
        records = []
        for kw in keywords:
            if kw in latest:
                records.append({
                    "keyword": kw,
                    "interest_score": int(latest[kw]),  # 0-100 scale
                    "is_partial": bool(latest.get("isPartial", False)),
                })
        return records
    except Exception as e:
        print(f"[Trends] Error for keywords {keywords}: {e}")
        return []


def scrape_trends(request):
    """HTTP Cloud Function entry point."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    now = datetime.now(timezone.utc)
    # Use 7-day timeframe for stability (Trends daily data is noisy for short windows)
    timeframe = f"{(now - timedelta(days=7)).strftime('%Y-%m-%d')} {now.strftime('%Y-%m-%d')}"

    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

    all_records = []
    for group in KEYWORD_GROUPS:
        records = fetch_trends_group(pytrends, group, timeframe)
        all_records.extend(records)

    # Tag all records with timestamp
    timestamp = now.isoformat()
    for r in all_records:
        r["timestamp_utc"] = timestamp

    # Write JSON-L
    date_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H")
    blob_path = f"scrapes/trends/{date_str}_{hour_str}.jsonl"
    blob = bucket.blob(blob_path)
    jsonl_content = "\n".join(json.dumps(r) for r in all_records)
    blob.upload_from_string(jsonl_content, content_type="application/jsonl")

    print(f"[Trends] Wrote {len(all_records)} keyword scores to gs://{BUCKET_NAME}/{blob_path}")
    return f"OK: {len(all_records)} trends captured", 200
