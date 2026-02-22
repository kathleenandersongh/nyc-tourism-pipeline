"""
Cloud Function: scrape_ticketmaster
Pulls upcoming events in NYC from the Ticketmaster Discovery API.
Runs once per day (event listings don't change hourly).
Stores: scrapes/ticketmaster/YYYY-MM-DD.jsonl

Free API key: https://developer.ticketmaster.com/products-and-docs/apis/getting-started/
Rate limit: 5000 req/day on free tier — daily scrape is well within limits.
"""

import json
import os
import requests
from datetime import datetime, timezone, timedelta
from google.cloud import storage

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name")
TM_API_KEY = os.environ.get("TM_API_KEY", "")  # Set in Cloud Function env vars

TM_BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

# NYC DMA code: 345 (covers metro area)
NYC_DMA_ID = "345"


def fetch_events(days_ahead: int = 14) -> list:
    """
    Fetches events in NYC for the next N days.
    Returns summarized records — count by date, segment, and price tier.
    """
    now = datetime.now(timezone.utc)
    start_dt = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_dt = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_events = []
    page = 0
    page_size = 200  # max allowed

    while True:
        params = {
            "apikey": TM_API_KEY,
            "dmaId": NYC_DMA_ID,
            "startDateTime": start_dt,
            "endDateTime": end_dt,
            "size": page_size,
            "page": page,
            "sort": "date,asc",
        }

        try:
            resp = requests.get(TM_BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[Ticketmaster] Error on page {page}: {e}")
            break

        embedded = data.get("_embedded", {})
        events = embedded.get("events", [])
        if not events:
            break

        all_events.extend(events)

        # Pagination
        page_info = data.get("page", {})
        total_pages = page_info.get("totalPages", 1)
        page += 1
        if page >= total_pages or page >= 10:  # safety cap at 10 pages = 2000 events
            break

    # Aggregate: count events per date + segment (Music, Sports, Arts, etc.)
    daily_counts = {}  # date -> {segment -> count, total -> count, avg_min_price}
    for event in all_events:
        # Date
        dates = event.get("dates", {}).get("start", {})
        event_date = dates.get("localDate", "unknown")

        # Segment (category)
        segment = "Other"
        classifications = event.get("classifications", [])
        if classifications:
            seg = classifications[0].get("segment", {})
            segment = seg.get("name", "Other")

        # Min price
        price_ranges = event.get("priceRanges", [])
        min_price = price_ranges[0].get("min") if price_ranges else None

        if event_date not in daily_counts:
            daily_counts[event_date] = {"total": 0, "segments": {}, "prices": []}

        daily_counts[event_date]["total"] += 1
        daily_counts[event_date]["segments"][segment] = (
            daily_counts[event_date]["segments"].get(segment, 0) + 1
        )
        if min_price is not None:
            daily_counts[event_date]["prices"].append(min_price)

    # Flatten into records
    timestamp_utc = now.isoformat()
    records = []
    for event_date, counts in daily_counts.items():
        prices = counts["prices"]
        record = {
            "timestamp_utc": timestamp_utc,
            "event_date": event_date,
            "total_events": counts["total"],
            "music_events": counts["segments"].get("Music", 0),
            "sports_events": counts["segments"].get("Sports", 0),
            "arts_events": counts["segments"].get("Arts & Theatre", 0),
            "avg_min_ticket_price": round(sum(prices) / len(prices), 2) if prices else None,
            "max_min_ticket_price": max(prices) if prices else None,
        }
        records.append(record)

    return records


def scrape_ticketmaster(request):
    """HTTP Cloud Function entry point."""
    if not TM_API_KEY:
        return "ERROR: TM_API_KEY environment variable not set", 500

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    records = fetch_events(days_ahead=14)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    blob_path = f"scrapes/ticketmaster/{date_str}.jsonl"

    blob = bucket.blob(blob_path)
    jsonl_content = "\n".join(json.dumps(r) for r in records)
    blob.upload_from_string(jsonl_content, content_type="application/jsonl")

    print(f"[Ticketmaster] {len(records)} event-date records → gs://{BUCKET_NAME}/{blob_path}")
    return f"OK: {len(records)} event-date aggregates", 200
