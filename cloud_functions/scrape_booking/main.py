import io
import os
import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import date, timedelta
import statistics
from google.cloud import storage

LAT = 40.7484
LON = -73.9857
RADIUS = 1.6
CHECKIN_DAYS = [7, 30, 60]

BUCKET_NAME = os.getenv("BUCKET_NAME", "nyc-tourism-kathleen-2026")

def build_url(checkin: date) -> str:
    checkout = checkin + timedelta(days=1)
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss=Empire+State+Building"
        f"&dest_type=landmark"
        f"&sslat={LAT}"
        f"&sslon={LON}"
        f"&radius={RADIUS}"
        f"&checkin={checkin}"
        f"&checkout={checkout}"
        f"&group_adults=2"
        f"&no_rooms=1"
    )

def scrape() -> pd.DataFrame:
    today = date.today()
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for days in CHECKIN_DAYS:
            checkin = today + timedelta(days=days)
            url = build_url(checkin)

            page.goto(url, timeout=60000)
            page.wait_for_timeout(12000)

            prices = []
            cards = page.query_selector_all('[data-testid="property-card"]')

            for c in cards:
                price_el = c.query_selector('[data-testid="price-and-discounted-price"]')
                if not price_el:
                    continue

                text = price_el.inner_text()
                digits = "".join(ch for ch in text if ch.isdigit())
                if digits:
                    prices.append(int(digits))

            if prices:
                rows.append(
                    {
                        "date": str(today),
                        "checkin_date": str(checkin),
                        "lead_days": days,
                        "hotel_count": len(prices),
                        "avg_price": statistics.mean(prices),
                        "median_price": statistics.median(prices),
                        "min_price": min(prices),
                        "max_price": max(prices),
                    }
                )

        browser.close()

    return pd.DataFrame(rows)

def upload_to_gcs(df: pd.DataFrame) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    object_name = f"scrapes/booking/booking_esb_features_{date.today()}.csv"

    buf = io.StringIO()
    df.to_csv(buf, index=False)

    blob = bucket.blob(object_name)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")

    return object_name

def scrape_booking(request):
    df = scrape()
    object_name = upload_to_gcs(df)
    return f"OK: rows={len(df)} wrote={object_name}\n"
