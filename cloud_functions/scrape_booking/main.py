import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import date, timedelta
import statistics

LAT = 40.7484
LON = -73.9857
RADIUS = 1.6

CHECKIN_DAYS = [7,30,60]

def build_url(checkin):

    checkout = checkin + timedelta(days=1)

    return f"https://www.booking.com/searchresults.html?ss=Empire+State+Building&dest_type=landmark&sslat={LAT}&sslon={LON}&radius={RADIUS}&checkin={checkin}&checkout={checkout}&group_adults=2&no_rooms=1"


def scrape():

    results = []

    today = date.today()

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=False)

        page = browser.new_page()

        for days in CHECKIN_DAYS:

            checkin = today + timedelta(days=days)

            url = build_url(checkin)

            print("Loading:",url)

            page.goto(url, timeout=60000)

            page.wait_for_timeout(12000)

            prices = []

            cards = page.query_selector_all(
                '[data-testid="property-card"]'
            )

            print("Hotels found:",len(cards))

            for c in cards:

                price_el = c.query_selector(
                '[data-testid="price-and-discounted-price"]'
                )

                if price_el:

                    text = price_el.inner_text()

                    digits = ''.join(
                    x for x in text if x.isdigit()
                    )

                    if digits:

                        prices.append(int(digits))

            if prices:

                results.append({

                    "date":today,

                    "checkin_date":checkin,

                    "lead_days":days,

                    "hotel_count":len(prices),

                    "avg_price":
                    statistics.mean(prices),

                    "median_price":
                    statistics.median(prices),

                    "min_price":
                    min(prices),

                    "max_price":
                    max(prices)

                })

        browser.close()

    return results


results = scrape()

df = pd.DataFrame(results)

print("\nFinal DataFrame:\n")
print(df)

df.to_csv("booking_esb_features.csv",index=False)

print("\nSaved file: booking_esb_features.csv")
print("Rows:",len(df))