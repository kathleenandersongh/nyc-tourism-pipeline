# NYC Tourism Demand Forecasting System
## OPIM-5516 Advanced Deep Learning — Project 1

A lightweight, continuously updating forecasting pipeline deployed on Google Cloud Platform.
Predicts short-term NYC tourism demand using real-time data streams, a two-headed DNN with
MC Dropout (epistemic uncertainty) and heteroscedastic loss (aleatoric uncertainty), and SHAP
for interpretability.

---

## Architecture

```
Every hour:        OpenSky (flights)  ──┐
Every 6 hours:     Open-Meteo (weather) ─┤→ GCS scrapes/ (JSON-L)
Every 6 hours:     Google Trends       ──┤
Daily at 1am:      Ticketmaster events ──┘
                                         ↓
Daily at 2am:      materialize → preds/listings_master.csv
Daily at 3am:      run_model   → preds/model_predictions.csv
                                → preds/coverage_analysis.png
                                → preds/shap_summary.png
```

**GCS Bucket Structure:**
```
gs://YOUR_BUCKET/
├── scrapes/
│   ├── opensky/      YYYY-MM-DD_HH.jsonl  (hourly)
│   ├── trends/       YYYY-MM-DD_HH.jsonl  (6-hourly)
│   ├── weather/      YYYY-MM-DD_HH.jsonl  (6-hourly)
│   └── ticketmaster/ YYYY-MM-DD.jsonl     (daily)
└── preds/
    ├── listings_master.csv       (growing dataset, 1 row per day)
    ├── model_predictions.csv     (test set predictions with uncertainty)
    ├── model_metrics.json        (MAE, RMSE, coverage)
    ├── coverage_analysis.png     (actual vs predicted + uncertainty bands)
    ├── shap_summary.png          (beeswarm plot)
    └── feature_importance.csv   (ranked SHAP values)
```

---

## Setup Instructions

### Step 1: GCP Setup
1. Create a GCP project and enable billing (use your Vertex credits!)
2. Enable these APIs:
   - Cloud Functions API
   - Cloud Scheduler API
   - Cloud Storage API
   - Cloud Build API
3. Create a GCS bucket: `gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME`
4. Create a Service Account with roles:
   - `roles/cloudfunctions.admin`
   - `roles/cloudscheduler.admin`
   - `roles/storage.admin`
5. Download the Service Account JSON key

### Step 2: Get API Keys
| Service | URL | Notes |
|---|---|---|
| OpenSky | https://opensky-network.org/index.php?option=com_users&view=registration | Free; anonymous works but is rate-limited |
| Ticketmaster | https://developer.ticketmaster.com | Free; 5000 req/day |
| Open-Meteo | No key needed | Fully free |
| Google Trends | No key needed (pytrends) | Rate-limited; 6hr schedule is safe |

### Step 3: GitHub Repository Secrets
Go to your repo → Settings → Secrets → Actions → New repository secret:

| Secret Name | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID (e.g., `my-tourism-project`) |
| `GCP_BUCKET_NAME` | Your GCS bucket name |
| `GCP_SA_KEY` | The full JSON content of your service account key |
| `OPENSKY_USER` | OpenSky username (or leave empty for anonymous) |
| `OPENSKY_PASS` | OpenSky password (or leave empty) |
| `TM_API_KEY` | Ticketmaster API key |

### Step 4: Deploy
Push to `main` branch → GitHub Actions will deploy all Cloud Functions and set up Cloud Scheduler automatically.

```bash
git add .
git commit -m "Initial pipeline deployment"
git push origin main
```

### Step 5: Bootstrap historical data (optional but recommended)
Backfill a few weeks of data before your model trains by manually triggering the
materialize function with specific dates:

```bash
# Trigger via curl (replace URL with your deployed function URL)
curl -X POST https://YOUR_MATERIALIZE_URL \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-02-01"}'
```

---

## Model Details

### Target Variable: `nyc_tourism_index` (0–100)
Weighted composite of:
- **50%** Total flight arrivals (JFK + LGA + EWR), normalized
- **30%** Google Trends average interest score for NYC travel keywords
- **20%** Ticketmaster events count for that day

### Features (X)
23 features across 5 categories: flights, trends, weather, events, calendar

### Architecture
```
Input (23 features)
    ↓
Dense(128) + BatchNorm + Dropout(0.3) ← training=True (MC Dropout)
    ↓
Dense(64)  + BatchNorm + Dropout(0.3)
    ↓
Dense(32)  + BatchNorm + Dropout(0.3)
    ↓
   ┌──────────────────────────────────┐
   │                                  │
Dense(1) → mu           Dense(1)*5 → log_var
(point estimate)        (aleatoric uncertainty)
   │                                  │
   └──────────┬───────────────────────┘
              │
         Concatenate → NLL Loss
```

### Uncertainty Sources
- **Aleatoric** (irreducible data noise): from `exp(log_var)` head
- **Epistemic** (model uncertainty): from variance across 50 MC Dropout passes
- **Total**: `sqrt(aleatoric_var + epistemic_var)`

---

## File Structure
```
nyc_tourism/
├── cloud_functions/
│   ├── scrape_opensky/      main.py, requirements.txt
│   ├── scrape_trends/       main.py, requirements.txt
│   ├── scrape_weather/      main.py, requirements.txt
│   ├── scrape_ticketmaster/ main.py, requirements.txt
│   ├── materialize/         main.py, requirements.txt
│   └── run_model/           main.py, requirements.txt
├── .github/workflows/
│   └── deploy.yml
└── README.md
```

---

## Cost Estimate (Free Tier)

| Service | Usage | Cost |
|---|---|---|
| Cloud Functions | ~750 invocations/month | Free (2M/month) |
| Cloud Scheduler | 6 jobs | Free (3 jobs free; $0.10/job/month after) |
| Cloud Storage | ~50MB/month | Free (5GB free) |
| **Total** | | **~$0.30/month** |

With Vertex credits: effectively free.

---

## Known Limitations & Future Work
- Need 30+ days of data before the model is meaningful
- Target variable weights (flights 50%, trends 30%, events 20%) are heuristic — revisit after EDA
- Google Trends pytrends can get blocked if called too frequently; 6-hour cadence is conservative
- OpenSky anonymous tier is rate-limited; authenticated access gives more reliable data
- Add MTA turnstile data for a ground-truth foot traffic signal
