"""
Cloud Function: run_model
Trains a two-headed DNN on the master CSV and writes predictions + uncertainty to GCS.

Architecture matches class (M1_2_DNNs_SHAP_Epistemic.ipynb):
  - Head 1: mu (predicted tourism index)
  - Head 2: log_var (predicted aleatoric uncertainty / data noise)
  - MC Dropout: 50 stochastic forward passes → epistemic uncertainty
  - Custom NLL loss couples both heads
  - SHAP for global + local interpretability

Output files:
  - preds/model_predictions.csv  (mu, aleatoric_std, epistemic_std, total_std per row)
  - preds/coverage_analysis.png
  - preds/shap_summary.png
  - preds/feature_importance.csv
"""

import io
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import shap

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name")

# ── FEATURE COLUMNS ──────────────────────────────────────────────────────────
# All numeric features used as model inputs (X)
FEATURE_COLS = [
    # Flight signals
    "total_arrivals", "international_arrivals", "jfk_arrivals", "intl_share",
    # Trends signals
    "trends_avg_score", "trends_flight_intent", "trends_hotel_intent",
    # Weather signals
    "temp_f", "precipitation_mm", "is_bad_weather", "wind_mph",
    "avg_precip_prob_3d", "bad_weather_days_3d",
    # Event signals
    "events_today", "events_tomorrow", "music_events_today",
    "sports_events_today", "avg_ticket_price_today",
    # Calendar features
    "day_of_week", "is_weekend", "month", "day_of_year", "week_of_year",
]

TARGET_COL = "nyc_tourism_index"


# ── CUSTOM LOSS: Negative Log Likelihood ─────────────────────────────────────
def nll_loss(y_true, y_pred):
    """
    Gaussian negative log likelihood loss.
    y_pred has shape (batch, 2): [mu, log_var]
    This couples both heads during training — class-style.
    """
    mu = y_pred[:, 0:1]
    log_var = y_pred[:, 1:2]

    # Clamp log_var for numerical stability
    log_var = tf.clip_by_value(log_var, -10, 10)
    var = tf.exp(log_var)

    nll = 0.5 * (log_var + tf.square(y_true - mu) / var)
    return tf.reduce_mean(nll)


# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────
def build_model(n_features: int, dropout_rate: float = 0.3) -> Model:
    """
    Two-headed DNN. Matches class architecture:
    - Shared trunk with BatchNorm + Dropout (Dropout stays ON at inference for MC)
    - Head 1: mu (mean prediction)
    - Head 2: log_var (log variance = aleatoric uncertainty)
    """
    inputs = keras.Input(shape=(n_features,), name="features")

    # Shared trunk
    x = layers.Dense(128, activation="relu", name="dense_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x, training=True)  # training=True → always active

    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x, training=True)

    x = layers.Dense(32, activation="relu", name="dense_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(dropout_rate, name="dropout_3")(x, training=True)

    # Head 1: mu — predicts the tourism index
    mu = layers.Dense(1, activation="linear", name="mu")(x)

    # Head 2: log_var — predicts aleatoric uncertainty (data noise)
    # Sigmoid-scaled to reasonable range, then shifted: outputs log_var in [-5, 5]
    log_var = layers.Dense(1, activation="tanh", name="log_var_raw")(x)
    log_var = layers.Lambda(lambda v: v * 5, name="log_var")(log_var)

    # Concatenate both heads for NLL loss
    output = layers.Concatenate(name="mu_logvar")([mu, log_var])

    model = Model(inputs=inputs, outputs=output, name="TwoHeadedDNN")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=nll_loss)
    return model


# ── MC DROPOUT PREDICTION ─────────────────────────────────────────────────────
def mc_predict(model: Model, X: np.ndarray, n_passes: int = 50) -> dict:
    """
    Run N stochastic forward passes with Dropout active.
    Returns mean, aleatoric uncertainty (from log_var), and epistemic uncertainty (from variance across passes).

    Matches class approach exactly.
    """
    # Stack predictions from all passes: shape (n_passes, n_samples, 2)
    all_preds = np.stack([model.predict(X, verbose=0) for _ in range(n_passes)], axis=0)

    # Head 1: mu predictions across passes
    mu_preds = all_preds[:, :, 0]          # shape (n_passes, n_samples)
    # Head 2: log_var predictions across passes
    log_var_preds = all_preds[:, :, 1]     # shape (n_passes, n_samples)

    # Mean prediction (average mu across passes)
    mu_mean = mu_preds.mean(axis=0)        # shape (n_samples,)

    # Aleatoric uncertainty: average of exp(log_var) across passes
    aleatoric_var = np.exp(log_var_preds).mean(axis=0)
    aleatoric_std = np.sqrt(aleatoric_var)

    # Epistemic uncertainty: variance of mu across passes
    epistemic_var = mu_preds.var(axis=0)
    epistemic_std = np.sqrt(epistemic_var)

    # Total uncertainty (both sources combined)
    total_std = np.sqrt(aleatoric_var + epistemic_var)

    return {
        "mu": mu_mean,
        "aleatoric_std": aleatoric_std,
        "epistemic_std": epistemic_std,
        "total_std": total_std,
    }


# ── COVERAGE ANALYSIS ─────────────────────────────────────────────────────────
def compute_coverage(y_true: np.ndarray, mu: np.ndarray, total_std: np.ndarray,
                     z: float = 1.96) -> float:
    """
    Compute empirical coverage: what fraction of true values fall within ±z*std?
    Target: ~95% for z=1.96 (Gaussian assumption).
    """
    lower = mu - z * total_std
    upper = mu + z * total_std
    in_interval = ((y_true >= lower) & (y_true <= upper)).mean()
    return float(in_interval)


def plot_coverage(y_true, mu, total_std, y_scaler, client, bucket_name, blob_path):
    """Generate actual vs predicted plot with uncertainty bands. Matches class style."""
    # Inverse transform if scaled
    # (if you scale Y, invert here — for index 0-100 you may not need scaling)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_true, mu, alpha=0.5, s=20, color="steelblue", label="Predictions")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Tourism Index")
    ax.set_ylabel("Predicted Tourism Index")
    ax.set_title("Actual vs Predicted (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Sorted predictions with uncertainty bands
    ax2 = axes[1]
    sort_idx = np.argsort(y_true)
    x_vals = np.arange(len(y_true))
    ax2.plot(x_vals, y_true[sort_idx], "k.", markersize=3, label="Actual", alpha=0.7)
    ax2.plot(x_vals, mu[sort_idx], "b-", lw=1.5, label="Predicted μ", alpha=0.8)
    ax2.fill_between(
        x_vals,
        mu[sort_idx] - 1.96 * total_std[sort_idx],
        mu[sort_idx] + 1.96 * total_std[sort_idx],
        alpha=0.25, color="blue", label="95% CI (total uncertainty)"
    )
    coverage = compute_coverage(y_true, mu, total_std)
    ax2.set_title(f"Predictions + Uncertainty | Coverage: {coverage:.1%}")
    ax2.set_xlabel("Sorted Samples")
    ax2.set_ylabel("Tourism Index")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    blob = client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_file(buf, content_type="image/png")
    plt.close()
    return coverage


def plot_shap_summary(model, X_test, feature_names, client, bucket_name, blob_path):
    """Compute SHAP values using DeepExplainer and save beeswarm plot."""
    # Use first 100 test samples as background for SHAP
    background = X_test[:min(100, len(X_test))]

    # Build a mu-only model for SHAP (SHAP needs scalar output)
    mu_model = Model(inputs=model.input, outputs=model.get_layer("mu").output)

    explainer = shap.DeepExplainer(mu_model, background)
    shap_values = explainer.shap_values(X_test[:200])  # cap at 200 for speed

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test[:200],
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance (NYC Tourism Index)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    blob = client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_file(buf, content_type="image/png")
    plt.close()

    # Return feature importance as dict
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap.tolist()))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def run_model(request):
    """HTTP Cloud Function entry point."""
    client = storage.Client()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("preds/listings_master.csv")
    try:
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return f"ERROR: Could not load master CSV: {e}", 500

    print(f"[Model] Loaded {len(df)} rows, {df.columns.tolist()}")

    # Filter to rows with complete target and sufficient completeness
    df = df.dropna(subset=[TARGET_COL])
    df = df[df.get("data_completeness", 1) >= 0.67]  # require at least 2/3 signals

    if len(df) < 30:
        return f"ERROR: Not enough data ({len(df)} rows). Need at least 30.", 500

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    # Only keep available feature columns
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df_model = df[available_features + [TARGET_COL]].copy()

    # Fill missing features with column median (simple imputation)
    for col in available_features:
        df_model[col] = df_model[col].fillna(df_model[col].median())

    X = df_model[available_features].values.astype(np.float32)
    y = df_model[TARGET_COL].values.astype(np.float32)

    # Train/test split (time-ordered — no shuffling for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale X (StandardScaler — matches class)
    x_scaler = StandardScaler()
    X_train_sc = x_scaler.fit_transform(X_train)
    X_test_sc = x_scaler.transform(X_test)

    # Scale Y to [0,1] (MinMaxScaler — matches class)
    y_scaler = MinMaxScaler()
    y_train_sc = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_sc = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"[Model] Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(available_features)}")

    # ── 3. Build + train model ─────────────────────────────────────────────────
    model = build_model(n_features=len(available_features))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=1),
    ]

    history = model.fit(
        X_train_sc, y_train_sc,
        validation_split=0.2,
        epochs=200,
        batch_size=min(32, len(X_train) // 4),
        callbacks=callbacks,
        verbose=1,
    )

    # ── 4. MC Dropout predictions ─────────────────────────────────────────────
    print("[Model] Running MC Dropout (50 passes)...")
    train_preds = mc_predict(model, X_train_sc, n_passes=50)
    test_preds = mc_predict(model, X_test_sc, n_passes=50)

    # Inverse transform predictions back to original scale
    def inv_transform(arr):
        return y_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    test_mu = inv_transform(test_preds["mu"])
    # Scale uncertainty back (multiply std by scaler range)
    y_range = y_scaler.data_range_[0] if hasattr(y_scaler, "data_range_") else 100
    test_aleatoric = test_preds["aleatoric_std"] * y_range
    test_epistemic = test_preds["epistemic_std"] * y_range
    test_total = test_preds["total_std"] * y_range

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    mae = np.abs(test_mu - y_test).mean()
    rmse = np.sqrt(np.square(test_mu - y_test).mean())
    coverage = compute_coverage(y_test, test_mu, test_total)

    metrics = {
        "test_mae": round(float(mae), 3),
        "test_rmse": round(float(rmse), 3),
        "coverage_95pct": round(float(coverage), 3),
        "mean_aleatoric_std": round(float(test_aleatoric.mean()), 3),
        "mean_epistemic_std": round(float(test_epistemic.mean()), 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(available_features),
        "features_used": available_features,
    }
    print(f"[Model] Metrics: {metrics}")

    # ── 6. Save prediction CSV ─────────────────────────────────────────────────
    test_dates = df["date"].iloc[split_idx:].values if "date" in df.columns else np.arange(len(X_test))
    pred_df = pd.DataFrame({
        "date": test_dates,
        "actual_tourism_index": y_test,
        "predicted_mu": test_mu,
        "aleatoric_std": test_aleatoric,
        "epistemic_std": test_epistemic,
        "total_std": test_total,
        "lower_95": test_mu - 1.96 * test_total,
        "upper_95": test_mu + 1.96 * test_total,
        "residual": y_test - test_mu,
        "in_interval": ((y_test >= test_mu - 1.96 * test_total) &
                        (y_test <= test_mu + 1.96 * test_total)).astype(int),
    })

    pred_blob = bucket.blob("preds/model_predictions.csv")
    pred_blob.upload_from_string(pred_df.to_csv(index=False), content_type="text/csv")

    # Save metrics
    metrics_blob = bucket.blob("preds/model_metrics.json")
    metrics_blob.upload_from_string(json.dumps(metrics, indent=2), content_type="application/json")

    # ── 7. Coverage plot ───────────────────────────────────────────────────────
    plot_coverage(y_test, test_mu, test_total, y_scaler, client, BUCKET_NAME,
                  "preds/coverage_analysis.png")

    # ── 8. SHAP ───────────────────────────────────────────────────────────────
    try:
        importance = plot_shap_summary(model, X_test_sc, available_features,
                                       client, BUCKET_NAME, "preds/shap_summary.png")
        imp_df = pd.DataFrame(list(importance.items()), columns=["feature", "mean_abs_shap"])
        imp_blob = bucket.blob("preds/feature_importance.csv")
        imp_blob.upload_from_string(imp_df.to_csv(index=False), content_type="text/csv")
        print(f"[Model] Top feature: {list(importance.keys())[0]}")
    except Exception as e:
        print(f"[Model] SHAP failed (non-fatal): {e}")

    return f"OK: MAE={mae:.2f}, RMSE={rmse:.2f}, Coverage={coverage:.1%}", 200
