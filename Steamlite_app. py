"""
Streamlit front-end for the ETL data pipeline.
Same pipeline logic as data_pipeline.py, wrapped in a Streamlit UI so it can
be deployed as a live web app on Streamlit Community Cloud.
Run locally: streamlit run streamlit_app.py
"""

import time
import warnings
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ENDPOINTS = [
    "/api/products", "/api/cart", "/api/checkout", "/api/login",
    "/api/search", "/api/users", "/api/orders", "/static/logo.png",
    "/api/reviews", "/health",
]
ENDPOINT_WEIGHTS = [0.22, 0.15, 0.08, 0.12, 0.18, 0.05, 0.09, 0.03, 0.06, 0.02]
METHODS = ["GET", "POST", "PUT", "DELETE"]
METHOD_WEIGHTS = [0.65, 0.22, 0.09, 0.04]
STATUS_CODES = [200, 201, 301, 400, 401, 403, 404, 500, 502]
STATUS_WEIGHTS = [0.72, 0.05, 0.03, 0.05, 0.03, 0.02, 0.06, 0.03, 0.01]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    "Mozilla/5.0 (Android 14; Mobile)",
]


# ---------------------------------------------------------------------------
# PIPELINE FUNCTIONS (identical logic to the CLI version)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_api_data() -> pd.DataFrame:
    url = "https://jsonplaceholder.typicode.com/users"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        raw = resp.json()
        return pd.DataFrame([
            {"user_id": u["id"], "user_name": u["name"],
             "city": u["address"]["city"], "company": u["company"]["name"]}
            for u in raw
        ])
    except Exception:
        fallback_names = [
            "Aarav Shah", "Diya Mehta", "Kabir Rao", "Isha Nair", "Vihaan Gupta",
            "Ananya Iyer", "Reyansh Joshi", "Myra Kapoor", "Arjun Verma", "Saanvi Pillai",
        ]
        cities = ["Mumbai", "Pune", "Bengaluru", "Delhi", "Chennai",
                  "Hyderabad", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
        companies = [f"{n.split()[0]} Traders" for n in fallback_names]
        return pd.DataFrame({"user_id": range(1, 11), "user_name": fallback_names,
                              "city": cities, "company": companies})


@st.cache_data(show_spinner=False)
def generate_apache_logs(n_rows: int) -> pd.DataFrame:
    start = datetime.now() - timedelta(days=30)
    timestamps = start + pd.to_timedelta(np.random.randint(0, 30 * 24 * 3600, n_rows), unit="s")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "user_id": np.random.randint(1, 11, n_rows),
        "endpoint": np.random.choice(ENDPOINTS, n_rows, p=ENDPOINT_WEIGHTS),
        "method": np.random.choice(METHODS, n_rows, p=METHOD_WEIGHTS),
        "status_code": np.random.choice(STATUS_CODES, n_rows, p=STATUS_WEIGHTS),
        "response_time_ms": np.random.lognormal(mean=4.2, sigma=0.6, size=n_rows),
        "user_agent": np.random.choice(USER_AGENTS, n_rows),
        "bytes_sent": np.random.randint(200, 50_000, n_rows),
    })
    missing_idx = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df.loc[missing_idx, "response_time_ms"] = np.nan
    outlier_idx = np.random.choice(n_rows, size=int(n_rows * 0.002), replace=False)
    df.loc[outlier_idx, "response_time_ms"] *= 50
    bad_status_idx = np.random.choice(n_rows, size=int(n_rows * 0.001), replace=False)
    df.loc[bad_status_idx, "status_code"] = -1
    dup_idx = np.random.choice(n_rows, size=int(n_rows * 0.002), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["status_code"].isin(set(STATUS_CODES))]
    df["response_time_ms"] = df.groupby("endpoint")["response_time_ms"] \
                                .transform(lambda s: s.fillna(s.median()))
    q1, q3 = df["response_time_ms"].quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)
    df["response_time_ms"] = df["response_time_ms"].clip(upper=upper)
    df["city"] = df["city"].fillna("Unknown")
    df["company"] = df["company"].fillna("Unknown")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_error"] = df["status_code"] >= 400
    return df


def aggregate_metrics(df: pd.DataFrame) -> Dict[str, object]:
    per_endpoint = df.groupby("endpoint").agg(
        total_requests=("endpoint", "size"),
        error_rate=("is_error", "mean"),
        avg_response_ms=("response_time_ms", "mean"),
    ).sort_values("total_requests", ascending=False).reset_index()
    per_hour = df.groupby("hour").agg(
        total_requests=("hour", "size"),
        avg_response_ms=("response_time_ms", "mean"),
    ).reset_index()
    hour_day_pivot = df.pivot_table(index="day_of_week", columns="hour",
                                     values="response_time_ms", aggfunc="mean")
    overall = {
        "total_requests": len(df),
        "overall_error_rate": df["is_error"].mean(),
        "avg_response_ms": df["response_time_ms"].mean(),
    }
    return {"per_endpoint": per_endpoint, "per_hour": per_hour,
            "hour_day_pivot": hour_day_pivot, "overall": overall}


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ETL Pipeline Demo", layout="wide")
st.title("📊 Real-Time Data Pipeline: Web Logs + Public API")
st.caption("Simulated Apache logs enriched with a live public API, cleaned, "
           "aggregated and visualised end-to-end.")

with st.sidebar:
    st.header("Pipeline Controls")
    n_rows = st.slider("Rows to simulate", 50_000, 1_000_000, 300_000, step=50_000,
                        help="Kept under 1M by default so it stays fast on the free cloud tier.")
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

if run_btn:
    t0 = time.time()
    with st.spinner("Extracting data (synthetic logs + live API)..."):
        logs = generate_apache_logs(n_rows)
        users = fetch_api_data()
        df = logs.merge(users, on="user_id", how="left")

    with st.spinner("Cleaning and engineering features..."):
        df = clean_data(df)
        df = feature_engineering(df)

    metrics = aggregate_metrics(df)
    overall = metrics["overall"]

    st.success(f"Pipeline finished in {time.time() - t0:.1f}s on {len(df):,} cleaned rows.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Requests", f"{overall['total_requests']:,}")
    c2.metric("Error Rate", f"{overall['overall_error_rate']:.2%}")
    c3.metric("Avg Response Time", f"{overall['avg_response_ms']:.1f} ms")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Requests per Hour", "Response Time by Endpoint", "Hour × Day Heatmap", "Raw Table"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=metrics["per_hour"], x="hour", y="total_requests", marker="o", ax=ax)
        ax.set_title("Total Requests by Hour of Day")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        data = metrics["per_endpoint"].sort_values("avg_response_ms", ascending=False)
        sns.barplot(data=data, x="avg_response_ms", y="endpoint", color="indianred", ax=ax)
        ax.set_title("Average Response Time by Endpoint (ms)")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(metrics["hour_day_pivot"], cmap="magma", cbar_kws={"label": "Avg ms"}, ax=ax)
        ax.set_title("Response Time Heatmap: Day of Week vs Hour")
        st.pyplot(fig)

    with tab4:
        st.dataframe(metrics["per_endpoint"], use_container_width=True)
        st.download_button(
            "⬇ Download cleaned_data.csv",
            metrics["per_endpoint"].to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
else:
    st.info("Set the row count in the sidebar and click **Run Pipeline** to start.")
