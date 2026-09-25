"""
Fraud Watch — Live Stream Monitor (Streamlit)
==============================================
Online SGD classifier (partial_fit) + DDM concept-drift detector + p99 latency,
same logic as the notebook's Part 2, wrapped in a live Streamlit UI.

Run:
    pip install streamlit pandas numpy scikit-learn altair
    streamlit run fraud_streamlit_app.py

Input:
    Upload an already-engineered CSV with numeric feature columns + an
    'is_fraud' column (e.g. an export of X_train/X_test + y from your
    notebook). If nothing is uploaded, a synthetic demo stream is used
    so the app still runs standalone.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Watch — Live Stream Monitor", layout="wide")


# ---------------------------------------------------------------- DDM ----
class DDM:
    """Drift Detection Method (Gama et al., 2004), simplified.
    Tracks the online error rate; flags 'warning'/'drift' once it climbs
    significantly above its historical minimum."""

    def __init__(self, warn_z=2.0, drift_z=3.0, min_n=30):
        self.warn_z, self.drift_z, self.min_n = warn_z, drift_z, min_n
        self.reset()

    def reset(self):
        self.n, self.p, self.s = 0, 0.0, 0.0
        self.p_min, self.s_min = float("inf"), float("inf")

    def update(self, error: int) -> str:
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n) if self.n else 0.0
        if self.n < self.min_n:
            return "ok"
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min, self.s_min = self.p, self.s
        if self.p + self.s > self.p_min + self.drift_z * self.s_min:
            self.reset()
            return "drift"
        if self.p + self.s > self.p_min + self.warn_z * self.s_min:
            return "warning"
        return "ok"


# ------------------------------------------------------------ sidebar ----
st.sidebar.header("Stream controls")
uploaded = st.sidebar.file_uploader(
    "Engineered CSV (numeric features + 'is_fraud')", type="csv"
)
warmup = st.sidebar.number_input("Warm-up rows", 100, 5000, 500, step=100)
limit = st.sidebar.number_input("Stream length (rows)", 500, 200_000, 5_000, step=500)
delay_ms = st.sidebar.slider("Playback delay (ms/event)", 0, 300, 20)
window = st.sidebar.number_input("PR-AUC / retrain window", 200, 20_000, 2_000, step=200)
run_btn = st.sidebar.button("▶ Start stream")


@st.cache_data
def load_demo(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(int)  # ~3% for demo visibility
    amt_dev = np.where(y == 1, rng.normal(180, 90, n), rng.normal(0, 30, n))
    dist = np.abs(np.where(y == 1, rng.normal(65, 40, n), rng.normal(8, 6, n)))
    X = pd.DataFrame(
        {
            "amt_deviation": amt_dev,
            "distance": dist,
            "f3": rng.normal(0, 1, n),
            "f4": rng.normal(0, 1, n),
        }
    )
    return X, y


if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "is_fraud" not in df.columns:
        st.error("CSV must contain an 'is_fraud' column.")
        st.stop()
    y_all = df["is_fraud"].to_numpy()
    X_all = df.drop(columns=["is_fraud"]).select_dtypes(include=[np.number])
else:
    X_all, y_all = load_demo()
    st.sidebar.info("No file uploaded — using synthetic demo data.")

scaler = StandardScaler().fit(X_all.iloc[: int(warmup)])
X_scaled = scaler.transform(X_all)

# -------------------------------------------------------------- layout ----
st.title("🚨 Fraud Watch — Live Stream Monitor")
c1, c2, c3, c4 = st.columns(4)
count_ph, fraud_ph, auc_ph, p99_ph = c1.empty(), c2.empty(), c3.empty(), c4.empty()
alert_ph = st.empty()
col_feed, col_chart = st.columns([1, 1.3])
feed_ph = col_feed.empty()
chart_ph = col_chart.empty()

if run_btn:
    end = min(len(y_all), warmup + limit)
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-4, learning_rate="optimal", random_state=42
    )
    model.partial_fit(X_scaled[:warmup], y_all[:warmup], classes=[0, 1])

    ddm = DDM()
    win_true, win_prob = [], []
    latencies = []
    pr_events, pr_curve = [], []
    drift_events = []
    feed_rows = []
    running_n, running_fraud = int(warmup), int(y_all[:warmup].sum())
    fraud_count = 0
    pr_auc = None

    for i in range(int(warmup), end):
        x_i, y_i = X_scaled[i : i + 1], int(y_all[i])

        # ---- time the full inference + online-update cycle (p99 latency) ----
        t0 = time.perf_counter()
        proba = model.predict_proba(x_i)[0, 1]
        pred = int(proba >= 0.5)

        running_n += 1
        running_fraud += y_i
        fraud_rate = max(running_fraud / running_n, 1e-4)
        sw = np.array([1.0 / fraud_rate if y_i else 1.0 / (1 - fraud_rate)])
        model.partial_fit(x_i, [y_i], sample_weight=sw)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

        status = ddm.update(int(pred != y_i))
        if status == "drift":
            recent = slice(max(0, i - window), i)
            model = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=1e-4,
                learning_rate="optimal", random_state=42,
            )
            model.partial_fit(X_scaled[recent], y_all[recent], classes=[0, 1])
            drift_events.append(i - warmup)

        win_true.append(y_i)
        win_prob.append(proba)
        if len(win_true) > window:
            win_true.pop(0)
            win_prob.pop(0)
        if len(set(win_true)) > 1:
            pr_auc = average_precision_score(win_true, win_prob)
            pr_events.append(i - warmup)
            pr_curve.append(pr_auc)

        if y_i:
            fraud_count += 1
            alert_ph.warning(f"⚠ Fraud alert — event #{i}, risk score {proba:.0%}")
        elif status == "drift":
            alert_ph.info(f"↻ Concept drift detected at event #{i} — model retrained on recent window")

        feed_rows.insert(
            0, {"event": i, "risk": f"{proba:.0%}", "fraud": bool(y_i), "flag": status}
        )
        feed_rows = feed_rows[:12]

        # throttle UI redraws for playback speed / performance
        if i % 5 == 0 or y_i or status != "ok" or i == end - 1:
            p99 = np.percentile(latencies, 99) if latencies else 0.0
            count_ph.metric("Events streamed", i - warmup + 1)
            fraud_ph.metric("Fraud caught", fraud_count)
            auc_ph.metric("PR-AUC (windowed)", f"{pr_auc:.3f}" if pr_auc is not None else "–")
            p99_ph.metric("p99 latency", f"{p99:.2f} ms")

            feed_ph.dataframe(pd.DataFrame(feed_rows), use_container_width=True, hide_index=True)

            if pr_curve:
                curve_df = pd.DataFrame({"event": pr_events, "pr_auc": pr_curve})
                line = (
                    alt.Chart(curve_df)
                    .mark_line(color="#5fb3a3")
                    .encode(x="event", y=alt.Y("pr_auc", scale=alt.Scale(domain=[0, 1])))
                )
                if drift_events:
                    rules = (
                        alt.Chart(pd.DataFrame({"event": drift_events}))
                        .mark_rule(color="#e0a458", strokeDash=[4, 4])
                        .encode(x="event")
                    )
                    chart_ph.altair_chart((line + rules).properties(height=280), use_container_width=True)
                else:
                    chart_ph.altair_chart(line.properties(height=280), use_container_width=True)

        if delay_ms:
            time.sleep(delay_ms / 1000)

    final_p99 = np.percentile(latencies, 99) if latencies else 0.0
    pr_auc_str = f"{pr_auc:.3f}" if pr_auc is not None else "n/a"
    st.success(
        f"Stream complete — {end - warmup:,} events | "
        f"final PR-AUC: {pr_auc_str} | "
        f"p99 latency: {final_p99:.2f} ms | drift events: {len(drift_events)}"
    )
else:
    st.info("Set your parameters in the sidebar, then click ▶ Start stream.")
