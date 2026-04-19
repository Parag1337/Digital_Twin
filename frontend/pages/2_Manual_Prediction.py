import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.predict import BatteryPredictor  # noqa: E402


st.set_page_config(page_title="Manual SOH Prediction", layout="wide")
st.title("Manual SOH Prediction")
st.caption("AI prediction using simple daily-life inputs + optional personalization from logger data.")


@st.cache_data(show_spinner=False)
def load_replay_data(path="log.csv"):
    file_path = ROOT / path
    if not file_path.exists():
        return None

    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "time" in line.lower():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    if not rows:
        return None

    return pd.DataFrame(
        rows,
        columns=["Time(s)", "Voltage(V)", "Current(A)", "Temp(C)", "SoC(%)", "Energy(Wh)"],
    )


@st.cache_resource(show_spinner=False)
def get_base_predictor():
    return BatteryPredictor("battery_soh_model.pkl")


@st.cache_resource(show_spinner=False)
def train_personalized_model(log_path: str, log_mtime: float):
    del log_mtime

    file_path = Path(log_path)
    if not file_path.exists():
        return None, 0

    df = load_replay_data(file_path.name)
    if df is None or len(df) < 30:
        return None, 0

    feature_cols = ["Voltage(V)", "Current(A)", "Temp(C)", "SoC(%)", "Energy(Wh)"]
    ah_used = df["Energy(Wh)"] / df["Voltage(V)"].replace(0, np.nan)
    soh_target = np.clip(100 - ((ah_used / 2.2) * 100), 0, 100)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.bfill().ffill().fillna(0.0)
    y = pd.Series(soh_target).bfill().ffill().fillna(0.0)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model, len(df)


def estimate_soc_from_voltage(v):
    if v <= 3.0:
        return 0.0
    if v >= 4.2:
        return 100.0
    points = [
        (3.0, 0.0),
        (3.3, 10.0),
        (3.5, 25.0),
        (3.7, 50.0),
        (3.85, 70.0),
        (4.0, 85.0),
        (4.1, 95.0),
        (4.2, 100.0),
    ]
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= v <= x2:
            ratio = (v - x1) / (x2 - x1)
            return y1 + ratio * (y2 - y1)
    return 50.0


def render_result_row(manual_result):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted SOH", f"{manual_result['final_soh']:.2f} %")
    col_b.metric("Base Model SOH", f"{manual_result['base_model_soh']:.2f} %")
    if manual_result["personalized_soh"] is not None:
        col_c.metric("Log-data AI SOH", f"{manual_result['personalized_soh']:.2f} %")
    else:
        col_c.metric("Age-based Prior", f"{manual_result['prior_soh']:.2f} %")


enable_personalized = st.toggle(
    "Enable Personalized mode (use logger data)",
    value=False,
    help="Off = use only trained .pkl model. On = blend with a model trained from log.csv.",
)

personal_model = None
if enable_personalized:
    log_path = ROOT / "log.csv"
    log_mtime = log_path.stat().st_mtime if log_path.exists() else 0.0
    personal_model, sample_count = train_personalized_model(str(log_path), log_mtime)
    if personal_model is not None:
        st.success(f"Personalized Model: ACTIVE ({sample_count} samples from log.csv)")
    else:
        st.warning("Personalized Model: INACTIVE (need at least 30 rows in log.csv)")
else:
    st.info("Using base AI model only (.pkl). Personalized mode is disabled.")

col1, col2 = st.columns(2)
years_used = col1.number_input("Battery used for how many years?", min_value=0.0, max_value=12.0, value=2.0, step=0.1)
voltage = col2.number_input(
    "Current battery voltage (V)",
    min_value=2.0,
    max_value=4.3,
    value=3.70,
    step=0.01,
    help="Use a realistic battery voltage, typically between 2.0 V and 4.2 V.",
)

col3, col4 = st.columns(2)
usage_type = col3.selectbox("Daily usage", ["Light", "Normal", "Heavy"], index=1)
ambient_temp = col4.number_input("Ambient temperature (C)", min_value=-10.0, max_value=55.0, value=30.0, step=0.5)

voltage_ok = 2.0 <= voltage <= 4.2
if not voltage_ok:
    st.session_state.manual_result = None
    st.markdown(
        "<div style='border:1px solid rgba(239,68,68,0.65);background:rgba(239,68,68,0.14);padding:14px 16px;border-radius:12px;color:#FCA5A5;'>"
        "Invalid Input: Voltage must be between 2.0 V and 4.2 V"
        "</div>",
        unsafe_allow_html=True,
    )
    st.error("Enter a realistic battery voltage. Predictions are hidden until the value is valid.")
    st.stop()

predict_clicked = st.button("Predict Battery Health")

if predict_clicked:
    usage_current_map = {"Light": 0.35, "Normal": 0.75, "Heavy": 1.20}
    usage_age_penalty_map = {"Light": 0.6, "Normal": 1.0, "Heavy": 1.5}

    est_current = usage_current_map[usage_type]
    soc_from_voltage = estimate_soc_from_voltage(voltage)
    age_penalty = years_used * 3.8 * usage_age_penalty_map[usage_type]
    prior_soh = float(np.clip(100 - age_penalty, 45, 100))

    est_soc = float(np.clip(soc_from_voltage, 3, 100))
    est_energy = max(0.01, (est_soc / 100.0) * 2.2 * 3.7)
    est_temp = float(np.clip(ambient_temp + (est_current * 2.5), -20, 80))

    model_input = {
        "Voltage(V)": float(voltage),
        "Current(A)": float(est_current),
        "Temp(C)": float(est_temp),
        "SoC(%)": float(est_soc),
        "Energy(Wh)": float(est_energy),
    }

    predictor = get_base_predictor()
    base_model_soh = predictor.predict_from_dict(model_input)

    personalized_soh = None
    if enable_personalized and personal_model is not None:
        personalized_soh = float(personal_model.predict(pd.DataFrame([model_input]))[0])

    if personalized_soh is not None:
        final_soh = float(np.clip((0.55 * personalized_soh) + (0.25 * base_model_soh) + (0.20 * prior_soh), 0.0, 100.0))
    else:
        final_soh = float(np.clip((0.65 * base_model_soh) + (0.35 * prior_soh), 0.0, 100.0))

    manual_result = {
        "final_soh": final_soh,
        "base_model_soh": base_model_soh,
        "personalized_soh": personalized_soh,
        "prior_soh": prior_soh,
        "est_current": est_current,
        "est_temp": est_temp,
        "est_soc": est_soc,
        "est_energy": est_energy,
    }

    render_result_row(manual_result)

    st.caption(
        f"Derived inputs -> Current: {est_current:.2f} A, Temp: {est_temp:.1f} C, "
        f"SoC(from voltage): {est_soc:.1f} %, Energy: {est_energy:.3f} Wh"
    )

    if final_soh >= 90:
        st.success("Battery health status: Excellent")
    elif final_soh >= 80:
        st.warning("Battery health status: Moderate")
    else:
        st.error("Battery health status: Degraded")

    with st.expander("Recommendation", expanded=True):
        if final_soh >= 90:
            st.success("Condition is healthy. Keep regular charging habits and periodic checks.")
        elif final_soh >= 80:
            st.warning("Condition is moderate. Reduce heavy usage and avoid overheating during charge.")
        else:
            st.error("Condition is critical. Consider replacing the battery soon for reliability.")

    st.info("This is an AI estimate. For highest confidence, use live sensor mode and collect more logger data.")
    
