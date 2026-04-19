import os
import sys
import time
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

try:
    import serial
except Exception:
    serial = None


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.predict import BatteryPredictor  # noqa: E402


st.set_page_config(page_title="Battery Digital Twin", layout="wide")
st.title("Battery Digital Twin Web Dashboard")

st.markdown(
    """
    <style>
    .status-card {
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0 14px 0;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .status-green {
        background: rgba(34, 197, 94, 0.14);
        border-color: rgba(34, 197, 94, 0.45);
    }
    .status-yellow {
        background: rgba(245, 158, 11, 0.16);
        border-color: rgba(245, 158, 11, 0.50);
    }
    .status-red {
        background: rgba(239, 68, 68, 0.16);
        border-color: rgba(239, 68, 68, 0.50);
    }
    .status-title {
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .status-tip {
        font-size: 0.93rem;
        opacity: 0.95;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_soh_status(soh_value):
    if soh_value >= 80:
        return {
            "label": "GREEN - Healthy",
            "css": "status-green",
            "tip": "Battery health is good. Continue normal usage and avoid deep discharges.",
        }
    if soh_value >= 60:
        return {
            "label": "YELLOW - Moderate Degradation",
            "css": "status-yellow",
            "tip": "Health is dropping. Reduce heavy loads and monitor charging temperature.",
        }
    return {
        "label": "RED - Replace Soon",
        "css": "status-red",
        "tip": "Battery is highly degraded. Plan replacement and avoid high-current stress.",
    }


def render_status_card(title, status):
    st.markdown(
        f"""
        <div class=\"status-card {status['css']}\">
            <div class=\"status-title\">{title}: {status['label']}</div>
            <div class=\"status-tip\">{status['tip']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evaluate_faults(sample, soh, prev_sample=None):
    faults = []

    temp_c = float(sample["Temp(C)"])
    current_a = float(sample["Current(A)"])
    voltage_v = float(sample["Voltage(V)"])

    if temp_c >= 45:
        faults.append(("RED", "Overheat", f"Temperature is high at {temp_c:.1f} C.", "Reduce load and cool battery."))
    elif temp_c >= 38:
        faults.append(("YELLOW", "Rising Temperature", f"Temperature is elevated at {temp_c:.1f} C.", "Avoid heavy charging/discharging."))

    if current_a >= 1.5:
        faults.append(("RED", "Overcurrent", f"Current spike at {current_a:.2f} A.", "Cut non-essential load immediately."))
    elif current_a >= 1.1:
        faults.append(("YELLOW", "High Current", f"Current is high at {current_a:.2f} A.", "Reduce load if possible."))

    if prev_sample is not None:
        dv = float(prev_sample["Voltage(V)"]) - voltage_v
        if dv >= 0.12:
            faults.append(("RED", "Rapid Voltage Drop", f"Voltage dropped by {dv:.3f} V in one step.", "Inspect battery and connections."))
        elif dv >= 0.07:
            faults.append(("YELLOW", "Voltage Sag", f"Voltage drop trend: {dv:.3f} V.", "Monitor next samples closely."))

    if soh < 55:
        faults.append(("RED", "Low SOH", f"Predicted SOH is critical at {soh:.2f}%.", "Plan battery replacement soon."))
    elif soh < 70:
        faults.append(("YELLOW", "SOH Degradation", f"Predicted SOH is moderate at {soh:.2f}%.", "Use conservative cycling."))

    if not faults:
        faults.append(("GREEN", "Normal", "No active fault conditions detected.", "Continue normal operation."))

    ts = datetime.now().strftime("%H:%M:%S")
    alert_rows = []
    for severity, ftype, message, action in faults:
        row = {
            "time": ts,
            "severity": severity,
            "type": ftype,
            "message": message,
            "action": action,
        }
        alert_rows.append(row)
        st.session_state.alerts.appendleft(row)

    return alert_rows


def highest_severity(alert_rows):
    if any(a["severity"] == "RED" for a in alert_rows):
        return "RED"
    if any(a["severity"] == "YELLOW" for a in alert_rows):
        return "YELLOW"
    return "GREEN"


def render_alert_banner(alert_rows):
    if not alert_rows:
        return

    sev = highest_severity(alert_rows)
    red_count = sum(1 for a in alert_rows if a["severity"] == "RED")
    yellow_count = sum(1 for a in alert_rows if a["severity"] == "YELLOW")

    if sev == "RED":
        st.error(f"Critical Alerts: {red_count} red, {yellow_count} yellow. Immediate attention required.")
    elif sev == "YELLOW":
        st.warning(f"Warnings Active: {yellow_count} yellow alerts. Monitor battery conditions.")
    else:
        st.success("System Normal: No active faults detected in latest sample.")


def init_state(max_points=200):
    if "history" not in st.session_state:
        st.session_state.history = {
            "time": deque(maxlen=max_points),
            "voltage": deque(maxlen=max_points),
            "current": deque(maxlen=max_points),
            "temp": deque(maxlen=max_points),
            "soc": deque(maxlen=max_points),
            "energy": deque(maxlen=max_points),
            "power": deque(maxlen=max_points),
            "soh": deque(maxlen=max_points),
        }

    if "sim_step" not in st.session_state:
        st.session_state.sim_step = 0
    if "sim_soc" not in st.session_state:
        st.session_state.sim_soc = 100.0
    if "sim_energy" not in st.session_state:
        st.session_state.sim_energy = 0.0
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "replay_df" not in st.session_state:
        st.session_state.replay_df = None
    if "replay_idx" not in st.session_state:
        st.session_state.replay_idx = 0
    if "predictor" not in st.session_state:
        st.session_state.predictor = BatteryPredictor("battery_soh_model.pkl")
    if "alerts" not in st.session_state:
        st.session_state.alerts = deque(maxlen=200)
    if "active_alerts" not in st.session_state:
        st.session_state.active_alerts = []
    if "last_sample" not in st.session_state:
        st.session_state.last_sample = None


def next_synthetic_sample(interval_sec=1.0):
    st.session_state.sim_step += 1
    step = st.session_state.sim_step

    discharge_wave = 0.03 * np.sin(step / 6.0)
    load_wave = 0.08 * np.sin(step / 8.0)
    temp_wave = 1.5 * np.sin(step / 10.0)

    current = max(0.12, 0.7 + load_wave)
    voltage = max(3.0, 4.2 - (step * 0.0025) - discharge_wave)
    temp = max(20.0, 26.0 + temp_wave + (current * 2.5))
    st.session_state.sim_soc = max(0.0, st.session_state.sim_soc - (current * 0.04))
    st.session_state.sim_energy = max(
        0.0,
        st.session_state.sim_energy + (voltage * current * (interval_sec / 3600.0)),
    )

    return {
        "Time(s)": time.time() - st.session_state.start_time,
        "Voltage(V)": voltage,
        "Current(A)": current,
        "Temp(C)": temp,
        "SoC(%)": st.session_state.sim_soc,
        "Energy(Wh)": st.session_state.sim_energy,
    }


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


def get_personalization_status(min_samples=30):
    replay_df = load_replay_data("log.csv")
    sample_count = 0 if replay_df is None else len(replay_df)
    is_active = sample_count >= min_samples
    return is_active, sample_count, min_samples


def save_history_as_log_csv(min_samples=30):
    hist = st.session_state.history
    n = len(hist["time"])
    if n < min_samples:
        return False, n

    df = pd.DataFrame(
        {
            "Time(s)": list(hist["time"]),
            "Voltage(V)": list(hist["voltage"]),
            "Current(A)": list(hist["current"]),
            "Temp(C)": list(hist["temp"]),
            "SoC(%)": list(hist["soc"]),
            "Energy(Wh)": list(hist["energy"]),
        }
    )
    (ROOT / "log.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    return True, n


@st.cache_resource(show_spinner=False)
def train_personalized_model(log_path: str, log_mtime: float):
    del log_mtime  # cache key invalidation helper

    file_path = Path(log_path)
    if not file_path.exists():
        return None, 0

    df = load_replay_data(file_path.name)
    if df is None or len(df) < 30:
        return None, 0

    feature_cols = ["Voltage(V)", "Current(A)", "Temp(C)", "SoC(%)", "Energy(Wh)"]

    # Derive SOH target from logged discharge progression (same capacity baseline as training script)
    ah_used = df["Energy(Wh)"] / df["Voltage(V)"].replace(0, np.nan)
    soh_target = np.clip(100 - ((ah_used / 2.2) * 100), 0, 100)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.bfill().ffill().fillna(0.0)
    y = pd.Series(soh_target).bfill().ffill().fillna(0.0)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    return model, len(df)


def next_replay_sample():
    df = st.session_state.replay_df
    if df is None or df.empty:
        return None

    idx = st.session_state.replay_idx % len(df)
    st.session_state.replay_idx += 1
    row = df.iloc[idx]

    return {
        "Time(s)": float(row["Time(s)"]),
        "Voltage(V)": float(row["Voltage(V)"]),
        "Current(A)": float(row["Current(A)"]),
        "Temp(C)": float(row["Temp(C)"]),
        "SoC(%)": float(row["SoC(%)"]),
        "Energy(Wh)": float(row["Energy(Wh)"]),
    }


def next_live_sample(com_port, baud_rate):
    if serial is None:
        return None

    try:
        with serial.Serial(com_port, baud_rate, timeout=1) as ser:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                return None
            values = [float(x) for x in parts]
            return {
                "Time(s)": values[0],
                "Voltage(V)": values[1],
                "Current(A)": values[2],
                "Temp(C)": values[3],
                "SoC(%)": values[4],
                "Energy(Wh)": values[5],
            }
    except Exception:
        return None


def append_history(sample, soh):
    hist = st.session_state.history
    hist["time"].append(sample["Time(s)"])
    hist["voltage"].append(sample["Voltage(V)"])
    hist["current"].append(sample["Current(A)"])
    hist["temp"].append(sample["Temp(C)"])
    hist["soc"].append(sample["SoC(%)"])
    hist["energy"].append(sample["Energy(Wh)"])
    hist["power"].append(sample["Voltage(V)"] * sample["Current(A)"])
    hist["soh"].append(soh)


def draw_dashboard():
    hist = st.session_state.history

    if len(hist["time"]) == 0:
        st.info("No samples yet. Click 'Step Once' or enable Auto Run.")
        return

    latest = {
        "V": hist["voltage"][-1],
        "I": hist["current"][-1],
        "T": hist["temp"][-1],
        "SoC": hist["soc"][-1],
        "P": hist["power"][-1],
        "SOH": hist["soh"][-1],
    }

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Voltage", f"{latest['V']:.3f} V")
    c2.metric("Current", f"{latest['I']:.3f} A")
    c3.metric("Temp", f"{latest['T']:.2f} C")
    c4.metric("SoC", f"{latest['SoC']:.2f} %")
    c5.metric("Power", f"{latest['P']:.3f} W")
    c6.metric("SOH (ML)", f"{latest['SOH']:.2f} %")

    render_alert_banner(st.session_state.active_alerts)

    status = get_soh_status(latest["SOH"])
    render_status_card("Live Battery Status", status)

    tip_col1, tip_col2, tip_col3 = st.columns(3)
    tip_col1.info("Tip: Keep temperature mostly below 40 C to slow aging.")
    tip_col2.info("Tip: Avoid frequent 0% to 100% full cycles.")
    tip_col3.info("Tip: If status is red, prioritize battery replacement planning.")

    df = pd.DataFrame(
        {
            "Time": list(hist["time"]),
            "Voltage": list(hist["voltage"]),
            "Current": list(hist["current"]),
            "Temp": list(hist["temp"]),
            "SoC": list(hist["soc"]),
            "Power": list(hist["power"]),
            "SOH": list(hist["soh"]),
        }
    )

    st.subheader("Live Graphs")

    top_row = st.columns(3)
    with top_row[0]:
        st.caption("Voltage")
        st.line_chart(df.set_index("Time")[["Voltage"]], color=["#00E5CC"])
    with top_row[1]:
        st.caption("Current")
        st.line_chart(df.set_index("Time")[["Current"]], color=["#FF4D6D"])
    with top_row[2]:
        st.caption("Temperature")
        st.line_chart(df.set_index("Time")[["Temp"]], color=["#FFB347"])

    mid_row = st.columns(3)
    with mid_row[0]:
        st.caption("Power")
        st.line_chart(df.set_index("Time")[["Power"]], color=["#BD93F9"])
    with mid_row[1]:
        st.caption("State of Charge")
        st.line_chart(df.set_index("Time")[["SoC"]], color=["#8BE9FD"])
    with mid_row[2]:
        st.caption("State of Health (ML)")
        st.line_chart(df.set_index("Time")[["SOH"]], color=["#F1FA8C"])

    st.subheader("Recent Samples")
    st.dataframe(df.tail(10), width="stretch")

    st.subheader("Alert History")
    if len(st.session_state.alerts) == 0:
        st.info("No alerts yet.")
    else:
        alerts_df = pd.DataFrame(list(st.session_state.alerts)[:20])
        st.dataframe(alerts_df[["time", "severity", "type", "message", "action"]], width="stretch")


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


def draw_manual_prediction_view():
    st.subheader("Manual SOH Prediction")
    st.caption("AI prediction from simple daily-life inputs.")

    if "manual_result" not in st.session_state:
        st.session_state.manual_result = None

    def render_manual_card(title, value, kind="normal"):
        if kind == "invalid":
            st.markdown(
                f"""
                <div style="border:1px solid rgba(239,68,68,0.65);background:rgba(239,68,68,0.14);padding:16px;border-radius:12px;">
                    <div style="font-size:0.95rem;font-weight:700;margin-bottom:6px;color:#FCA5A5;">{title}</div>
                    <div style="font-size:1.2rem;font-weight:700;color:#F87171;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        st.metric(title, value)

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
        st.info("Using base AI model only (.pkl).")

    c1, c2 = st.columns(2)
    years_used = c1.number_input("Battery used for how many years?", min_value=0.0, max_value=12.0, value=2.0, step=0.1)
    voltage = c2.number_input(
        "Current battery voltage (V)",
        min_value=2.0,
        max_value=4.3,
        value=3.70,
        step=0.01,
        help="Use a realistic battery voltage, typically between 2.0 V and 4.2 V.",
    )

    c3, c4 = st.columns(2)
    usage_type = c3.selectbox("Daily usage", ["Light", "Normal", "Heavy"], index=1)
    ambient_temp = c4.number_input("Ambient temperature (C)", min_value=-10.0, max_value=55.0, value=30.0, step=0.5)

    voltage_ok = 2.0 <= voltage <= 4.2
    if not voltage_ok:
        st.session_state.manual_result = None
        render_manual_card("Invalid Input", "Voltage must be between 3.0 V and 4.2 V", kind="invalid")
        render_manual_card("Invalid Input", "Voltage must be between 2.0 V and 4.2 V", kind="invalid")
        st.error("Enter a realistic battery voltage. Predictions are hidden until the value is valid.")
        return

    predict_clicked = st.button("Predict Battery Health")

    if not predict_clicked:
        manual_result = st.session_state.manual_result
        if manual_result is None:
            return
        render_manual_card("Predicted SOH", f"{manual_result['final_soh']:.2f} %")
        render_manual_card("Base Model SOH", f"{manual_result['base_model_soh']:.2f} %")
        if manual_result["personalized_soh"] is not None:
            render_manual_card("Log-data AI SOH", f"{manual_result['personalized_soh']:.2f} %")
        else:
            render_manual_card("Age-based Prior", f"{manual_result['prior_soh']:.2f} %")

        st.caption(
            f"Derived inputs -> Current: {manual_result['est_current']:.2f} A, Temp: {manual_result['est_temp']:.1f} C, "
            f"SoC(from voltage): {manual_result['est_soc']:.1f} %, Energy: {manual_result['est_energy']:.3f} Wh"
        )

        if manual_result["final_soh"] >= 90:
            st.success("Battery health status: Excellent")
        elif manual_result["final_soh"] >= 80:
            st.warning("Battery health status: Moderate")
        else:
            st.error("Battery health status: Degraded")

        st.info("This is an AI estimate. For highest confidence, use live sensor mode and collect more logger data.")
        return

    usage_current_map = {
        "Light": 0.35,
        "Normal": 0.75,
        "Heavy": 1.20,
    }
    usage_age_penalty_map = {
        "Light": 0.6,
        "Normal": 1.0,
        "Heavy": 1.5,
    }

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

    predictor = st.session_state.predictor
    base_model_soh = predictor.predict_from_dict(model_input)

    personalized_soh = None
    if enable_personalized and personal_model is not None:
        personalized_soh = float(personal_model.predict(pd.DataFrame([model_input]))[0])

    if personalized_soh is not None:
        final_soh = float(np.clip((0.55 * personalized_soh) + (0.25 * base_model_soh) + (0.20 * prior_soh), 0.0, 100.0))
    else:
        final_soh = float(np.clip((0.65 * base_model_soh) + (0.35 * prior_soh), 0.0, 100.0))

    st.session_state.manual_result = {
        "final_soh": final_soh,
        "base_model_soh": base_model_soh,
        "personalized_soh": personalized_soh,
        "prior_soh": prior_soh,
        "est_current": est_current,
        "est_temp": est_temp,
        "est_soc": est_soc,
        "est_energy": est_energy,
    }

    render_manual_card("Predicted SOH", f"{final_soh:.2f} %")
    render_manual_card("Base Model SOH", f"{base_model_soh:.2f} %")
    if personalized_soh is not None:
        render_manual_card("Log-data AI SOH", f"{personalized_soh:.2f} %")
    else:
        render_manual_card("Age-based Prior", f"{prior_soh:.2f} %")

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

    st.info("This is an AI estimate. For highest confidence, use live sensor mode and collect more logger data.")


init_state()

st.sidebar.header("Controls")
nav_page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Manual Prediction"],
    index=0,
)

if nav_page == "Manual Prediction":
    draw_manual_prediction_view()
    st.stop()


mode = st.sidebar.selectbox("Data Source", ["Synthetic", "Replay log.csv", "Live Serial"], index=0)
interval_sec = st.sidebar.slider("Update Interval (s)", 0.2, 3.0, 1.0, 0.1)
max_points = st.sidebar.slider("Max Points", 50, 500, 200, 10)

if max_points != st.session_state.history["time"].maxlen:
    old = st.session_state.history
    st.session_state.history = {
        k: deque(list(v), maxlen=max_points) for k, v in old.items()
    }

com_port = st.sidebar.text_input("COM Port", value=os.environ.get("ESP32_PORT", "COM10"))
baud_rate = st.sidebar.number_input("Baud Rate", min_value=9600, max_value=921600, value=115200, step=100)

if mode == "Replay log.csv" and st.session_state.replay_df is None:
    st.session_state.replay_df = load_replay_data("log.csv")
    if st.session_state.replay_df is None:
        st.warning("log.csv not found or empty. Run backend/logger.py to collect real data first.")

col_a, col_b, col_c = st.columns(3)
step_once = col_a.button("Step Once")
auto_run = col_b.checkbox("Auto Run", value=True)
reset = col_c.button("Reset")

if reset:
    st.session_state.clear()
    st.rerun()

sample = None
if step_once or auto_run:
    if mode == "Synthetic":
        sample = next_synthetic_sample(interval_sec)
    elif mode == "Replay log.csv":
        sample = next_replay_sample()
        if sample is None:
            st.warning("Replay has no data. Switch mode or collect log.csv.")
    else:
        sample = next_live_sample(com_port, int(baud_rate))
        if sample is None:
            st.warning("No serial sample received. Check COM port and device output format.")

if sample is not None:
    predictor = st.session_state.predictor
    model_input = {
        "Voltage(V)": sample["Voltage(V)"],
        "Current(A)": sample["Current(A)"],
        "Temp(C)": sample["Temp(C)"],
        "SoC(%)": sample["SoC(%)"],
        "Energy(Wh)": sample["Energy(Wh)"],
    }
    soh = predictor.predict_from_dict(model_input)
    st.session_state.active_alerts = evaluate_faults(sample, soh, st.session_state.last_sample)
    st.session_state.last_sample = sample.copy()
    append_history(sample, soh)

draw_dashboard()

if auto_run:
    time.sleep(interval_sec)
    st.rerun()
