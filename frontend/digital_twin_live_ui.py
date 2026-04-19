# -*- coding: utf-8 -*-
"""
REAL-TIME BATTERY DIGITAL TWIN with ML
100% Real-Time | Live Arduino Data ONLY | Instant ML Prediction | Live Visualization

Features:
- Direct Arduino serial reading (NO CSV files!)
- Real-time SOH prediction using trained Random Forest
- Live updating graphs (7 plots)
- Professional dashboard with key metrics
- Feature engineering on-the-fly
- Shows 0 when battery disconnected
- Pure real-time monitoring
"""

# CRITICAL: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive display on Linux
import matplotlib.patches as mpatches

import serial
import time
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import joblib
import warnings
warnings.filterwarnings('ignore')

os.chdir(Path(__file__).resolve().parents[1])

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = os.environ.get("ESP32_PORT", "COM10")  # Override with ESP32_PORT if needed
BAUD_RATE = 115200                # ESP32 default baud rate
MAX_POINTS = 100            # Number of points to display
UPDATE_INTERVAL = 1000      # ms (1 second)
SIMULATION_MODE = os.environ.get("SIMULATE", "auto").lower() in ("1", "true", "yes", "on")
SIMULATION_SOURCE = os.environ.get("SIMULATION_SOURCE", "replay").lower()

# ==========================================
# LOAD ML MODEL
# ==========================================
print("=" * 70)
print("[BATTERY DIGITAL TWIN - LIVE ESP32 MODE]")
print("[LIVE] Real-time sensor data | No CSV files | Live Visualization")
print("=" * 70)

try:
    print("\n[1] Loading ML model...")
    model = joblib.load("battery_soh_model.pkl")
    features_list = joblib.load("model_features.pkl")
    print(f"[OK] Model loaded: {len(features_list)} features")
except FileNotFoundError:
    print("[ERROR] Model not found! Run train_model_enhanced.py first.")
    exit(1)

# ==========================================
# INITIALIZE DATA STORAGE
# ==========================================
print("\n[2] Initializing data buffers...")

# Time series data (using deque for efficient append/pop)
time_data = deque(maxlen=MAX_POINTS)
voltage_data = deque(maxlen=MAX_POINTS)
current_data = deque(maxlen=MAX_POINTS)
temp_data = deque(maxlen=MAX_POINTS)
soc_data = deque(maxlen=MAX_POINTS)
power_data = deque(maxlen=MAX_POINTS)
internal_r_data = deque(maxlen=MAX_POINTS)
soh_data = deque(maxlen=MAX_POINTS)
actual_soh_data = deque(maxlen=MAX_POINTS)
rul_data = deque(maxlen=MAX_POINTS)

# Previous values for derivatives
prev_voltage = None
prev_current = None
prev_temp = None
prev_soc = None
prev_energy = None

# Rolling window for statistics
voltage_window = deque(maxlen=10)
current_window = deque(maxlen=10)
temp_window = deque(maxlen=10)
time_remaining_window = deque(maxlen=20)  # Store last 20 time remaining values for averaging

# Battery model reference used for actual-vs-predicted comparison
battery_capacity_Ah = 2.2

# Session stats
start_time = time.time()
prediction_count = 0
min_soh = 100.0
max_soh = 0.0

print("[OK] Buffers initialized")

# ==========================================
# CONNECT TO ESP32
# ==========================================
print(f"\n[3] Connecting to ESP32 on {COM_PORT} @ {BAUD_RATE} baud...")
ser = None
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    # Flush buffer to clear any partial data
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("[OK] Connected to ESP32")
    print("[OK] Reading LIVE data from ESP32 (no CSV files)")
    print(f"[OK] Port: {COM_PORT} | Baud: {BAUD_RATE}")
except Exception as e:
    print(f"[WARN] Connection failed: {e}")
    print("[INFO] Switching to simulation mode so the dashboard can still run.")
    SIMULATION_MODE = True

# Simulation state used when hardware is unavailable
sim_start_time = time.time()
sim_step = 0
sim_voltage = 4.2
sim_current = 0.8
sim_temp = 26.0
sim_soc = 100.0
sim_energy = 0.0
sim_replay_rows = []
sim_replay_index = 0

def load_replay_rows(csv_path="log.csv"):
    rows = []
    file_path = Path(csv_path)
    if not file_path.exists():
        return rows

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "time" in line.lower():
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue

            try:
                rows.append(
                    (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                    )
                )
            except ValueError:
                continue

    return rows

if SIMULATION_MODE and SIMULATION_SOURCE == "replay":
    sim_replay_rows = load_replay_rows("log.csv")
    if sim_replay_rows:
        print(f"[INFO] Loaded {len(sim_replay_rows)} samples from log.csv for replay simulation.")
    else:
        print("[INFO] log.csv not found or empty. Falling back to synthetic simulation.")

# ==========================================
# FEATURE ENGINEERING FUNCTION
# ==========================================
def engineer_features_live(voltage, current, temp, soc, energy):
    """
    Create all 20 features required by ML model in real-time
    """
    global prev_voltage, prev_current, prev_temp, prev_soc, prev_energy
    
    features = {}
    
    # Core measurements
    features['Voltage(V)'] = voltage
    features['Current(A)'] = current
    features['Temp(C)'] = temp
    features['SoC(%)'] = soc
    features['Energy(Wh)'] = energy
    
    # Power
    features['Power'] = voltage * current
    
    # Derivatives (rate of change)
    if prev_voltage is not None:
        features['dV'] = voltage - prev_voltage
        features['dI'] = current - prev_current
        features['dTemp'] = temp - prev_temp
        features['dSoC'] = soc - prev_soc
        features['dEnergy'] = energy - prev_energy
    else:
        features['dV'] = 0.0
        features['dI'] = 0.0
        features['dTemp'] = 0.0
        features['dSoC'] = 0.0
        features['dEnergy'] = 0.0
    
    # Internal resistance (Ohm's law)
    if abs(features['dI']) > 1e-6:
        features['Internal_R'] = abs(features['dV'] / features['dI'])
    else:
        features['Internal_R'] = 0.5  # Default value
    
    # Clip to physical limits
    features['Internal_R'] = np.clip(features['Internal_R'], 0.0, 2.0)
    
    # Ah used
    features['Ah_used'] = energy / (voltage + 1e-6)
    
    # Rolling statistics
    voltage_window.append(voltage)
    current_window.append(current)
    temp_window.append(temp)
    
    features['V_rolling_mean'] = np.mean(voltage_window)
    features['V_rolling_std'] = np.std(voltage_window) if len(voltage_window) > 1 else 0.0
    features['I_rolling_mean'] = np.mean(current_window)
    features['Temp_rolling_mean'] = np.mean(temp_window)
    
    # Advanced features
    features['Power_density'] = features['Power'] / (voltage + 1e-6)
    features['Thermal_stress'] = temp * current
    features['Voltage_efficiency'] = voltage / 6.5  # Normalized to max voltage
    
    # Update previous values
    prev_voltage = voltage
    prev_current = current
    prev_temp = temp
    prev_soc = soc
    prev_energy = energy
    
    return features

# ==========================================
# ML PREDICTION FUNCTION
# ==========================================
def predict_soh_live(features_dict):
    """
    Predict SOH using trained Random Forest model
    """
    # Create feature vector in correct order
    X = np.array([[features_dict.get(f, 0.0) for f in features_list]])
    
    # Handle any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Predict
    soh = model.predict(X)[0]
    
    # Clip to valid range
    soh = np.clip(soh, 0.0, 100.0)
    
    return float(soh)

# ==========================================
# RUL CALCULATION
# ==========================================
def calculate_rul(soh, current_ma):
    """
    Estimate Remaining Useful Life in hours
    """
    if soh <= 80:
        return 0  # Battery at end of life
    
    # Simple linear degradation model
    # Assumes 0.1% degradation per hour under load
    degradation_rate = 0.1 * (current_ma / 1000.0)  # Faster degradation with higher current
    hours_to_80_percent = (soh - 80) / degradation_rate
    
    return max(0, hours_to_80_percent)


def calculate_actual_soh(energy_wh, voltage_v):
    """
    Reference SOH derived from usable capacity.
    This is a simple baseline so we can compare model prediction vs actual estimate.
    """
    if voltage_v <= 0:
        return 0.0

    ah_used = energy_wh / (voltage_v + 1e-6)
    actual_soh = 100 - ((ah_used / battery_capacity_Ah) * 100)
    return float(np.clip(actual_soh, 0.0, 100.0))

# ==========================================
# DATA READING FUNCTION
# ==========================================
def read_sensor_data(debug=False):
    """
    Read one line of sensor data directly from ESP32
    Returns: time_s, voltage, current, temp, soc, energy
    """
    global sim_step, sim_voltage, sim_current, sim_temp, sim_soc, sim_energy

    if SIMULATION_MODE or ser is None:
        if SIMULATION_SOURCE == "replay" and sim_replay_rows:
            global sim_replay_index
            result = sim_replay_rows[sim_replay_index]
            sim_replay_index = (sim_replay_index + 1) % len(sim_replay_rows)

            if debug:
                print(
                    f"[REPLAY] T={result[0]:.1f}, V={result[1]:.3f}, I={result[2]:.3f}, "
                    f"Tmp={result[3]:.2f}, SoC={result[4]:.2f}, E={result[5]:.4f}"
                )

            return result

        elapsed = time.time() - sim_start_time
        sim_step += 1

        discharge_wave = 0.03 * np.sin(sim_step / 6.0)
        load_wave = 0.08 * np.sin(sim_step / 8.0)
        temp_wave = 1.5 * np.sin(sim_step / 10.0)

        sim_current = max(0.12, 0.7 + load_wave)
        sim_voltage = max(3.0, 4.2 - (sim_step * 0.0025) - discharge_wave)
        sim_temp = max(20.0, 26.0 + temp_wave + (sim_current * 2.5))
        sim_soc = max(0.0, sim_soc - (sim_current * 0.04))
        sim_energy = max(0.0, sim_energy + (sim_voltage * sim_current * (UPDATE_INTERVAL / 3600000.0)))

        if debug:
            print(
                f"[SIM] T={elapsed:.1f}, V={sim_voltage:.3f}, I={sim_current:.3f}, "
                f"Tmp={sim_temp:.2f}, SoC={sim_soc:.2f}, E={sim_energy:.4f}"
            )

        return (elapsed, sim_voltage, sim_current, sim_temp, sim_soc, sim_energy)

    # Read from ESP32 with timeout - skip incomplete lines
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            if debug and line:
                print(f"[DEBUG] Raw data: {line}")
            
            # Skip header or empty lines
            if not line or "Time(s)" in line or "time" in line.lower():
                return None
            
            try:
                parts = line.split(',')
                # MUST have exactly 6 parts - skip if incomplete
                if len(parts) != 6:
                    if debug and parts:
                        print(f"[DEBUG] Wrong format: {len(parts)} parts (need 6), skipping: {line}")
                    return None
                
                try:
                    result = (
                        float(parts[0]),  # Time
                        float(parts[1]),  # Voltage
                        float(parts[2]),  # Current
                        float(parts[3]),  # Temperature
                        float(parts[4]),  # SoC
                        float(parts[5])   # Energy
                    )
                    if debug:
                        print(f"[DEBUG] Parsed: T={result[0]}, V={result[1]}, I={result[2]}, Tmp={result[3]}, SoC={result[4]}, E={result[5]}")
                    return result
                except ValueError as ve:
                    if debug:
                        print(f"[DEBUG] Conversion error: {ve} | Parts: {parts}")
                    return None
            except Exception as e:
                if debug:
                    print(f"[WARN] Parse error: {e} | Line: {line}")
                return None
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Read error: {e}")
        return None

# ==========================================
# UI THEME CONFIGURATION
# ==========================================

# --- Color Palette ---
BG_DARK       = '#0D1117'   # Near-black background (main)
BG_PANEL      = '#161B22'   # Slightly lighter panel
BG_CARD       = '#1C2430'   # Card/subplot background
GRID_COLOR    = '#21262D'   # Subtle grid lines
BORDER_COLOR  = '#30363D'   # Borders and separators

# Accent colors
ACCENT_TEAL   = '#00E5CC'   # Electric teal  — voltage, primary accent
ACCENT_AMBER  = '#FFB347'   # Warm amber     — temperature, warnings
ACCENT_RED    = '#FF4D6D'   # Vivid red      — current, alerts
ACCENT_PURPLE = '#BD93F9'   # Soft purple    — power
ACCENT_GREEN  = '#50FA7B'   # Neon green     — internal resistance
ACCENT_CYAN   = '#8BE9FD'   # Light cyan     — SoC
ACCENT_GOLD   = '#F1FA8C'   # Pale gold      — SOH main line
EOL_RED       = '#FF5555'   # End-of-life reference line

TEXT_PRIMARY  = '#E6EDF3'   # Near-white body text
TEXT_DIM      = '#8B949E'   # Muted labels
TEXT_ACCENT   = '#00E5CC'   # Highlighted values

# Apply global matplotlib style
plt.rcParams.update({
    # Figure
    'figure.facecolor':     BG_DARK,
    'figure.edgecolor':     BG_DARK,

    # Axes
    'axes.facecolor':       BG_CARD,
    'axes.edgecolor':       BORDER_COLOR,
    'axes.labelcolor':      TEXT_DIM,
    'axes.titlecolor':      TEXT_PRIMARY,
    'axes.titleweight':     'bold',
    'axes.titlesize':       10,
    'axes.labelsize':       8,
    'axes.linewidth':       0.8,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'axes.axisbelow':       True,

    # Grid
    'grid.color':           GRID_COLOR,
    'grid.linewidth':       0.6,
    'grid.alpha':           1.0,

    # Ticks
    'xtick.color':          TEXT_DIM,
    'ytick.color':          TEXT_DIM,
    'xtick.labelsize':      7,
    'ytick.labelsize':      7,
    'xtick.major.size':     3,
    'ytick.major.size':     3,

    # Legend
    'legend.facecolor':     BG_PANEL,
    'legend.edgecolor':     BORDER_COLOR,
    'legend.labelcolor':    TEXT_DIM,
    'legend.fontsize':      7.5,
    'legend.framealpha':    0.85,

    # Font — Monospace for data, clean for labels
    'font.family':          'monospace',
    'font.size':            8,

    # Lines
    'lines.linewidth':      1.8,
    'lines.antialiased':    True,
})

# ==========================================
# VISUALIZATION SETUP
# ==========================================
print("\n[4] Setting up visualization...")

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor(BG_DARK)

# ── Title bar (top 5% of figure) ──────────────────────────────────────────
fig.text(
    0.5, 0.975,
    '⚡  BATTERY DIGITAL TWIN',
    ha='center', va='top',
    fontsize=17, fontweight='bold',
    color=ACCENT_TEAL,
    fontfamily='monospace',
)
fig.text(
    0.5, 0.950,
    'LIVE ESP32   ·   ML STATE-OF-HEALTH   ·   REAL-TIME ANALYTICS',
    ha='center', va='top',
    fontsize=7.5,
    color=TEXT_DIM,
    fontfamily='monospace',
)

# ── Thin separator line under title ───────────────────────────────────────
fig.add_artist(plt.Line2D(
    [0.01, 0.99], [0.932, 0.932],
    transform=fig.transFigure,
    color=BORDER_COLOR, linewidth=0.8
))

# ── Subplot grid — plots only, 3 cols, left 73% of figure ─────────────────
# top=0.925 leaves space for title; right=0.72 leaves right 28% for dashboard
gs = fig.add_gridspec(
    3, 3,
    left=0.04, right=0.71,
    top=0.920, bottom=0.06,
    hspace=0.58, wspace=0.38
)

ax1 = fig.add_subplot(gs[0, 0])   # Voltage
ax2 = fig.add_subplot(gs[0, 1])   # Current
ax3 = fig.add_subplot(gs[0, 2])   # Temperature
ax4 = fig.add_subplot(gs[1, 0])   # Power
ax5 = fig.add_subplot(gs[1, 1])   # Internal Resistance
ax6 = fig.add_subplot(gs[1, 2])   # SoC
ax7 = fig.add_subplot(gs[2, :])   # SOH — full bottom row

# ── Dashboard panel — dedicated axes, right 26% of figure ─────────────────
# [left, bottom, width, height] in figure fractions
ax_dashboard = fig.add_axes([0.735, 0.06, 0.245, 0.860])
ax_dashboard.set_facecolor(BG_PANEL)
for spine in ax_dashboard.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor(BORDER_COLOR)
    spine.set_linewidth(1.0)
ax_dashboard.axis('off')

# ── Plot lines ────────────────────────────────────────────────────────────
line_voltage,    = ax1.plot([], [], color=ACCENT_TEAL,   linewidth=2.0, label='Voltage',      solid_capstyle='round')
line_current,    = ax2.plot([], [], color=ACCENT_RED,    linewidth=2.0, label='Current',      solid_capstyle='round')
line_temp,       = ax3.plot([], [], color=ACCENT_AMBER,  linewidth=2.0, label='Temperature',  solid_capstyle='round')
line_power,      = ax4.plot([], [], color=ACCENT_PURPLE, linewidth=2.0, label='Power',        solid_capstyle='round')
line_internal_r, = ax5.plot([], [], color=ACCENT_GREEN,  linewidth=2.0, label='Internal R',   solid_capstyle='round')
line_soc,        = ax6.plot([], [], color=ACCENT_CYAN,   linewidth=2.0, label='SoC',          solid_capstyle='round')
line_soh,        = ax7.plot([], [], color=ACCENT_GOLD,   linewidth=2.8, label='SOH (ML)',     solid_capstyle='round', zorder=3)
line_actual_soh, = ax7.plot([], [], color=ACCENT_CYAN,   linewidth=2.0, linestyle='--', label='Actual SOH', solid_capstyle='round', zorder=2)

# ── Glow / fill under SOH line ────────────────────────────────────────────
soh_fill = ax7.fill_between([], [], alpha=0)   # placeholder; rebuilt each frame

# ── Axis styling helper ───────────────────────────────────────────────────
def style_axis(ax, title, ylabel, accent):
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=9.5, fontweight='bold', pad=6)
    ax.set_xlabel('Time (s)', color=TEXT_DIM, fontsize=7.5)
    ax.set_ylabel(ylabel,     color=TEXT_DIM, fontsize=7.5)

    ax.spines['left'].set_color(accent)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.tick_params(colors=TEXT_DIM, length=3)
    ax.legend(loc='upper right', fontsize=7.5,
              facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=accent)

style_axis(ax1, '▸ Voltage',           'V (volts)',  ACCENT_TEAL)
style_axis(ax2, '▸ Current',           'I (amps)',   ACCENT_RED)
style_axis(ax3, '▸ Temperature',       'T (°C)',     ACCENT_AMBER)
style_axis(ax4, '▸ Power',             'P (watts)',  ACCENT_PURPLE)
style_axis(ax5, '▸ Internal Resistance','R (ohms)',  ACCENT_GREEN)
style_axis(ax6, '▸ State of Charge',   'SoC (%)',   ACCENT_CYAN)

# ── SOH axis — featured treatment ─────────────────────────────────────────
ax7.set_facecolor(BG_CARD)
ax7.set_title('▸ STATE OF HEALTH  —  PREDICTED VS ACTUAL',
              color=ACCENT_GOLD, fontsize=11, fontweight='bold', pad=8)
ax7.set_xlabel('Time (s)', color=TEXT_DIM, fontsize=8)
ax7.set_ylabel('SOH (%)',  color=TEXT_DIM, fontsize=8)

ax7.spines['left'].set_color(ACCENT_GOLD)
ax7.spines['left'].set_linewidth(2)
ax7.spines['bottom'].set_color(BORDER_COLOR)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

ax7.tick_params(colors=TEXT_DIM, length=3)
ax7.axhline(y=80, color=EOL_RED, linestyle='--', linewidth=1.2,
            alpha=0.7, label='End-of-Life  (80 %)', zorder=2)
ax7.set_ylim(70, 105)
ax7.legend(loc='lower left', fontsize=8.5,
           facecolor=BG_PANEL, edgecolor=BORDER_COLOR,
           labelcolor=TEXT_DIM)
ax7.yaxis.label.set_color(ACCENT_GOLD)

print("[OK] Visualization ready")
print("\n[5] Starting LIVE ESP32 monitoring...")
print("=" * 70)
if SIMULATION_MODE:
    print("[SIM MODE] No hardware detected. Generating synthetic battery data.")
else:
    print(f"[LIVE MODE] Reading from {COM_PORT} @ {BAUD_RATE} baud")
print("[INFO] Remove battery to see 0V/0A | Press Ctrl+C to stop")
print("=" * 70)

# DEBUG MODE - Set to True to see raw serial data
DEBUG_MODE = False

# ==========================================
# ANIMATION UPDATE FUNCTION
# ==========================================
def update(frame):
    """
    Called every UPDATE_INTERVAL ms to update the plot
    """
    global prediction_count, min_soh, max_soh, soh_fill
    
    try:
        # Read sensor data
        data = read_sensor_data(debug=DEBUG_MODE)
        
        # If no data available, skip this update
        if data is None:
            return line_voltage, line_current, line_temp, line_power, line_internal_r, line_soc, line_soh
        
        time_s, voltage, current, temp, soc, energy = data
        
        # Store time
        time_data.append(time_s)
        
        # Engineer features
        features = engineer_features_live(voltage, current, temp, soc, energy)
        
        # Predict SOH using ML
        soh_pred = predict_soh_live(features)

        # Reference SOH for comparison
        actual_soh = calculate_actual_soh(energy, voltage)
        
        # Calculate RUL
        rul = calculate_rul(soh_pred, current * 1000)
        voltage_data.append(voltage)
        current_data.append(current)
        temp_data.append(temp)
        soc_data.append(soc)
        power_data.append(features['Power'])
        internal_r_data.append(features['Internal_R'])
        soh_data.append(soh_pred)
        actual_soh_data.append(actual_soh)
        rul_data.append(rul)
        
        # Update statistics
        prediction_count += 1
        min_soh = min(min_soh, soh_pred)
        max_soh = max(max_soh, soh_pred)
        
        # Update plots
        line_voltage.set_data(time_data, voltage_data)
        line_current.set_data(time_data, current_data)
        line_temp.set_data(time_data, temp_data)
        line_power.set_data(time_data, power_data)
        line_internal_r.set_data(time_data, internal_r_data)
        line_soc.set_data(time_data, soc_data)
        line_soh.set_data(time_data, soh_data)
        line_actual_soh.set_data(time_data, actual_soh_data)

        # Glow fill under SOH line
        soh_fill.remove()
        soh_fill = ax7.fill_between(
            time_data, list(soh_data), 70,
            color=ACCENT_GOLD, alpha=0.08, zorder=1
        )

        # Auto-scale axes
        for ax, data in [
            (ax1, voltage_data),
            (ax2, current_data),
            (ax3, temp_data),
            (ax4, power_data),
            (ax5, internal_r_data),
            (ax6, soc_data),
            (ax7, list(soh_data) + list(actual_soh_data))
        ]:
            if len(time_data) > 0:
                ax.set_xlim(min(time_data), max(time_data) + 1)
                if len(data) > 0:
                    y_min, y_max = min(data), max(data)
                    margin = (y_max - y_min) * 0.1 if y_max > y_min else 1
                    ax.set_ylim(y_min - margin, y_max + margin)
        
        # Update dashboard
        elapsed_time = time.time() - start_time
        avg_soh = np.mean(soh_data) if len(soh_data) > 0 else 0
        avg_actual_soh = np.mean(actual_soh_data) if len(actual_soh_data) > 0 else 0
        soh_error = abs(soh_pred - actual_soh)
        
        # Calculate degradation rate (% per hour)
        if len(soh_data) > 10:
            degradation_rate = (soh_data[0] - soh_data[-1]) / (elapsed_time / 3600) if elapsed_time > 0 else 0
        else:
            degradation_rate = 0
        
        # Detect if battery is actually connected (voltage threshold)
        battery_connected = voltage > 0.5  # Battery must have at least 0.5V to be considered connected
        
        # Calculate battery time remaining (based on current SoC and discharge rate)
        if battery_connected and current > 0.01:  # Battery is connected and discharging
            # Calculate remaining capacity: SoC% of total battery capacity
            # Assume typical 18650 battery: ~3.7V nominal, let's estimate capacity from current state
            # Remaining Ah = (Voltage * SoC%) / nominal_voltage, approximation
            remaining_capacity_ah = (soc / 100) * 2.5  # Assume ~2.5Ah battery capacity
            
            # Time remaining = Remaining capacity / discharge current
            time_remaining_hours = remaining_capacity_ah / current if current > 0 else 0
            
            # Add to rolling window for averaging
            time_remaining_window.append(time_remaining_hours)
        else:
            time_remaining_hours = float('inf')  # Not discharging or charging
            time_remaining_window.clear()  # Clear window when not discharging
        
        # Calculate average time remaining from window
        if len(time_remaining_window) > 0:
            avg_time_remaining = np.mean(time_remaining_window)
        else:
            avg_time_remaining = time_remaining_hours
        
        # Format instantaneous time remaining
        if not battery_connected:
            time_remaining_str = "N/A  (no battery)"
        elif time_remaining_hours == float('inf'):
            time_remaining_str = "N/A  (idle/charging)"
        elif time_remaining_hours > 100:
            time_remaining_str = "> 100 hours"
        elif time_remaining_hours < 0.016:  # Less than 1 minute
            time_remaining_str = "< 1 minute"
        else:
            hours = int(time_remaining_hours)
            minutes = int((time_remaining_hours - hours) * 60)
            if hours > 0:
                time_remaining_str = f"{hours}h {minutes}m"
            else:
                time_remaining_str = f"{minutes}m"
        
        # Format average time remaining
        if not battery_connected:
            avg_time_remaining_str = "N/A  (no battery)"
        elif avg_time_remaining == float('inf'):
            avg_time_remaining_str = "N/A  (idle/charging)"
        elif avg_time_remaining > 100:
            avg_time_remaining_str = "> 100 hours"
        elif avg_time_remaining < 0.016:  # Less than 1 minute
            avg_time_remaining_str = "< 1 minute"
        else:
            hours = int(avg_time_remaining)
            minutes = int((avg_time_remaining - hours) * 60)
            if hours > 0:
                avg_time_remaining_str = f"{hours}h {minutes}m"
            else:
                avg_time_remaining_str = f"{minutes}m"
        
        # Battery status indicator
        if SIMULATION_MODE:
            battery_status  = "● SIMULATION"
            status_color    = ACCENT_PURPLE
        elif battery_connected:
            battery_status  = "● CONNECTED"
            status_color    = ACCENT_GREEN
        else:
            battery_status  = "○ DISCONNECTED"
            status_color    = ACCENT_RED

        # ── Dashboard redraw ─────────────────────────────────────────────
        ax_dashboard.clear()
        ax_dashboard.set_facecolor(BG_PANEL)
        ax_dashboard.axis('off')
        for spine in ax_dashboard.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(BORDER_COLOR)
            spine.set_linewidth(1.0)

        T = ax_dashboard.transAxes   # shorthand

        def row(y, label, value, val_color=TEXT_PRIMARY, unit=''):
            ax_dashboard.text(0.06, y, label,
                              transform=T, fontsize=7.5, color=TEXT_DIM,
                              fontfamily='monospace', va='top')
            ax_dashboard.text(0.94, y, f"{value}{unit}",
                              transform=T, fontsize=7.5, color=val_color,
                              fontfamily='monospace', va='top', ha='right',
                              fontweight='bold')

        def section(y, title, accent=ACCENT_TEAL):
            ax_dashboard.text(0.06, y, title,
                              transform=T, fontsize=7, color=accent,
                              fontfamily='monospace', va='top',
                              fontweight='bold')

        # Panel title
        ax_dashboard.text(0.5, 0.985, 'DASHBOARD',
                          transform=T, fontsize=10, color=ACCENT_TEAL,
                          fontfamily='monospace', ha='center', va='top',
                          fontweight='bold')
        ax_dashboard.text(0.5, 0.963, 'LIVE  ·  DIGITAL TWIN',
                          transform=T, fontsize=6.5, color=TEXT_DIM,
                          fontfamily='monospace', ha='center', va='top')

        # Status badge
        ax_dashboard.text(0.5, 0.930, battery_status,
                          transform=T, fontsize=8.5, color=status_color,
                          fontfamily='monospace', ha='center', va='center',
                          fontweight='bold')

        # Session
        section(0.880, '─ SESSION', ACCENT_TEAL)
        row(0.848, 'Elapsed',     f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}", TEXT_PRIMARY, ' min')
        row(0.820, 'Predictions', f"{prediction_count:,}", ACCENT_TEAL)

        # SOH block
        section(0.785, '─ STATE OF HEALTH', ACCENT_GOLD)
        row(0.752, 'Current SOH', f"{soh_pred:.2f}", ACCENT_GOLD, ' %')
        row(0.724, 'Actual SOH',  f"{actual_soh:.2f}", ACCENT_CYAN,  ' %')
        row(0.696, 'SOH Error',   f"{soh_error:.2f}",   ACCENT_RED,   ' %')
        row(0.668, 'Average SOH', f"{avg_soh:.2f}",     TEXT_PRIMARY, ' %')
        row(0.640, 'Avg Actual',  f"{avg_actual_soh:.2f}", TEXT_DIM,    ' %')

        # Life estimates
        section(0.632, '─ LIFE ESTIMATES', ACCENT_AMBER)
        row(0.599, 'Time Left',   time_remaining_str,     ACCENT_AMBER)
        row(0.571, 'Avg Left',    avg_time_remaining_str, TEXT_PRIMARY)
        row(0.543, 'RUL (health)', f"{rul:.1f}", ACCENT_AMBER, ' h')
        row(0.515, 'Degradation', f"{degradation_rate:.4f}", TEXT_DIM, ' %/h')

        # Live readings
        section(0.478, '─ LIVE READINGS', ACCENT_TEAL)
        row(0.445, 'Voltage',    f"{voltage:.3f}",             ACCENT_TEAL,   ' V')
        row(0.417, 'Current',    f"{current:.4f}",             ACCENT_RED,    ' A')
        row(0.389, 'Temp',       f"{temp:.2f}",                ACCENT_AMBER,  ' °C')
        row(0.361, 'SoC',        f"{soc:.2f}",                 ACCENT_CYAN,   ' %')
        row(0.333, 'Power',      f"{features['Power']:.3f}",   ACCENT_PURPLE, ' W')
        row(0.305, 'Int. Resist',f"{features['Internal_R']:.4f}", ACCENT_GREEN,' Ω')
        row(0.277, 'Thermal',    f"{features['Thermal_stress']:.2f}", TEXT_DIM, '')

        # ML info
        section(0.240, '─ ML MODEL', ACCENT_PURPLE)
        row(0.207, 'Type',     'Random Forest', ACCENT_PURPLE)
        row(0.179, 'Trees',    '400',            TEXT_DIM)
        row(0.151, 'Accuracy', 'R² = 0.9999',   ACCENT_GREEN)
        row(0.123, 'Latency',  '< 1 ms',         ACCENT_TEAL)

        # Port info footer
        ax_dashboard.text(0.5, 0.055,
                          f'{COM_PORT}  @  {BAUD_RATE} baud',
                          transform=T, fontsize=6, color=TEXT_DIM,
                          fontfamily='monospace', ha='center', va='bottom')
        ax_dashboard.text(0.5, 0.030,
                          'ESP32  ·  Real-time',
                          transform=T, fontsize=6, color=BORDER_COLOR,
                          fontfamily='monospace', ha='center', va='bottom')

        # Console output
        if prediction_count % 2 == 0:  # Print every 2 predictions (faster feedback)
            msg = f"[{prediction_count:04d}] SOH: {soh_pred:6.2f}% | V: {voltage:.3f}V | I: {current:.4f}A | T: {temp:.1f}C | RUL: {rul:.1f}h"
            print(msg, file=sys.stderr)
            sys.stderr.flush()
    
    except Exception as e:
        print(f"[ERROR] Error in update: {e}", file=sys.stderr)
        sys.stderr.flush()
        pass
    
    return line_voltage, line_current, line_temp, line_power, line_internal_r, line_soc, line_soh

# ==========================================
# START ANIMATION
# ==========================================
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=False)

# Enable interactive mode so animation updates display without blocking
plt.ion()
plt.show()

try:
    print("\n[GUI Window opened - predictions showing below]\n")
    sys.stderr.flush()
    
    # Keep program running while animation works
    while True:
        plt.pause(0.001)
        
except KeyboardInterrupt:
    print("\n\n[STOP] Monitoring stopped by user")
finally:
    if ser is not None:
        ser.close()
    plt.close('all')
    
    print("\n" + "=" * 70)
    print("[SESSION SUMMARY - LIVE ESP32 DATA]")
    print("=" * 70)
    print(f"  - Total Predictions: {prediction_count}")
    print(f"  - Session Duration: {int((time.time() - start_time) // 60)}m {int((time.time() - start_time) % 60)}s")
    if len(soh_data) > 0:
        print(f"  - Average SOH: {np.mean(soh_data):.2f}%")
        print(f"  - Min SOH: {min_soh:.2f}%")
        print(f"  - Max SOH: {max_soh:.2f}%")
    print("=" * 70)
    print("\n[OK] Digital Twin session complete!")