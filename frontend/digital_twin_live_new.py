# -*- coding: utf-8 -*-
"""
REAL-TIME BATTERY DIGITAL TWIN with ML
100% Real-Time | Live ESP32 Data | Instant ML Prediction | Live Visualization

Features:
- Direct ESP32 serial reading (NO CSV files!)
- Real-time SOH prediction using trained Random Forest
- Live updating graphs (7 plots)
- Professional dashboard with key metrics
- Feature engineering on-the-fly
- Pure real-time monitoring
"""

# CRITICAL: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive display on Linux

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import joblib
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

os.chdir(Path(__file__).resolve().parents[1])

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = "/dev/ttyUSB0"         # ESP32 MicroUSB on Linux
BAUD_RATE = 115200                # ESP32 default baud rate
MAX_POINTS = 100                  # Number of points to display
UPDATE_INTERVAL = 1000            # ms (1 second)

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
time_remaining_window = deque(maxlen=20)

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
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("[OK] Connected to ESP32")
    print(f"[OK] Port: {COM_PORT} | Baud: {BAUD_RATE}")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print("\n[INFO] Troubleshooting:")
    print("   1. Check if ESP32 is connected via MicroUSB")
    print("   2. Verify port: ls /dev/ttyUSB*")
    print("   3. Check Arduino IDE baud rate is 115200")
    print("   4. Make sure ESP32 code is uploaded and running")
    exit(1)

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
        features['Internal_R'] = 0.5
    
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
    features['Voltage_efficiency'] = voltage / 6.5
    
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
    X = np.array([[features_dict.get(f, 0.0) for f in features_list]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    soh = model.predict(X)[0]
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
        return 0
    degradation_rate = 0.1 * (current_ma / 1000.0)
    hours_to_80_percent = (soh - 80) / degradation_rate
    return max(0, hours_to_80_percent)

# ==========================================
# DATA READING FUNCTION
# ==========================================
def read_sensor_data(debug=False):
    """
    Read one line of sensor data directly from ESP32
    Returns: time_s, voltage, current, temp, soc, energy
    """
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
                if len(parts) >= 6:
                    return (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5])
                    )
            except ValueError:
                return None
    except Exception as e:
        print(f"[ERROR] Read error: {e}")
    return None

# ==========================================
# VISUALIZATION SETUP
# ==========================================
print("\n[4] Setting up visualization...")

# Create figure
fig = plt.figure(figsize=(18, 11))
fig.suptitle('[LIVE ESP32 DATA] Battery Digital Twin with ML', fontsize=16, fontweight='bold', color='darkblue')

# Define subplots
ax1 = plt.subplot(3, 4, 1)   # Voltage
ax2 = plt.subplot(3, 4, 2)   # Current
ax3 = plt.subplot(3, 4, 3)   # Temperature
ax4 = plt.subplot(3, 4, 5)   # Power
ax5 = plt.subplot(3, 4, 6)   # Internal Resistance
ax6 = plt.subplot(3, 4, 7)   # SoC
ax7 = plt.subplot(3, 2, 5)   # SOH (bottom left, larger)

# Dashboard text (right side)
ax_dashboard = plt.subplot(3, 4, (4, 12))
ax_dashboard.axis('off')

# Initialize line plots
line_voltage, = ax1.plot([], [], 'b-', linewidth=2, label='Voltage')
line_current, = ax2.plot([], [], 'r-', linewidth=2, label='Current')
line_temp, = ax3.plot([], [], 'orange', linewidth=2, label='Temperature')
line_power, = ax4.plot([], [], 'purple', linewidth=2, label='Power')
line_internal_r, = ax5.plot([], [], 'green', linewidth=2, label='Internal R')
line_soc, = ax6.plot([], [], 'cyan', linewidth=2, label='SoC')
line_soh, = ax7.plot([], [], 'darkblue', linewidth=3, label='SOH (ML)', marker='o', markersize=3)

# Configure axes
def setup_axis(ax, title, ylabel, color):
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)

setup_axis(ax1, '[LIVE] Voltage (ESP32)', 'Voltage (V)', 'blue')
setup_axis(ax2, '[LIVE] Current (ESP32)', 'Current (A)', 'red')
setup_axis(ax3, '[LIVE] Temperature (ESP32)', 'Temp (C)', 'orange')
setup_axis(ax4, '[LIVE] Power (ESP32)', 'Power (W)', 'purple')
setup_axis(ax5, '[LIVE] Internal Resistance', 'Resistance (Ohm)', 'green')
setup_axis(ax6, '[LIVE] SoC (ESP32)', 'SoC (%)', 'cyan')

# SOH plot
ax7.set_title('[ML PREDICTION] State of Health', fontweight='bold', fontsize=12)
ax7.set_xlabel('Time (s)', fontsize=10)
ax7.set_ylabel('SOH (%)', fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.legend(loc='lower left', fontsize=10)
ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='End of Life (80%)')
ax7.set_ylim(70, 105)

plt.tight_layout()

print("[OK] Visualization ready")
print("\n[5] Starting LIVE ESP32 monitoring...")
print("=" * 70)
print(f"[LIVE MODE] Reading from {COM_PORT} @ {BAUD_RATE} baud")
print("[INFO] Press Ctrl+C to stop")
print("=" * 70)

DEBUG_MODE = False

# ==========================================
# ANIMATION UPDATE FUNCTION
# ==========================================
def update(frame):
    """
    Called every UPDATE_INTERVAL ms to update the plot
    """
    global prediction_count, min_soh, max_soh
    
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
        
        # Calculate RUL
        rul = calculate_rul(soh_pred, current * 1000)
        voltage_data.append(voltage)
        current_data.append(current)
        temp_data.append(temp)
        soc_data.append(soc)
        power_data.append(features['Power'])
        internal_r_data.append(features['Internal_R'])
        soh_data.append(soh_pred)
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
        
        # Auto-scale axes
        for ax, data in [
            (ax1, voltage_data),
            (ax2, current_data),
            (ax3, temp_data),
            (ax4, power_data),
            (ax5, internal_r_data),
            (ax6, soc_data),
            (ax7, soh_data)
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
        
        # Calculate degradation rate (% per hour)
        if len(soh_data) > 10:
            degradation_rate = (soh_data[0] - soh_data[-1]) / (elapsed_time / 3600) if elapsed_time > 0 else 0
        else:
            degradation_rate = 0
        
        # Detect if battery is actually connected (voltage threshold)
        battery_connected = voltage > 0.5
        
        # Calculate battery time remaining
        if battery_connected and current > 0.01:
            remaining_capacity_ah = (soc / 100) * 2.5
            time_remaining_hours = remaining_capacity_ah / current if current > 0 else 0
            time_remaining_window.append(time_remaining_hours)
        else:
            time_remaining_hours = float('inf')
            time_remaining_window.clear()
        
        # Calculate average time remaining
        if len(time_remaining_window) > 0:
            avg_time_remaining = np.mean(time_remaining_window)
        else:
            avg_time_remaining = time_remaining_hours
        
        # Format times
        def format_time(hours):
            if hours == float('inf'):
                return "N/A"
            elif hours > 100:
                return "> 100h"
            elif hours < 0.016:
                return "< 1m"
            else:
                h = int(hours)
                m = int((hours - h) * 60)
                return f"{h}h {m}m" if h > 0 else f"{m}m"
        
        time_remaining_str = format_time(time_remaining_hours) if battery_connected else "N/A"
        avg_time_remaining_str = format_time(avg_time_remaining) if battery_connected else "N/A"
        
        battery_status = "[CONNECTED]" if battery_connected else "[DISCONNECTED]"
        
        dashboard_text = f"""
╔════════════════════════════════════════════╗
║   [LIVE] Digital Twin Dashboard           ║
╚════════════════════════════════════════════╝

 BATTERY STATUS: {battery_status}
 
 SESSION INFO:
 ───────────────────────────────────────────
  Time          : {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s
  Predictions   : {prediction_count}

 STATE OF HEALTH (SOH):
 ───────────────────────────────────────────
  Current SOH   : {soh_pred:.2f} %
  Average SOH   : {avg_soh:.2f} %
  Min SOH       : {min_soh:.2f} %
  Max SOH       : {max_soh:.2f} %

 BATTERY LIFE ESTIMATES:
 ───────────────────────────────────────────
  Time Left     : {time_remaining_str}
  Avg Time Left : {avg_time_remaining_str}
  RUL (Health)  : {rul:.1f} hours
  Degradation   : {degradation_rate:.4f} %/h

 LIVE READINGS:
 ───────────────────────────────────────────
  Voltage       : {voltage:.3f} V
  Current       : {current:.4f} A
  Temperature   : {temp:.2f} C
  SoC           : {soc:.2f} %
  Power         : {features['Power']:.3f} W
  Internal R    : {features['Internal_R']:.4f} Ohm
  Thermal       : {features['Thermal_stress']:.2f}

 ML MODEL INFO:
 ───────────────────────────────────────────
  Type          : Random Forest
  Trees         : 400
  Accuracy      : R2 = 0.9999
  Speed         : < 1ms

╚════════════════════════════════════════════╝
        """
        
        ax_dashboard.clear()
        ax_dashboard.axis('off')
        ax_dashboard.text(0.02, 0.98, dashboard_text, 
                         transform=ax_dashboard.transAxes,
                         fontsize=8.5,
                         verticalalignment='top',
                         fontfamily='monospace',
                         linespacing=1.6,
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2, pad=0.8))
        
        # Console output
        if prediction_count % 10 == 0:
            print(f"[{prediction_count:04d}] SOH: {soh_pred:6.2f}% | V: {voltage:.3f}V | I: {current:.4f}A | T: {temp:.1f}C | RUL: {rul:.1f}h")
    
    except Exception as e:
        print(f"[ERROR] Error in update: {e}")
        pass
    
    return line_voltage, line_current, line_temp, line_power, line_internal_r, line_soc, line_soh

# ==========================================
# START ANIMATION
# ==========================================
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=False)

try:
    plt.show()
    print("\n\n[STOP] Monitoring stopped by user")
except KeyboardInterrupt:
    print("\n\n[STOP] Monitoring stopped by user")
finally:
    ser.close()
    
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
matplotlib.use('TkAgg')

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from collections import deque
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
MAX_POINTS = 100
UPDATE_INTERVAL = 1000

# ==========================================
# DESIGN TOKENS — Cyberpunk / Industrial Dark
# ==========================================
BG_DARK      = "#080C14"       # Near-black navy
BG_PANEL     = "#0D1520"       # Panel background
BG_CARD      = "#111D2E"       # Card background
ACCENT_CYAN  = "#00F5FF"       # Primary accent
ACCENT_GREEN = "#00FF88"       # Positive / OK
ACCENT_AMBER = "#FFB800"       # Warning
ACCENT_RED   = "#FF3A5C"       # Danger / EOL
ACCENT_BLUE  = "#3A8EFF"       # Voltage
ACCENT_PURP  = "#B06DFF"       # Power
TEXT_PRIMARY = "#E8F4FF"       # Main text
TEXT_DIM     = "#4A6080"       # Subdued text
GRID_COLOR   = "#0F2035"       # Gridlines

GLOW_CYAN    = {"color": ACCENT_CYAN,  "alpha": 0.08}
GLOW_GREEN   = {"color": ACCENT_GREEN, "alpha": 0.08}

# ==========================================
# LOAD ML MODEL
# ==========================================
print("=" * 70)
print("  ⚡  BATTERY DIGITAL TWIN  |  LIVE ESP32 MODE")
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

time_data       = deque(maxlen=MAX_POINTS)
voltage_data    = deque(maxlen=MAX_POINTS)
current_data    = deque(maxlen=MAX_POINTS)
temp_data       = deque(maxlen=MAX_POINTS)
soc_data        = deque(maxlen=MAX_POINTS)
power_data      = deque(maxlen=MAX_POINTS)
internal_r_data = deque(maxlen=MAX_POINTS)
soh_data        = deque(maxlen=MAX_POINTS)
rul_data        = deque(maxlen=MAX_POINTS)

prev_voltage = prev_current = prev_temp = prev_soc = prev_energy = None

voltage_window       = deque(maxlen=10)
current_window       = deque(maxlen=10)
temp_window          = deque(maxlen=10)
time_remaining_window= deque(maxlen=20)

start_time       = time.time()
prediction_count = 0
min_soh          = 100.0
max_soh          = 0.0

print("[OK] Buffers initialized")

# ==========================================
# CONNECT TO ESP32
# ==========================================
print(f"\n[3] Connecting to ESP32 on {COM_PORT} @ {BAUD_RATE} baud...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print(f"[OK] Connected  |  Port: {COM_PORT}  |  Baud: {BAUD_RATE}")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print("\n  Troubleshooting:")
    print("   1. Check if ESP32 is connected via MicroUSB")
    print("   2. Verify port: ls /dev/ttyUSB*")
    print("   3. Check Arduino IDE baud rate is 115200")
    print("   4. Make sure ESP32 code is uploaded and running")
    print("   5. ESP32 must send CSV: Time,Voltage,Current,Temp,SoC,Energy")
    exit(1)

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def engineer_features_live(voltage, current, temp, soc, energy):
    global prev_voltage, prev_current, prev_temp, prev_soc, prev_energy
    f = {}
    f['Voltage(V)'] = voltage
    f['Current(A)'] = current
    f['Temp(C)']    = temp
    f['SoC(%)']     = soc
    f['Energy(Wh)'] = energy
    f['Power']      = voltage * current

    if prev_voltage is not None:
        f['dV']      = voltage - prev_voltage
        f['dI']      = current - prev_current
        f['dTemp']   = temp    - prev_temp
        f['dSoC']    = soc     - prev_soc
        f['dEnergy'] = energy  - prev_energy
    else:
        f['dV'] = f['dI'] = f['dTemp'] = f['dSoC'] = f['dEnergy'] = 0.0

    f['Internal_R'] = np.clip(
        abs(f['dV'] / f['dI']) if abs(f['dI']) > 1e-6 else 0.5,
        0.0, 2.0
    )
    f['Ah_used'] = energy / (voltage + 1e-6)

    voltage_window.append(voltage)
    current_window.append(current)
    temp_window.append(temp)

    f['V_rolling_mean']  = np.mean(voltage_window)
    f['V_rolling_std']   = np.std(voltage_window) if len(voltage_window) > 1 else 0.0
    f['I_rolling_mean']  = np.mean(current_window)
    f['Temp_rolling_mean']= np.mean(temp_window)
    f['Power_density']   = f['Power'] / (voltage + 1e-6)
    f['Thermal_stress']  = temp * current
    f['Voltage_efficiency'] = voltage / 6.5

    prev_voltage = voltage;  prev_current = current
    prev_temp    = temp;     prev_soc     = soc
    prev_energy  = energy
    return f

# ==========================================
# ML PREDICTION
# ==========================================
def predict_soh_live(features_dict):
    X = np.array([[features_dict.get(f, 0.0) for f in features_list]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.clip(model.predict(X)[0], 0.0, 100.0))

# ==========================================
# RUL CALCULATION
# ==========================================
def calculate_rul(soh, current_ma):
    if soh <= 80:
        return 0
    degradation_rate = 0.1 * (current_ma / 1000.0)
    return max(0, (soh - 80) / degradation_rate)

# ==========================================
# SERIAL READ
# ==========================================
DEBUG_MODE = False

def read_sensor_data(debug=False):
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if debug and line:
                print(f"[DEBUG] Raw: {line}")
            if not line or "Time(s)" in line or "time" in line.lower():
                return None
            parts = line.split(',')
            if len(parts) >= 6:
                try:
                    return (float(parts[0]), float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]), float(parts[5]))
                except ValueError:
                    return None
    except Exception as e:
        print(f"[ERROR] Read error: {e}")
    return None

# ==========================================
# VISUALIZATION — MODERN DARK DASHBOARD
# ==========================================
print("\n[4] Building modern dashboard...")

plt.rcParams.update({
    'figure.facecolor':  BG_DARK,
    'axes.facecolor':    BG_PANEL,
    'axes.edgecolor':    TEXT_DIM,
    'axes.labelcolor':   TEXT_DIM,
    'xtick.color':       TEXT_DIM,
    'ytick.color':       TEXT_DIM,
    'text.color':        TEXT_PRIMARY,
    'grid.color':        GRID_COLOR,
    'grid.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'monospace',
})

fig = plt.figure(figsize=(22, 13), constrained_layout=False)
fig.patch.set_facecolor(BG_DARK)

# ── Grid layout ────────────────────────────────────────────────────────────
gs_main  = GridSpec(1, 2, figure=fig,
                    left=0.01, right=0.99, top=0.94, bottom=0.04,
                    wspace=0.04, width_ratios=[3, 1])
gs_plots = GridSpec(3, 3, figure=fig,
                    left=0.04, right=0.71, top=0.91, bottom=0.06,
                    hspace=0.52, wspace=0.38)

ax1  = fig.add_subplot(gs_plots[0, 0])   # Voltage
ax2  = fig.add_subplot(gs_plots[0, 1])   # Current
ax3  = fig.add_subplot(gs_plots[0, 2])   # Temperature
ax4  = fig.add_subplot(gs_plots[1, 0])   # Power
ax5  = fig.add_subplot(gs_plots[1, 1])   # Internal R
ax6  = fig.add_subplot(gs_plots[1, 2])   # SoC
ax7  = fig.add_subplot(gs_plots[2, :])   # SOH (full width)
ax_d = fig.add_subplot(gs_main[0, 1])    # Dashboard panel

# ── Title bar ───────────────────────────────────────────────────────────────
fig.text(0.375, 0.975, "⚡  BATTERY DIGITAL TWIN",
         ha='center', va='top', fontsize=18, fontweight='bold',
         color=ACCENT_CYAN,
         path_effects=[pe.withSimplePatchShadow(shadow_rgbFace=(0,0.9,1),
                                                 alpha=0.18, rho=0.9)])
fig.text(0.375, 0.956, "REAL-TIME  ·  ML-POWERED  ·  LIVE ESP32",
         ha='center', va='top', fontsize=8.5, color=TEXT_DIM)

# ── Helper: style a small plot ───────────────────────────────────────────────
def style_ax(ax, title, ylabel, accent):
    ax.set_facecolor(BG_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(0.8)
        spine.set_alpha(0.5)
    ax.set_title(title, color=accent, fontsize=8.5, fontweight='bold',
                 pad=5, loc='left')
    ax.set_ylabel(ylabel, fontsize=7.5, color=TEXT_DIM)
    ax.set_xlabel('t (s)', fontsize=7, color=TEXT_DIM)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=7, colors=TEXT_DIM)

style_ax(ax1, "VOLTAGE",        "V",    ACCENT_BLUE)
style_ax(ax2, "CURRENT",        "A",    ACCENT_RED)
style_ax(ax3, "TEMPERATURE",    "°C",   ACCENT_AMBER)
style_ax(ax4, "POWER",          "W",    ACCENT_PURP)
style_ax(ax5, "INTERNAL  R",    "Ω",    ACCENT_GREEN)
style_ax(ax6, "STATE OF CHARGE","%",    ACCENT_CYAN)

# SOH plot — wider treatment
ax7.set_facecolor(BG_CARD)
for sp in ax7.spines.values():
    sp.set_edgecolor(ACCENT_GREEN); sp.set_linewidth(1.2); sp.set_alpha(0.6)
ax7.set_title("STATE OF HEALTH  ·  ML PREDICTION (Random Forest  ·  R²=0.9999)",
              color=ACCENT_GREEN, fontsize=9.5, fontweight='bold', pad=6, loc='left')
ax7.set_ylabel("SOH  (%)", fontsize=8, color=TEXT_DIM)
ax7.set_xlabel("Time (s)", fontsize=8, color=TEXT_DIM)
ax7.grid(True, alpha=0.4)
ax7.tick_params(labelsize=8)
ax7.set_ylim(70, 105)
ax7.axhline(y=80, color=ACCENT_RED, linewidth=1.2, linestyle='--', alpha=0.55)
ax7.text(0, 80.5, " END OF LIFE  80%", color=ACCENT_RED,
         fontsize=7, alpha=0.8, fontfamily='monospace')

# EOL fill zone
ax7.axhspan(70, 80, color=ACCENT_RED, alpha=0.06)

# ── Line objects ─────────────────────────────────────────────────────────────
def make_line(ax, color, lw=1.8):
    line, = ax.plot([], [], color=color, linewidth=lw,
                    solid_capstyle='round', solid_joinstyle='round')
    # Glow twin
    glow, = ax.plot([], [], color=color, linewidth=lw*3.5, alpha=0.12,
                    solid_capstyle='round')
    return line, glow

lv,  lv_g  = make_line(ax1, ACCENT_BLUE)
li,  li_g  = make_line(ax2, ACCENT_RED)
lt,  lt_g  = make_line(ax3, ACCENT_AMBER)
lp,  lp_g  = make_line(ax4, ACCENT_PURP)
lr,  lr_g  = make_line(ax5, ACCENT_GREEN)
ls,  ls_g  = make_line(ax6, ACCENT_CYAN)

# SOH gets special thick + glow treatment
lsoh, = ax7.plot([], [], color=ACCENT_GREEN, linewidth=2.5,
                  solid_capstyle='round', zorder=3)
lsoh_g, = ax7.plot([], [], color=ACCENT_GREEN, linewidth=10, alpha=0.12,
                    solid_capstyle='round', zorder=2)
soh_dot, = ax7.plot([], [], 'o', color=ACCENT_GREEN, markersize=8, zorder=4)
soh_dot_g,= ax7.plot([], [], 'o', color=ACCENT_GREEN, markersize=20, alpha=0.2, zorder=3)

# ── Dashboard panel setup ────────────────────────────────────────────────────
ax_d.set_facecolor(BG_PANEL)
ax_d.axis('off')
for sp in ax_d.spines.values():
    sp.set_visible(False)

# Decorative side accent bar
fig.patches.append(
    mpatches.FancyBboxPatch(
        (0.717, 0.04), 0.004, 0.87,
        boxstyle="square,pad=0",
        linewidth=0, facecolor=ACCENT_CYAN, alpha=0.6,
        transform=fig.transFigure, clip_on=False, zorder=5
    )
)

# Dashboard background card
fig.patches.append(
    mpatches.FancyBboxPatch(
        (0.724, 0.04), 0.265, 0.87,
        boxstyle="round,pad=0.005",
        linewidth=1, edgecolor=ACCENT_CYAN, facecolor=BG_CARD, alpha=0.95,
        transform=fig.transFigure, clip_on=False, zorder=4
    )
)

print("[OK] Dashboard built")
print("\n[5] Starting live monitoring...")
print("=" * 70)
print(f"  Port: {COM_PORT}  |  Baud: {BAUD_RATE}  |  Ctrl+C to stop")
print("=" * 70)

# ── Animated scan-line effect on SOH plot ────────────────────────────────────
scan_line = ax7.axvline(x=0, color=ACCENT_CYAN, linewidth=1, alpha=0.4, zorder=5)

# ==========================================
# UPDATE FUNCTION
# ==========================================
def update(frame):
    global prediction_count, min_soh, max_soh

    try:
        data = read_sensor_data(debug=DEBUG_MODE)
        if data is None:
            return (lv, lv_g, li, li_g, lt, lt_g, lp, lp_g,
                    lr, lr_g, ls, ls_g, lsoh, lsoh_g, soh_dot, soh_dot_g)

        time_s, voltage, current, temp, soc, energy = data
        time_data.append(time_s)

        features = engineer_features_live(voltage, current, temp, soc, energy)
        soh_pred = predict_soh_live(features)
        rul      = calculate_rul(soh_pred, current * 1000)

        voltage_data.append(voltage)
        current_data.append(current)
        temp_data.append(temp)
        soc_data.append(soc)
        power_data.append(features['Power'])
        internal_r_data.append(features['Internal_R'])
        soh_data.append(soh_pred)
        rul_data.append(rul)

        prediction_count += 1
        min_soh = min(min_soh, soh_pred)
        max_soh = max(max_soh, soh_pred)

        # ── Update line data ────────────────────────────────────────────────
        t_arr = list(time_data)

        def set_xy(line, glow, ydata):
            line.set_data(t_arr, ydata)
            glow.set_data(t_arr, ydata)

        set_xy(lv,  lv_g,  list(voltage_data))
        set_xy(li,  li_g,  list(current_data))
        set_xy(lt,  lt_g,  list(temp_data))
        set_xy(lp,  lp_g,  list(power_data))
        set_xy(lr,  lr_g,  list(internal_r_data))
        set_xy(ls,  ls_g,  list(soc_data))
        lsoh.set_data(t_arr,  list(soh_data))
        lsoh_g.set_data(t_arr, list(soh_data))

        # Moving dot at latest SOH
        if t_arr:
            soh_dot.set_data([t_arr[-1]], [soh_pred])
            soh_dot_g.set_data([t_arr[-1]], [soh_pred])
            scan_line.set_xdata([t_arr[-1], t_arr[-1]])

        # ── Auto-scale plots ────────────────────────────────────────────────
        pairs = [
            (ax1, voltage_data),  (ax2, current_data), (ax3, temp_data),
            (ax4, power_data),    (ax5, internal_r_data),(ax6, soc_data),
            (ax7, soh_data),
        ]
        for ax, d in pairs:
            if len(t_arr) > 0:
                ax.set_xlim(min(t_arr), max(t_arr) + 1)
            if len(d) > 0:
                y0, y1 = min(d), max(d)
                m = (y1 - y0) * 0.12 if y1 > y0 else 1
                if ax is not ax7:
                    ax.set_ylim(y0 - m, y1 + m)

        # ── Time remaining calculation ──────────────────────────────────────
        battery_connected = voltage > 0.5
        if battery_connected and current > 0.01:
            remaining_ah    = (soc / 100) * 2.5
            time_rem_h      = remaining_ah / current if current > 0 else 0
            time_remaining_window.append(time_rem_h)
        else:
            time_rem_h = float('inf')
            time_remaining_window.clear()

        avg_tr = np.mean(time_remaining_window) if time_remaining_window else time_rem_h

        def fmt_time(h):
            if not battery_connected: return "NO BATTERY"
            if h == float('inf'):    return "IDLE / CHG"
            if h > 100:              return "> 100 h"
            if h < 0.016:            return "< 1 min"
            hh, mm = int(h), int((h - int(h)) * 60)
            return f"{hh}h {mm:02d}m" if hh else f"{mm}m"

        elapsed = time.time() - start_time
        avg_soh = np.mean(soh_data) if soh_data else 0

        degradation_rate = 0.0
        if len(soh_data) > 10:
            degradation_rate = (soh_data[0] - soh_data[-1]) / (elapsed / 3600) if elapsed > 0 else 0

        # SOH colour
        soh_color = ACCENT_GREEN if soh_pred >= 90 else ACCENT_AMBER if soh_pred >= 80 else ACCENT_RED
        status_str= "● CONNECTED" if battery_connected else "○ NO SIGNAL"
        status_clr= ACCENT_GREEN if battery_connected else ACCENT_RED

        # ── Dashboard text ──────────────────────────────────────────────────
        ax_d.clear()
        ax_d.set_facecolor(BG_PANEL)
        ax_d.axis('off')

        X = 0.08   # left margin inside panel

        def label(y, text, size=7.5, color=TEXT_DIM, bold=False):
            ax_d.text(X, y, text, transform=ax_d.transAxes,
                      fontsize=size, color=color, va='top', fontfamily='monospace',
                      fontweight='bold' if bold else 'normal')

        def value(y, text, size=12, color=TEXT_PRIMARY):
            ax_d.text(1 - X, y, text, transform=ax_d.transAxes,
                      fontsize=size, color=color, va='top', ha='right',
                      fontfamily='monospace', fontweight='bold')

        def hline(y):
            ax_d.plot([0.05, 0.95], [y, y], transform=ax_d.transAxes,
                      color=TEXT_DIM, linewidth=0.4, alpha=0.4, clip_on=False)

        # Header
        ax_d.text(0.5, 0.985, "DIGITAL TWIN", transform=ax_d.transAxes,
                  fontsize=11, color=ACCENT_CYAN, va='top', ha='center',
                  fontfamily='monospace', fontweight='bold',
                  path_effects=[pe.withSimplePatchShadow(
                      shadow_rgbFace=(0,0.9,1), alpha=0.25, rho=0.9)])
        ax_d.text(0.5, 0.962, "LIVE  MONITORING  PANEL", transform=ax_d.transAxes,
                  fontsize=6.5, color=TEXT_DIM, va='top', ha='center',
                  fontfamily='monospace')
        hline(0.955)

        # Status
        ax_d.text(0.5, 0.935, status_str, transform=ax_d.transAxes,
                  fontsize=9, color=status_clr, va='top', ha='center',
                  fontfamily='monospace', fontweight='bold')
        hline(0.915)

        # Session
        label(0.905, "SESSION", size=6.5, bold=True, color=TEXT_DIM)
        label(0.883, "Elapsed")
        value(0.883, f"{int(elapsed//60)}m {int(elapsed%60):02d}s")
        label(0.862, "Predictions")
        value(0.862, f"{prediction_count:,}")
        hline(0.848)

        # SOH block
        label(0.838, "STATE OF HEALTH", size=6.5, bold=True, color=TEXT_DIM)
        ax_d.text(0.5, 0.815, f"{soh_pred:.2f}%", transform=ax_d.transAxes,
                  fontsize=22, color=soh_color, va='top', ha='center',
                  fontfamily='monospace', fontweight='bold',
                  path_effects=[pe.withSimplePatchShadow(
                      shadow_rgbFace=soh_color, alpha=0.3, rho=0.95)])
        label(0.782, "Average")
        value(0.782, f"{avg_soh:.2f}%")
        label(0.762, "Min / Max")
        value(0.762, f"{min_soh:.1f}  /  {max_soh:.1f}%")
        hline(0.748)

        # Battery life
        label(0.738, "BATTERY LIFE", size=6.5, bold=True, color=TEXT_DIM)
        label(0.718, "Time Left")
        value(0.718, fmt_time(time_rem_h), color=ACCENT_CYAN)
        label(0.698, "Avg Time Left")
        value(0.698, fmt_time(avg_tr), color=ACCENT_CYAN)
        label(0.678, "RUL (health)")
        value(0.678, f"{rul:.1f} h", color=ACCENT_AMBER)
        label(0.658, "Degradation")
        value(0.658, f"{degradation_rate:.4f} %/h", color=ACCENT_RED if degradation_rate > 0.05 else TEXT_PRIMARY)
        hline(0.642)

        # Live readings
        label(0.632, "LIVE READINGS", size=6.5, bold=True, color=TEXT_DIM)
        readings = [
            ("Voltage",    f"{voltage:.3f} V",          ACCENT_BLUE),
            ("Current",    f"{current:.4f} A",           ACCENT_RED),
            ("Temperature",f"{temp:.2f} °C",             ACCENT_AMBER),
            ("SoC",        f"{soc:.2f} %",               ACCENT_CYAN),
            ("Power",      f"{features['Power']:.3f} W", ACCENT_PURP),
            ("Internal R", f"{features['Internal_R']:.4f} Ω", ACCENT_GREEN),
            ("Thermal",    f"{features['Thermal_stress']:.2f}", TEXT_DIM),
        ]
        y = 0.612
        for lbl, val, col in readings:
            label(y, lbl)
            value(y, val, size=9, color=col)
            y -= 0.021
        hline(y + 0.005)

        # Model info
        y -= 0.01
        label(y, "ML MODEL", size=6.5, bold=True, color=TEXT_DIM); y -= 0.021
        label(y, "Type");      value(y, "Random Forest", size=8); y -= 0.021
        label(y, "Trees");     value(y, "400", size=8); y -= 0.021
        label(y, "R² Score");  value(y, "0.9999", size=8, color=ACCENT_GREEN); y -= 0.021
        label(y, "Latency");   value(y, "< 1 ms", size=8, color=ACCENT_CYAN)

        # Console
        if prediction_count % 10 == 0:
            print(f"  [{prediction_count:04d}]  SOH {soh_pred:6.2f}%  |  "
                  f"V {voltage:.3f}  |  I {current:.4f}  |  T {temp:.1f}°C  |  "
                  f"RUL {rul:.1f}h")

    except Exception as e:
        print(f"[ERROR] {e}")

    return (lv, lv_g, li, li_g, lt, lt_g, lp, lp_g,
            lr, lr_g, ls, ls_g, lsoh, lsoh_g, soh_dot, soh_dot_g)

# ==========================================
# START ANIMATION
# ==========================================
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=False)

try:
    plt.show()
    print("\n\n[STOP] Monitoring stopped by user")
except KeyboardInterrupt:
    print("\n\n[STOP] Monitoring stopped by user")
finally:
    ser.close()
    print("\n" + "=" * 70)
    print("  SESSION SUMMARY  —  LIVE ESP32 DATA")
    print("=" * 70)
    print(f"  Predictions  : {prediction_count}")
    print(f"  Duration     : {int((time.time()-start_time)//60)}m "
          f"{int((time.time()-start_time)%60)}s")
    if soh_data:
        print(f"  Avg SOH      : {np.mean(soh_data):.2f}%")
        print(f"  Min/Max SOH  : {min_soh:.2f}%  /  {max_soh:.2f}%")
    print("=" * 70)
    print("  Digital Twin session complete.\n")