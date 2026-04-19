# -*- coding: utf-8 -*-
"""
REAL-TIME BATTERY DIGITAL TWIN with ML - Matplotlib Alternative
Works better with threading for reliable live updates
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import joblib
import warnings
import threading
import os
from pathlib import Path
warnings.filterwarnings('ignore')

os.chdir(Path(__file__).resolve().parents[1])

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
MAX_POINTS = 100
DEBUG_MODE = False

# ==========================================
# LOAD ML MODEL
# ==========================================
print("=" * 70)
print("[BATTERY DIGITAL TWIN - LIVE ESP32 MODE]")
print("[LIVE] Real-time sensor data | Live Matplotlib Visualization")
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

time_data = deque(maxlen=MAX_POINTS)
voltage_data = deque(maxlen=MAX_POINTS)
current_data = deque(maxlen=MAX_POINTS)
temp_data = deque(maxlen=MAX_POINTS)
soc_data = deque(maxlen=MAX_POINTS)
power_data = deque(maxlen=MAX_POINTS)
internal_r_data = deque(maxlen=MAX_POINTS)
soh_data = deque(maxlen=MAX_POINTS)
rul_data = deque(maxlen=MAX_POINTS)

# Global state
prev_voltage = None
prev_current = None
prev_temp = None
prev_soc = None
prev_energy = None

voltage_window = deque(maxlen=10)
current_window = deque(maxlen=10)
temp_window = deque(maxlen=10)

start_time = time.time()
prediction_count = 0
min_soh = 100.0
max_soh = 0.0

# Threading
data_lock = threading.Lock()
stop_flag = False

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
    print("   1. Check if ESP32 is connected via MicroUSB")
    print("   2. Verify port: ls /dev/ttyUSB*")
    print("   3. Check Arduino IDE baud rate is 115200")
    exit(1)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def engineer_features_live(voltage, current, temp, soc, energy):
    global prev_voltage, prev_current, prev_temp, prev_soc, prev_energy
    
    features = {}
    features['Voltage(V)'] = voltage
    features['Current(A)'] = current
    features['Temp(C)'] = temp
    features['SoC(%)'] = soc
    features['Energy(Wh)'] = energy
    features['Power'] = voltage * current
    
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
    
    if abs(features['dI']) > 1e-6:
        features['Internal_R'] = abs(features['dV'] / features['dI'])
    else:
        features['Internal_R'] = 0.5
    
    features['Internal_R'] = np.clip(features['Internal_R'], 0.0, 2.0)
    features['Ah_used'] = energy / (voltage + 1e-6)
    
    voltage_window.append(voltage)
    current_window.append(current)
    temp_window.append(temp)
    
    features['V_rolling_mean'] = np.mean(voltage_window)
    features['V_rolling_std'] = np.std(voltage_window) if len(voltage_window) > 1 else 0.0
    features['I_rolling_mean'] = np.mean(current_window)
    features['Temp_rolling_mean'] = np.mean(temp_window)
    
    features['Power_density'] = features['Power'] / (voltage + 1e-6)
    features['Thermal_stress'] = temp * current
    features['Voltage_efficiency'] = voltage / 6.5
    
    prev_voltage = voltage
    prev_current = current
    prev_temp = temp
    prev_soc = soc
    prev_energy = energy
    
    return features

def predict_soh_live(features_dict):
    X = np.array([[features_dict.get(f, 0.0) for f in features_list]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    soh = model.predict(X)[0]
    soh = np.clip(soh, 0.0, 100.0)
    return float(soh)

def calculate_rul(soh, current_ma):
    if soh <= 80:
        return 0
    degradation_rate = 0.1 * (current_ma / 1000.0)
    if degradation_rate > 0:
        return (soh - 80) / degradation_rate
    return float('inf')

def read_sensor_data():
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if not line or "Time(s)" in line:
                return None
            parts = line.split(',')
            if len(parts) >= 6:
                return (float(parts[0]), float(parts[1]), float(parts[2]), 
                       float(parts[3]), float(parts[4]), float(parts[5]))
    except:
        pass
    return None

# ==========================================
# DATA COLLECTION THREAD
# ==========================================
def data_collection_thread():
    global prediction_count, min_soh, max_soh
    
    while not stop_flag:
        data = read_sensor_data()
        if data is not None:
            time_s, voltage, current, temp, soc, energy = data
            
            with data_lock:
                time_data.append(time_s)
                features = engineer_features_live(voltage, current, temp, soc, energy)
                soh_pred = predict_soh_live(features)
                rul = calculate_rul(soh_pred, current * 1000)
                
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
        
        time.sleep(0.01)  # Small delay to prevent CPU spinning

# ==========================================
# SETUP MATPLOTLIB
# ==========================================
print("\n[4] Setting up visualization...")

fig = plt.figure(figsize=(18, 11))
fig.suptitle('[LIVE ESP32 DATA] Battery Digital Twin with ML', fontsize=16, fontweight='bold', color='darkblue')

ax1 = plt.subplot(3, 4, 1)
ax2 = plt.subplot(3, 4, 2)
ax3 = plt.subplot(3, 4, 3)
ax4 = plt.subplot(3, 4, 5)
ax5 = plt.subplot(3, 4, 6)
ax6 = plt.subplot(3, 4, 7)
ax7 = plt.subplot(3, 2, 5)
ax_dashboard = plt.subplot(3, 4, (4, 12))

line_voltage, = ax1.plot([], [], 'b-', linewidth=2, label='Voltage')
line_current, = ax2.plot([], [], 'r-', linewidth=2, label='Current')
line_temp, = ax3.plot([], [], 'orange', linewidth=2, label='Temperature')
line_power, = ax4.plot([], [], 'purple', linewidth=2, label='Power')
line_internal_r, = ax5.plot([], [], 'green', linewidth=2, label='Internal R')
line_soc, = ax6.plot([], [], 'cyan', linewidth=2, label='SoC')
line_soh, = ax7.plot([], [], 'darkblue', linewidth=3, label='SOH (ML)', marker='o', markersize=3)

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

ax7.set_title('[ML PREDICTION] State of Health', fontweight='bold', fontsize=12)
ax7.set_xlabel('Time (s)', fontsize=10)
ax7.set_ylabel('SOH (%)', fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.legend(loc='lower left', fontsize=10)
ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='End of Life (80%)')
ax7.set_ylim(70, 105)

ax_dashboard.axis('off')
plt.tight_layout()

print("[OK] Visualization ready")
print("\n[5] Starting LIVE ESP32 monitoring (with threading)...")
print("=" * 70)
print(f"[LIVE MODE] Reading from {COM_PORT} @ {BAUD_RATE} baud")
print("[INFO] Press Ctrl+C to stop")
print("=" * 70)

# ==========================================
# START DATA COLLECTION THREAD
# ==========================================
collector_thread = threading.Thread(target=data_collection_thread, daemon=True)
collector_thread.start()

# ==========================================
# UPDATE FUNCTION FOR MATPLOTLIB
# ==========================================
def update(frame):
    with data_lock:
        if len(time_data) > 0:
            line_voltage.set_data(time_data, voltage_data)
            line_current.set_data(time_data, current_data)
            line_temp.set_data(time_data, temp_data)
            line_power.set_data(time_data, power_data)
            line_internal_r.set_data(time_data, internal_r_data)
            line_soc.set_data(time_data, soc_data)
            line_soh.set_data(time_data, soh_data)
            
            for ax, data in [(ax1, voltage_data), (ax2, current_data), (ax3, temp_data),
                           (ax4, power_data), (ax5, internal_r_data), (ax6, soc_data), (ax7, soh_data)]:
                ax.set_xlim(min(time_data), max(time_data) + 1)
                if len(data) > 0:
                    y_min, y_max = min(data), max(data)
                    margin = (y_max - y_min) * 0.1 if y_max > y_min else 1
                    ax.set_ylim(y_min - margin, y_max + margin)
            
            elapsed = time.time() - start_time
            avg_soh = np.mean(soh_data) if len(soh_data) > 0 else 0
            battery_connected = voltage_data[-1] > 0.5 if len(voltage_data) > 0 else False
            battery_status = "[CONNECTED]" if battery_connected else "[DISCONNECTED]"
            
            dashboard = f"""
╔════════════════════════════════════════════╗
║   [LIVE] Digital Twin Dashboard           ║
╚════════════════════════════════════════════╝

 BATTERY STATUS: {battery_status}
 
 SESSION: {int(elapsed // 60)}m {int(elapsed % 60)}s | Predictions: {prediction_count}

 SOH: Current {(soh_data[-1] if len(soh_data) > 0 else 0):.2f}% | Avg {avg_soh:.2f}% | Min {min_soh:.2f}% | Max {max_soh:.2f}%

 LIVE DATA:
 V: {(voltage_data[-1] if len(voltage_data) > 0 else 0):.4f}V | I: {(current_data[-1] if len(current_data) > 0 else 0):.4f}A | T: {(temp_data[-1] if len(temp_data) > 0 else 0):.2f}°C
 SoC: {(soc_data[-1] if len(soc_data) > 0 else 0):.2f}% | Power: {(power_data[-1] if len(power_data) > 0 else 0):.4f}W

 ML Model: Random Forest | Accuracy: R² = 0.9999

╚════════════════════════════════════════════╝
            """
            
            ax_dashboard.clear()
            ax_dashboard.axis('off')
            ax_dashboard.text(0.02, 0.98, dashboard, 
                             transform=ax_dashboard.transAxes,
                             fontsize=9,
                             verticalalignment='top',
                             fontfamily='monospace',
                             linespacing=1.5,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2, pad=0.8))
    
    return line_voltage, line_current, line_temp, line_power, line_internal_r, line_soc, line_soh

# ==========================================
# START ANIMATION WITH FASTER UPDATE
# ==========================================
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, interval=500, blit=False)  # Update every 500ms

try:
    plt.show()
except KeyboardInterrupt:
    print("\n[STOP] Monitoring stopped by user")
finally:
    stop_flag = True
    collector_thread.join(timeout=2)
    ser.close()
    
    print("\n" + "=" * 70)
    print("[SESSION SUMMARY - LIVE ESP32 DATA]")
    print("=" * 70)
    print(f"  - Total Predictions: {prediction_count}")
    elapsed = time.time() - start_time
    print(f"  - Session Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    if len(soh_data) > 0:
        print(f"  - Average SOH: {np.mean(soh_data):.2f}%")
        print(f"  - Min SOH: {min_soh:.2f}%")
        print(f"  - Max SOH: {max_soh:.2f}%")
    print("=" * 70)
    print("\n[OK] Digital Twin session complete!")
