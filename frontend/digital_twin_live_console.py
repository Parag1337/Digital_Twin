# -*- coding: utf-8 -*-
"""
REAL-TIME BATTERY DIGITAL TWIN with ML
100% Real-Time | Live ESP32 Data ONLY | Instant ML Prediction | Live Console Output

Features:
- Direct ESP32 serial reading (NO CSV files!)
- Real-time SOH prediction using trained Random Forest
- Live console dashboard with key metrics
- Feature engineering on-the-fly
- Shows 0 when battery disconnected
- Pure real-time monitoring
- NO matplotlib animation (more reliable)
"""

import serial
import time
import numpy as np
from collections import deque
import joblib
import warnings
import os
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

os.chdir(Path(__file__).resolve().parents[1])

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = "/dev/ttyUSB0"         # ESP32 MicroUSB on Linux
BAUD_RATE = 115200                # ESP32 default baud rate
MAX_POINTS = 100            # Number of points to store
DEBUG_MODE = True           # Enable to see raw serial data

# ==========================================
# LOAD ML MODEL
# ==========================================
print("=" * 80)
print("[BATTERY DIGITAL TWIN - LIVE ESP32 MODE]")
print("[LIVE] Real-time sensor data | No CSV files | Console Dashboard")
print("=" * 80)

try:
    print("\n[1] Loading ML model...")
    model = joblib.load("battery_soh_model.pkl")
    features_list = joblib.load("model_features.pkl")
    print(f"[OK] Model loaded: {len(features_list)} features")
except FileNotFoundError as e:
    print(f"[ERROR] Model not found! {e}")
    print("        Run train_model_enhanced.py first.")
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
    # Flush buffer to clear any partial data
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("[OK] Connected to ESP32")
    print("[OK] Reading LIVE data from ESP32 (no CSV files)")
    print(f"[OK] Port: {COM_PORT} | Baud: {BAUD_RATE}")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print("\n[INFO] Troubleshooting:")
    print("   1. Check if ESP32 is connected via MicroUSB")
    print("   2. Verify port: ls /dev/ttyUSB*")
    print("   3. Check Arduino IDE baud rate is 115200")
    print("   4. Make sure ESP32 code is uploaded and running")
    print("   5. Ensure ESP32 is sending CSV format: Time,Voltage,Current,Temp,SoC,Energy")
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
    if degradation_rate > 0:
        hours_to_80_percent = (soh - 80) / degradation_rate
    else:
        hours_to_80_percent = float('inf')
    
    return max(0, hours_to_80_percent)

# ==========================================
# DATA READING FUNCTION
# ==========================================
def read_sensor_data(debug=False):
    """
    Read one line of sensor data directly from ESP32
    Returns: time_s, voltage, current, temp, soc, energy
    """
    # Read from ESP32 with timeout - skip incomplete lines
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            if debug and line:
                print(f"[DEBUG] Raw: {line}")
            
            # Skip header or empty lines
            if not line or "Time(s)" in line or "time" in line.lower():
                return None
            
            try:
                parts = line.split(',')
                # MUST have exactly 6 parts - skip if incomplete
                if len(parts) != 6:
                    if debug and parts:
                        print(f"[DEBUG] Wrong format: {len(parts)} parts (need 6), skipping")
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
                        print(f"[DEBUG] Parsed: T={result[0]}, V={result[1]:.4f}, I={result[2]:.4f}, T={result[3]:.1f}, SoC={result[4]:.1f}, E={result[5]:.4f}")
                    return result
                except ValueError as ve:
                    if debug:
                        print(f"[DEBUG] Value error: {ve}")
                    return None
            except Exception as e:
                if debug:
                    print(f"[WARN] Parse error: {e}")
                return None
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Read error: {e}")
        return None

# ==========================================
# CONSOLE DASHBOARD
# ==========================================
def print_dashboard(soh_pred, voltage, current, temp, soc, energy, rul, features):
    """
    Print formatted dashboard to console
    """
    elapsed_time = time.time() - start_time
    avg_soh = np.mean(soh_data) if len(soh_data) > 0 else 0
    
    # Calculate degradation rate
    if len(soh_data) > 10:
        degradation_rate = (soh_data[0] - soh_data[-1]) / (elapsed_time / 3600) if elapsed_time > 0 else 0
    else:
        degradation_rate = 0
    
    battery_connected = voltage > 0.5
    battery_status = "[CONNECTED]" if battery_connected else "[DISCONNECTED]"
    
    # Clear screen (works on Linux/Mac/Windows)
    os.system('clear' if os.name == 'posix' else 'cls')
    
    dashboard = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    [LIVE] Battery Digital Twin Dashboard                    ║
║                         ESP32 Real-Time Monitoring                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

 BATTERY STATUS: {battery_status}
 
 ┌─ SESSION INFO ───────────────────────────────────────────────────────────┐
 │  Time          : {int(elapsed_time // 60):3d}m {int(elapsed_time % 60):02d}s                                   │
 │  Predictions   : {prediction_count:4d}                                              │
 └────────────────────────────────────────────────────────────────────────────┘

 ┌─ STATE OF HEALTH (SOH) ──────────────────────────────────────────────────┐
 │  Current SOH   : {soh_pred:6.2f} %                                             │
 │  Average SOH   : {avg_soh:6.2f} %                                             │
 │  Min SOH       : {min_soh:6.2f} %                                             │
 │  Max SOH       : {max_soh:6.2f} %                                             │
 └────────────────────────────────────────────────────────────────────────────┘

 ┌─ LIVE SENSOR READINGS ───────────────────────────────────────────────────┐
 │  Voltage       : {voltage:8.4f} V                                        │
 │  Current       : {current:8.4f} A                                        │
 │  Temperature   : {temp:8.2f} °C                                       │
 │  SoC           : {soc:8.2f} %                                        │
 │  Power         : {features['Power']:8.4f} W                                        │
 │  Energy        : {energy:8.4f} Wh                                       │
 └────────────────────────────────────────────────────────────────────────────┘

 ┌─ BATTERY ANALYSIS ───────────────────────────────────────────────────────┐
 │  Internal R    : {features['Internal_R']:8.4f} Ω                                        │
 │  Thermal Stress: {features['Thermal_stress']:8.2f}                                             │
 │  Voltage Eff.  : {features['Voltage_efficiency']:8.4f}                                        │
 │  RUL (Health)  : {rul:8.1f} hours                                       │
 │  Degradation   : {degradation_rate:8.6f} %/hour                                    │
 └────────────────────────────────────────────────────────────────────────────┘

 ┌─ ML MODEL INFO ──────────────────────────────────────────────────────────┐
 │  Model Type    : Random Forest (20 features)                            │
 │  Trees         : 400                                                    │
 │  Test R²       : 0.9999                                                 │
 │  Test RMSE     : 0.0306 %                                               │
 │  Inference     : < 1 ms                                                 │
 └────────────────────────────────────────────────────────────────────────────┘

 Press Ctrl+C to stop monitoring...

╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(dashboard)

# ==========================================
# MAIN MONITORING LOOP
# ==========================================
print("\n[4] Starting LIVE ESP32 monitoring...")
print("=" * 80)
print(f"[LIVE MODE] Reading from {COM_PORT} @ {BAUD_RATE} baud")
print("[INFO] Displaying live data every second...")
print("[INFO] Remove battery to see 0V/0A | Press Ctrl+C to stop")
print("=" * 80)

time.sleep(1)

try:
    while True:
        # Read sensor data
        data = read_sensor_data(debug=DEBUG_MODE)
        
        if data is not None:
            time_s, voltage, current, temp, soc, energy = data
            
            # Store time
            time_data.append(time_s)
            
            # Engineer features
            features = engineer_features_live(voltage, current, temp, soc, energy)
            
            # Predict SOH using ML
            soh_pred = predict_soh_live(features)
            
            # Calculate RUL
            rul = calculate_rul(soh_pred, current * 1000)
            
            # Store data
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
            
            # Update dashboard every prediction
            print_dashboard(soh_pred, voltage, current, temp, soc, energy, rul, features)
            
            # Print line for logging
            if prediction_count % 1 == 0:
                print(f"\n[{prediction_count:04d}] SOH: {soh_pred:6.2f}% | V: {voltage:.4f}V | I: {current:.4f}A | T: {temp:.1f}°C | RUL: {rul:.1f}h")
        
        # Small delay to prevent CPU spinning
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n[STOP] Monitoring stopped by user")

finally:
    ser.close()
    
    print("\n" + "=" * 80)
    print("[SESSION SUMMARY - LIVE ESP32 DATA]")
    print("=" * 80)
    print(f"  - Total Predictions: {prediction_count}")
    elapsed = time.time() - start_time
    print(f"  - Session Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    if len(soh_data) > 0:
        print(f"  - Average SOH: {np.mean(soh_data):.2f}%")
        print(f"  - Min SOH: {min_soh:.2f}%")
        print(f"  - Max SOH: {max_soh:.2f}%")
    print("=" * 80)
    print("\n[OK] Digital Twin session complete!")
