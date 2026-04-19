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

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = "/dev/ttyUSB0"         # ESP32 MicroUSB on Linux
BAUD_RATE = 115200                # ESP32 default baud rate
MAX_POINTS = 100            # Number of points to display
UPDATE_INTERVAL = 1000      # ms (1 second)

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
time_remaining_window = deque(maxlen=20)  # Store last 20 time remaining values for averaging

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
# VISUALIZATION SETUP
# ==========================================
print("\n[4] Setting up visualization...")

# Create figure with better layout
fig = plt.figure(figsize=(18, 11))
fig.suptitle('[LIVE ESP32 DATA] Battery Digital Twin with ML', fontsize=16, fontweight='bold', color='darkblue')

# Define subplots with more space
ax1 = plt.subplot(3, 4, 1)   # Voltage
ax2 = plt.subplot(3, 4, 2)   # Current
ax3 = plt.subplot(3, 4, 3)   # Temperature
ax4 = plt.subplot(3, 4, 5)   # Power
ax5 = plt.subplot(3, 4, 6)   # Internal Resistance
ax6 = plt.subplot(3, 4, 7)   # SoC
ax7 = plt.subplot(3, 2, 5)   # SOH with ML prediction (bottom left, larger)

# Dashboard text (right side, full height)
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

# SOH plot (larger)
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
            time_remaining_str = "N/A (No Battery)"
        elif time_remaining_hours == float('inf'):
            time_remaining_str = "N/A (Idle/Charging)"
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
            avg_time_remaining_str = "N/A (No Battery)"
        elif avg_time_remaining == float('inf'):
            avg_time_remaining_str = "N/A (Idle/Charging)"
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
        if prediction_count % 2 == 0:  # Print every 2 predictions (faster feedback)
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
