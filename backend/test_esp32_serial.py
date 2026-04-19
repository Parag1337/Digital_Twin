#!/usr/bin/env python3
"""
ESP32 Serial Communication Tester
Checks if ESP32 is properly connected and sending data
"""

import serial
import time
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

# Configuration
COM_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

print("=" * 70)
print("[ESP32 SERIAL CONNECTION TESTER]")
print("=" * 70)

# Step 1: Check connection
print(f"\n[1] Attempting to connect to {COM_PORT} @ {BAUD_RATE} baud...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("[✓] Connection successful!")
except Exception as e:
    print(f"[✗] Connection failed: {e}")
    print("\nTroubleshooting:")
    print("  - Run: ls /dev/ttyUSB* (to find the port)")
    print("  - Check: Is ESP32 plugged in via MicroUSB?")
    print("  - Verify: Arduino IDE shows correct COM port")
    exit(1)

# Step 2: Check for incoming data
print(f"\n[2] Waiting for data from ESP32 (10 seconds)...")
print("    Listening for CSV format: Time(s), Voltage(V), Current(A), Temp(C), SoC(%), Energy(Wh)")
print("-" * 70)

start_time = time.time()
data_count = 0
sample_lines = []

while time.time() - start_time < 10:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data_count += 1
                print(f"[{data_count}] {line}")
                sample_lines.append(line)
        except Exception as e:
            print(f"[Error reading line] {e}")

ser.close()

print("-" * 70)

# Step 3: Analysis
if data_count == 0:
    print("\n[✗] NO DATA RECEIVED!")
    print("\nPossible issues:")
    print("  1. ESP32 is not sending data")
    print("     → Check your ESP32 Arduino code is printing sensor data")
    print("  2. Wrong baud rate (should be 115200)")
    print("     → Verify in Arduino IDE: Tools → Upload Speed")
    print("  3. Serial data not printing")
    print("     → Add Serial.begin(115200) in your ESP32 setup()")
    print("  4. USB driver issue")
    print("     → Try: sudo chmod 666 /dev/ttyUSB0")
else:
    print(f"\n[✓] SUCCESS! Received {data_count} lines of data")
    
    # Check format
    if sample_lines:
        print("\n[3] Checking data format...")
        first_line = sample_lines[0]
        parts = first_line.split(',')
        
        if len(parts) >= 6:
            print(f"[✓] Correct format detected: {len(parts)} columns")
            print(f"\n    Sample: {first_line}")
            print("\n    Columns:")
            try:
                print(f"      - Time(s):   {float(parts[0])}")
                print(f"      - Voltage(V): {float(parts[1])}")
                print(f"      - Current(A): {float(parts[2])}")
                print(f"      - Temp(C):   {float(parts[3])}")
                print(f"      - SoC(%):    {float(parts[4])}")
                print(f"      - Energy(Wh): {float(parts[5])}")
                print("\n[✓] All values are numeric - Ready for digital_twin_live.py!")
            except ValueError as e:
                print(f"[✗] Error parsing values: {e}")
                print("    Make sure all values are numbers (not strings)")
        else:
            print(f"[✗] Wrong format! Expected 6+ columns, got {len(parts)}")
            print(f"    Sample: {first_line}")
            print("\n    Expected format:")
            print("    Time(s), Voltage(V), Current(A), Temp(C), SoC(%), Energy(Wh)")

print("\n" + "=" * 70)
print("[TEST COMPLETE]")
print("=" * 70)
