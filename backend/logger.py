"""
Battery Data Logger
Collects sensor data from Arduino and saves to CSV file for ML training

Purpose:
- Run this when you want to collect training data
- Saves data to log.csv for later model training
- Run for 10-30 minutes to collect good dataset
"""

import serial
import time
from datetime import datetime
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = "COM10"          # Change to your Arduino port
BAUD_RATE = 9600
OUTPUT_FILE = "log.csv"     # Output CSV file

print("=" * 70)
print("📊 BATTERY DATA LOGGER")
print("=" * 70)

# Connect to Arduino
print(f"\n[1] Connecting to Arduino on {COM_PORT}...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print(f"✓ Connected to Arduino on {COM_PORT}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check if Arduino is connected")
    print("   2. Check COM port in Device Manager")
    print("   3. Close Arduino IDE if open")
    exit(1)

# Open CSV file
print(f"\n[2] Opening output file: {OUTPUT_FILE}")
try:
    file = open(OUTPUT_FILE, "w")
    print(f"✓ File opened: {OUTPUT_FILE}")
except Exception as e:
    print(f"❌ Cannot create file: {e}")
    ser.close()
    exit(1)

# Start logging
print("\n[3] Starting data collection...")
print("=" * 70)
print("🟢 LOGGING | Press Ctrl+C to stop")
print("=" * 70)

sample_count = 0
start_time = time.time()

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            if line:
                # Print to console
                print(line)
                
                # Write to file
                file.write(line + "\n")
                file.flush()  # Ensure data is written immediately
                
                sample_count += 1
                
except KeyboardInterrupt:
    print("\n\n⏹️  Logging stopped by user")
    
finally:
    # Close connections
    file.close()
    ser.close()
    
    # Statistics
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("📊 LOGGING SUMMARY")
    print("=" * 70)
    print(f"  • Samples Collected : {sample_count}")
    print(f"  • Duration          : {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
    print(f"  • File Saved        : {OUTPUT_FILE}")
    print(f"  • File Size         : ~{sample_count * 50} bytes")
    print("=" * 70)
    print("\n✅ Data collection complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Run: python train_model_enhanced.py")
    print(f"   2. This will train a new model with your data")
