"""
Battery SOH Prediction Module
Real-time prediction system for battery State of Health using trained Random Forest model

Functions:
- load_model: Load trained model from disk
- engineer_features: Compute all ML features from raw data
- validate_input: Validate sensor data
- preprocess_live_data: Parse CSV line from Arduino
- predict_soh: Predict SOH using trained model
- BatteryPredictor: Main class for real-time predictions
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

os.chdir(Path(__file__).resolve().parents[1])


class BatteryPredictor:
    """
    Main predictor class for battery SOH estimation
    Handles feature engineering, validation, and prediction
    """
    
    def __init__(self, model_path: str = "battery_soh_model.pkl"):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model file
        """
        self.model = load_model(model_path)
        self.features = self._load_features()
        self.metadata = self._load_metadata()
        self.previous_data = None
        self.prediction_history = []
        
        print(f"✓ BatteryPredictor initialized")
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Features: {len(self.features)}")
        
    def _load_features(self) -> List[str]:
        """Load feature list from disk"""
        try:
            return joblib.load("model_features.pkl")
        except FileNotFoundError:
            # Default feature list if file not found
            return [
                "Voltage(V)", "Current(A)", "Temp(C)", "SoC(%)", "Energy(Wh)",
                "Power", "Internal_R", "Ah_used", "dV", "dI", "dTemp", "dSoC", "dEnergy",
                "V_rolling_mean", "V_rolling_std", "I_rolling_mean", "Temp_rolling_mean",
                "Power_density", "Thermal_stress", "Voltage_efficiency"
            ]
    
    def _load_metadata(self) -> Dict:
        """Load model metadata"""
        try:
            with open("model_metadata.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def predict_from_csv(self, csv_line: str) -> Tuple[float, Dict]:
        """
        Predict SOH from Arduino CSV line
        
        Args:
            csv_line: CSV string like "1.04,6.427,1.1492,29.33,99.93,0.0021"
            
        Returns:
            Tuple of (predicted_soh, feature_dict)
        """
        # Parse CSV line
        data = preprocess_live_data(csv_line)
        
        # Engineer features
        features = engineer_features(data, self.previous_data)
        
        # Validate
        validate_input(features)
        
        # Predict
        soh = predict_soh(self.model, features, self.features)
        
        # Store for next iteration
        self.previous_data = data.copy()
        self.prediction_history.append(soh)
        
        return soh, features
    
    def predict_from_dict(self, data: Dict) -> float:
        """
        Predict SOH from data dictionary
        
        Args:
            data: Dictionary with keys like 'Voltage(V)', 'Current(A)', etc.
            
        Returns:
            Predicted SOH value
        """
        features = engineer_features(data, self.previous_data)
        validate_input(features)
        soh = predict_soh(self.model, features, self.features)
        
        self.previous_data = data.copy()
        self.prediction_history.append(soh)
        
        return soh
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        if not self.prediction_history:
            return {}
        
        return {
            "mean_soh": np.mean(self.prediction_history),
            "min_soh": np.min(self.prediction_history),
            "max_soh": np.max(self.prediction_history),
            "std_soh": np.std(self.prediction_history),
            "latest_soh": self.prediction_history[-1],
            "n_predictions": len(self.prediction_history)
        }
    
    def reset(self):
        """Reset predictor state"""
        self.previous_data = None
        self.prediction_history = []


def load_model(model_path: str = "battery_soh_model.pkl"):
    """
    Load trained Random Forest model from disk
    
    Args:
        model_path: Path to .pkl model file
        
    Returns:
        Trained sklearn model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model file is corrupted
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


def engineer_features(data: Dict, prev_data: Optional[Dict] = None) -> Dict:
    """
    Engineer all features required for ML prediction
    
    Args:
        data: Current sensor readings
        prev_data: Previous sensor readings (for derivatives)
        
    Returns:
        Dictionary with all engineered features
    """
    features = data.copy()
    
    # Basic power calculation
    features['Power'] = data['Voltage(V)'] * data['Current(A)']
    
    # Compute Ah used
    features['Ah_used'] = data['Energy(Wh)'] / (data['Voltage(V)'] + 1e-6)
    
    # Rate of change features (derivatives)
    if prev_data is not None:
        features['dV'] = data['Voltage(V)'] - prev_data.get('Voltage(V)', data['Voltage(V)'])
        features['dI'] = data['Current(A)'] - prev_data.get('Current(A)', data['Current(A)'])
        features['dTemp'] = data['Temp(C)'] - prev_data.get('Temp(C)', data['Temp(C)'])
        features['dSoC'] = data['SoC(%)'] - prev_data.get('SoC(%)', data['SoC(%)'])
        features['dEnergy'] = data['Energy(Wh)'] - prev_data.get('Energy(Wh)', data['Energy(Wh)'])
    else:
        # No previous data, set derivatives to zero
        features['dV'] = 0.0
        features['dI'] = 0.0
        features['dTemp'] = 0.0
        features['dSoC'] = 0.0
        features['dEnergy'] = 0.0
    
    # Internal resistance (Ohm's law: R = dV/dI)
    if abs(features['dI']) > 1e-6:
        features['Internal_R'] = abs(features['dV'] / features['dI'])
    else:
        # Use default or previous value if current didn't change
        if prev_data is not None and 'Internal_R' in prev_data:
            features['Internal_R'] = prev_data['Internal_R']
        else:
            features['Internal_R'] = 0.5  # Default value
    
    # Clip internal resistance to physical limits
    features['Internal_R'] = np.clip(features['Internal_R'], 0.0, 2.0)
    
    # Rolling statistics (approximated for single sample)
    # In real-time, you'd maintain a buffer of recent values
    features['V_rolling_mean'] = data['Voltage(V)']
    features['V_rolling_std'] = 0.0  # Can't compute std from single sample
    features['I_rolling_mean'] = data['Current(A)']
    features['Temp_rolling_mean'] = data['Temp(C)']
    
    # Advanced features
    features['Power_density'] = features['Power'] / (data['Voltage(V)'] + 1e-6)
    features['Thermal_stress'] = data['Temp(C)'] * data['Current(A)']
    features['Voltage_efficiency'] = data['Voltage(V)'] / 6.5  # Assuming max voltage ~6.5V
    
    # Handle any NaN or inf
    for key in features:
        if isinstance(features[key], float):
            if np.isnan(features[key]) or np.isinf(features[key]):
                features[key] = 0.0
    
    return features


def validate_input(data: Dict) -> bool:
    """
    Validate sensor input data
    
    Args:
        data: Dictionary with sensor readings
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If data is invalid
    """
    required_fields = ['Voltage(V)', 'Current(A)', 'Temp(C)', 'SoC(%)', 'Energy(Wh)']
    
    # Check all required fields present
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate voltage (should be positive, reasonable range)
    if data['Voltage(V)'] < 0 or data['Voltage(V)'] > 10:
        raise ValueError(f"Invalid voltage: {data['Voltage(V)']} V")
    
    # Validate current (should be non-negative in discharge)
    if data['Current(A)'] < 0 or data['Current(A)'] > 5:
        raise ValueError(f"Invalid current: {data['Current(A)']} A")
    
    # Validate temperature (reasonable range)
    if data['Temp(C)'] < -20 or data['Temp(C)'] > 80:
        raise ValueError(f"Invalid temperature: {data['Temp(C)']} °C")
    
    # Validate SoC (should be 0-100%)
    if data['SoC(%)'] < 0 or data['SoC(%)'] > 100:
        raise ValueError(f"Invalid SoC: {data['SoC(%)']} %")
    
    # Validate Energy (should be non-negative)
    if data['Energy(Wh)'] < 0:
        raise ValueError(f"Invalid energy: {data['Energy(Wh)']} Wh")
    
    return True


def preprocess_live_data(csv_line: str) -> Dict:
    """
    Parse CSV line from Arduino serial output
    
    Args:
        csv_line: String like "1.04,6.427,1.1492,29.33,99.93,0.0021"
        
    Returns:
        Dictionary with parsed values
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        parts = csv_line.strip().split(',')
        
        if len(parts) != 6:
            raise ValueError(f"Expected 6 values, got {len(parts)}")
        
        data = {
            'Time(s)': float(parts[0]),
            'Voltage(V)': float(parts[1]),
            'Current(A)': float(parts[2]),
            'Temp(C)': float(parts[3]),
            'SoC(%)': float(parts[4]),
            'Energy(Wh)': float(parts[5])
        }
        
        return data
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse CSV line: {str(e)}")


def predict_soh(model, features: Dict, feature_list: List[str]) -> float:
    """
    Predict SOH using trained model
    
    Args:
        model: Trained sklearn model
        features: Dictionary with all features
        feature_list: List of feature names in correct order
        
    Returns:
        Predicted SOH value (0-100%)
    """
    # Create feature vector with column names to match training schema
    row = {f: features.get(f, 0.0) for f in feature_list}
    X = pd.DataFrame([row], columns=feature_list)

    # Handle any remaining NaN or inf
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # Predict
    soh_pred = model.predict(X)[0]
    
    # Clip to valid range
    soh_pred = np.clip(soh_pred, 0.0, 100.0)
    
    return float(soh_pred)


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("🔋 BATTERY SOH PREDICTOR - TESTING")
    print("=" * 70)
    
    # Initialize predictor
    try:
        predictor = BatteryPredictor()
    except FileNotFoundError:
        print("\n❌ Error: Model file not found!")
        print("   Please run train_model_enhanced.py first to train the model.")
        sys.exit(1)
    
    # Test with sample data
    print("\n[TEST] Sample prediction...")
    
    sample_csv = "10.11,6.427,1.1757,32.26,99.35,0.021"
    
    try:
        soh, features = predictor.predict_from_csv(sample_csv)
        
        print(f"\n✓ Input: {sample_csv}")
        print(f"\n✓ Predicted SOH: {soh:.2f}%")
        
        print(f"\n✓ Key Features:")
        print(f"  • Voltage: {features['Voltage(V)']:.3f} V")
        print(f"  • Current: {features['Current(A)']:.4f} A")
        print(f"  • Temperature: {features['Temp(C)']:.2f} °C")
        print(f"  • Power: {features['Power']:.4f} W")
        print(f"  • Internal R: {features['Internal_R']:.4f} Ω")
        
        # Test multiple predictions
        print("\n[TEST] Multiple predictions...")
        test_lines = [
            "20.2,6.427,1.1757,29.81,98.7,0.0419",
            "30.28,6.403,1.1492,29.33,98.05,0.0625",
            "40.37,6.378,1.07,29.33,97.42,0.0829"
        ]
        
        for line in test_lines:
            soh, _ = predictor.predict_from_csv(line)
            print(f"  • {line[:30]}... → SOH: {soh:.2f}%")
        
        # Show statistics
        stats = predictor.get_statistics()
        print(f"\n📊 Prediction Statistics:")
        print(f"  • Mean SOH: {stats['mean_soh']:.2f}%")
        print(f"  • Min SOH: {stats['min_soh']:.2f}%")
        print(f"  • Max SOH: {stats['max_soh']:.2f}%")
        print(f"  • Std Dev: {stats['std_soh']:.4f}%")
        print(f"  • Total predictions: {stats['n_predictions']}")
        
        print("\n" + "=" * 70)
        print("✅ PREDICTOR TEST COMPLETE!")
        print("=" * 70)
        print("\n💡 Usage in your code:")
        print("   from predict import BatteryPredictor")
        print("   predictor = BatteryPredictor()")
        print("   soh, features = predictor.predict_from_csv(csv_line)")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        sys.exit(1)
