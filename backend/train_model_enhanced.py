"""
Enhanced Battery SOH Prediction Model Training Script
Features:
- Advanced feature engineering
- Cross-validation
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Comprehensive model evaluation
- Model persistence with metadata
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

os.chdir(Path(__file__).resolve().parents[1])

print("=" * 70)
print("🔋 BATTERY SOH PREDICTION - ML MODEL TRAINING")
print("=" * 70)

# -------------------------------
# 1. LOAD DATA
# -------------------------------
print("\n[1] Loading data from log.csv...")
df = pd.read_csv("log.csv")
print(f"✓ Loaded {len(df)} samples")
print(f"✓ Columns: {list(df.columns)}")

# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------
print("\n[2] Engineering features...")

# Basic power calculation
df["Power"] = df["Voltage(V)"] * df["Current(A)"]

# First-order derivatives (rate of change)
df["dV"] = df["Voltage(V)"].diff().fillna(0)
df["dI"] = df["Current(A)"].diff().fillna(0)
df["dTemp"] = df["Temp(C)"].diff().fillna(0)
df["dSoC"] = df["SoC(%)"].diff().fillna(0)
df["dEnergy"] = df["Energy(Wh)"].diff().fillna(0)

# Internal resistance (Ohm's law approximation)
df["Internal_R"] = df["dV"] / df["dI"].replace(0, np.nan)
df["Internal_R"] = df["Internal_R"].fillna(method="bfill").fillna(method="ffill")
df["Internal_R"] = df["Internal_R"].clip(lower=0, upper=2.0)  # Physical limits

# Cumulative energy consumption
df["Ah_used"] = df["Energy(Wh)"] / df["Voltage(V)"].replace(0, np.nan)
df["Ah_used"] = df["Ah_used"].fillna(0)

# Rolling statistics (smoothed features)
window = 10
df["V_rolling_mean"] = df["Voltage(V)"].rolling(window=window, min_periods=1).mean()
df["V_rolling_std"] = df["Voltage(V)"].rolling(window=window, min_periods=1).std().fillna(0)
df["I_rolling_mean"] = df["Current(A)"].rolling(window=window, min_periods=1).mean()
df["Temp_rolling_mean"] = df["Temp(C)"].rolling(window=window, min_periods=1).mean()

# Advanced features
df["Power_density"] = df["Power"] / (df["Voltage(V)"] + 1e-6)  # Normalized power
df["Thermal_stress"] = df["Temp(C)"] * df["Current(A)"]  # Temperature-current interaction
df["Voltage_efficiency"] = df["Voltage(V)"] / df["Voltage(V)"].max()  # Voltage drop indicator

print(f"✓ Created {len(df.columns) - 6} engineered features")

# -------------------------------
# 3. TARGET VARIABLE (SOH)
# -------------------------------
print("\n[3] Computing SOH target...")

battery_capacity_Ah = 2.2  # Your battery rated capacity
df["SOH"] = 100 - ((df["Ah_used"] / battery_capacity_Ah) * 100)
df["SOH"] = df["SOH"].clip(lower=0, upper=100)

print(f"✓ SOH range: {df['SOH'].min():.2f}% - {df['SOH'].max():.2f}%")
print(f"✓ Mean SOH: {df['SOH'].mean():.2f}%")

# -------------------------------
# 4. PREPARE FEATURES
# -------------------------------
print("\n[4] Preparing feature matrix...")

features = [
    # Core sensor readings
    "Voltage(V)",
    "Current(A)",
    "Temp(C)",
    "SoC(%)",
    "Energy(Wh)",
    
    # Derived features
    "Power",
    "Internal_R",
    "Ah_used",
    
    # Rate of change features
    "dV",
    "dI",
    "dTemp",
    "dSoC",
    "dEnergy",
    
    # Rolling statistics
    "V_rolling_mean",
    "V_rolling_std",
    "I_rolling_mean",
    "Temp_rolling_mean",
    
    # Advanced features
    "Power_density",
    "Thermal_stress",
    "Voltage_efficiency"
]

X = df[features].copy()
y = df["SOH"].copy()

# Handle any remaining NaN or inf values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(method="bfill").fillna(method="ffill").fillna(0)

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target vector shape: {y.shape}")

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
print("\n[5] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# -------------------------------
# 6. HYPERPARAMETER TUNING
# -------------------------------
print("\n[6] Performing hyperparameter tuning...")
print("   (This may take a few minutes...)")

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"✓ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  • {param}: {value}")

# Use the best model
model = grid_search.best_estimator_

# -------------------------------
# 7. CROSS-VALIDATION
# -------------------------------
print("\n[7] Performing cross-validation...")

cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

cv_rmse_scores = np.sqrt(-cv_scores)
print(f"✓ Cross-validation RMSE scores: {cv_rmse_scores}")
print(f"✓ Mean CV RMSE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")

# -------------------------------
# 8. MODEL EVALUATION
# -------------------------------
print("\n[8] Evaluating model performance...")

# Training performance
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Testing performance
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n📊 TRAINING SET METRICS:")
print(f"  • MSE:  {train_mse:.4f}")
print(f"  • RMSE: {train_rmse:.4f}")
print(f"  • MAE:  {train_mae:.4f}")
print(f"  • R²:   {train_r2:.4f}")

print("\n📊 TESTING SET METRICS:")
print(f"  • MSE:  {test_mse:.4f}")
print(f"  • RMSE: {test_rmse:.4f}")
print(f"  • MAE:  {test_mae:.4f}")
print(f"  • R²:   {test_r2:.4f}")

# Prediction accuracy percentage
accuracy_within_1 = np.mean(np.abs(y_test - y_test_pred) <= 1.0) * 100
accuracy_within_2 = np.mean(np.abs(y_test - y_test_pred) <= 2.0) * 100

print(f"\n🎯 PREDICTION ACCURACY:")
print(f"  • Within ±1%: {accuracy_within_1:.2f}%")
print(f"  • Within ±2%: {accuracy_within_2:.2f}%")

# -------------------------------
# 9. FEATURE IMPORTANCE
# -------------------------------
print("\n[9] Analyzing feature importance...")

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:20s} : {row['importance']:.4f}")

# -------------------------------
# 10. SAVE MODEL
# -------------------------------
print("\n[10] Saving model and metadata...")

# Save the model
joblib.dump(model, "battery_soh_model.pkl")
print("✓ Model saved as: battery_soh_model.pkl")

# Save feature list
joblib.dump(features, "model_features.pkl")
print("✓ Features saved as: model_features.pkl")

# Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)
print("✓ Feature importance saved as: feature_importance.csv")

# Save metadata
metadata = {
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_samples": len(df),
    "n_features": len(features),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "best_params": grid_search.best_params_,
    "cv_rmse_mean": float(cv_rmse_scores.mean()),
    "cv_rmse_std": float(cv_rmse_scores.std()),
    "train_rmse": float(train_rmse),
    "train_r2": float(train_r2),
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2),
    "test_mae": float(test_mae),
    "accuracy_within_1pct": float(accuracy_within_1),
    "accuracy_within_2pct": float(accuracy_within_2),
    "battery_capacity_Ah": battery_capacity_Ah
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("✓ Metadata saved as: model_metadata.json")

# -------------------------------
# SUMMARY
# -------------------------------
print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📈 Model Performance Summary:")
print(f"  • Test RMSE: {test_rmse:.4f}%")
print(f"  • Test R² Score: {test_r2:.4f}")
print(f"  • Predictions within ±1%: {accuracy_within_1:.2f}%")
print(f"  • Total features used: {len(features)}")
print(f"\n💾 Files created:")
print(f"  • battery_soh_model.pkl")
print(f"  • model_features.pkl")
print(f"  • feature_importance.csv")
print(f"  • model_metadata.json")
print("\n🚀 Ready to use for real-time predictions!")
print("=" * 70)
