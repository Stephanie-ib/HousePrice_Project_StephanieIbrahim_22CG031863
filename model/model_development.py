"""
House Price Prediction - Model Development
===========================================
This script trains a Linear Regression model using 6 selected features
from the Kaggle House Prices dataset.

Author: [Your Name]
Matric No: [Your Matric Number]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL DEVELOPMENT")
print("=" * 60)

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print("\n[1] Loading Dataset...")

# Download dataset from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Place 'train.csv' in the same directory as this script

try:
    df = pd.read_csv('train.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
except FileNotFoundError:
    print("ERROR: train.csv not found!")
    print("Please download it from Kaggle and place it in the /model/ directory")
    exit()

# ============================================
# STEP 2: SELECT FEATURES
# ============================================
print("\n[2] Selecting Features...")

# Selected 6 features from the allowed list
selected_features = [
    'OverallQual',     # Overall material and finish quality
    'GrLivArea',       # Above grade living area (sq ft)
    'TotalBsmtSF',     # Total basement area (sq ft)
    'GarageCars',      # Garage capacity in cars
    'YearBuilt',       # Original construction date
    'Neighborhood'     # Physical location (categorical)
]

target = 'SalePrice'

# Create a subset with selected features + target
df_subset = df[selected_features + [target]].copy()
print(f"✓ Selected {len(selected_features)} features:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i}. {feature}")

# ============================================
# STEP 3: HANDLE MISSING VALUES
# ============================================
print("\n[3] Handling Missing Values...")

print(f"Missing values before cleaning:")
missing = df_subset.isnull().sum()
for col in missing[missing > 0].index:
    print(f"  {col}: {missing[col]} missing")

# Fill missing numerical values with median
numerical_cols = ['TotalBsmtSF', 'GarageCars']
for col in numerical_cols:
    if df_subset[col].isnull().sum() > 0:
        median_val = df_subset[col].median()
        df_subset[col].fillna(median_val, inplace=True)
        print(f"✓ Filled {col} with median: {median_val}")

# Drop rows with missing target values
df_subset.dropna(subset=[target], inplace=True)

print(f"✓ Final dataset shape: {df_subset.shape}")

# ============================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ============================================
print("\n[4] Encoding Categorical Variables...")

# Encode 'Neighborhood' using Label Encoding
le = LabelEncoder()
df_subset['Neighborhood_Encoded'] = le.fit_transform(df_subset['Neighborhood'])

print(f"✓ Encoded 'Neighborhood' into numerical values")
print(f"  Unique neighborhoods: {len(le.classes_)}")

# Drop original categorical column
df_subset.drop('Neighborhood', axis=1, inplace=True)

# Save the encoder for later use in predictions
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✓ Label encoder saved as 'label_encoder.pkl'")

# ============================================
# STEP 5: PREPARE FEATURES AND TARGET
# ============================================
print("\n[5] Preparing Features and Target...")

X = df_subset.drop(target, axis=1)
y = df_subset[target]

print(f"✓ Features (X): {X.shape}")
print(f"✓ Target (y): {y.shape}")

# ============================================
# STEP 6: SPLIT DATA
# ============================================
print("\n[6] Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# ============================================
# STEP 7: TRAIN MODEL
# ============================================
print("\n[7] Training Linear Regression Model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("✓ Model trained successfully!")

# ============================================
# STEP 8: EVALUATE MODEL
# ============================================
print("\n[8] Evaluating Model Performance...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
print(f"Mean Absolute Error (MAE):  ${mae:,.2f}")
print(f"Mean Squared Error (MSE):   ${mse:,.2f}")
print(f"Root Mean Squared Error:    ${rmse:,.2f}")
print(f"R² Score:                   {r2:.4f}")
print("=" * 60)

# ============================================
# STEP 9: SAVE MODEL
# ============================================
print("\n[9] Saving Model...")

with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✓ Model saved as 'house_price_model.pkl'")

# ============================================
# STEP 10: TEST MODEL RELOADING
# ============================================
print("\n[10] Testing Model Reload (WITHOUT Retraining)...")

with open('house_price_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Test prediction with loaded model
test_sample = X_test.iloc[0:1]
original_prediction = model.predict(test_sample)[0]
loaded_prediction = loaded_model.predict(test_sample)[0]

print(f"✓ Original model prediction: ${original_prediction:,.2f}")
print(f"✓ Loaded model prediction:   ${loaded_prediction:,.2f}")
print(f"✓ Match: {np.isclose(original_prediction, loaded_prediction)}")

print("\n" + "=" * 60)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nNext Steps:")
print("1. Copy 'house_price_model.pkl' to the main project directory")
print("2. Copy 'label_encoder.pkl' to the main project directory")
print("3. Run the Flask app using: python app.py")
print("=" * 60)
