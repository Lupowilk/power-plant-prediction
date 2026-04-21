# Section 1: Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Section 2: Load training data
print("Loading training data...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.columns.tolist()}")

# Section 3: Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model trained!")

# Section 4: 5-fold Cross-Validation
print("\nPerforming 5-fold cross-validation...")

# Calculate negative MSE (sklearn convention)
cv_mse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_mse)

# Calculate MAE
cv_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae = -cv_mae

# Calculate R²
cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

# Print results
print("\n" + "="*50)
print("Cross-Validation Results (5-fold):")
print("="*50)
print(f"RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")
print(f"MAE:  {cv_mae.mean():.2f} (+/- {cv_mae.std():.2f})")
print(f"R²:   {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
print("="*50)

# Section 5: Save the trained model
print("\nSaving model...")
joblib.dump(model, 'models/random_forest_model.pkl')
print("✅ Model saved to models/random_forest_model.pkl")

print("\n" + "="*50)
print("Random Forest Model Training Complete!")
print("="*50)
