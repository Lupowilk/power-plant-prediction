import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Section 2: Load training data
print("Loading training data...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.columns.tolist()}")

# Section 3: Train Linear Regression model
print("\nTraining Linear Regression model...")
model = LinearRegression()
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
joblib.dump(model, 'models/linear_regression_model.pkl')
print("✅ Model saved to models/linear_regression_model.pkl")

print("\n" + "="*50)
print("Linear Regression Model Training Complete!")
print("="*50)
