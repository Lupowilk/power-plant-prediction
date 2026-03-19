import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/raw/power_plant.csv')
print(f"Total samples: {len(df)}")


# Separate features (X) and target (y)
X = df[['AT', 'V', 'AP', 'RH']] #features
y = df['PE'] #target


# Split into train (80%) and test (20%)
# random_state=42 ensures same split every time (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state=42
)


print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

# Save splits to processed data folder
print("\nSaving splits...")
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("✅ Data split complete!")
print("\nValidation strategy: 5-fold cross-validation will be used during model training")
