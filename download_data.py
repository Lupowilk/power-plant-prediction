import pandas as pd

print("Converting Excel to CSV...")

# Read the Excel file
df = pd.read_excel('data/raw/Folds5x2_pp.xlsx')

# Save as CSV
df.to_csv('data/raw/power_plant.csv', index=False)

print(f"\n✅ Conversion successful!")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())
