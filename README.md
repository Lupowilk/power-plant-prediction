# Power Plant Energy Prediction

Predict electrical energy output of a Combined Cycle Power Plant from ambient conditions.

## Dataset
- 9,568 hourly observations from UCI ML Repository
- 4 features: Temperature, Vacuum, Pressure, Humidity
- Target: Power output (420-495 MW)

## Results

**Random Forest Model:**
- RMSE: 3.24 MW (<1% error)
- R²: 96.37%
- Temperature accounts for 90% of predictions

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python src/data_split.py
python src/train_random_forest.py
jupyter notebook  # View notebooks/
```

## Files
- `notebooks/` - EDA and evaluation
- `src/` - Training scripts
- `models/` - Saved models
- `results/` - Performance metrics

## Reference
UCI ML Repository: [Combined Cycle Power Plant Dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
