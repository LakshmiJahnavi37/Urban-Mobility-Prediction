# Traffic Flow Prediction (AI133)

Branch: `Traffic-Flow-Prediction_AI133` | Part of the **Urban-Mobility-Prediction** project

## Problem
Predict the number of vehicles passing a junction in a given hour, so congestion can be anticipated and traffic managed better.

## Dataset
Kaggle - **Traffic Prediction Dataset** (`data/raw/traffic.csv`, ~1.8 MB, 48,120 hourly records, Nov 2015 - Jun 2017)

| Column | Description |
|---|---|
| DateTime | Hourly timestamp |
| Junction | Junction number (1-4) |
| Vehicles | Vehicles counted in that hour (target) |
| ID | Unique id (dropped) |

## Approach
1. **Preprocessing** (`src/preprocess.py`): parse dates, drop `ID`, remove duplicates/missing, sort per junction
2. **Feature engineering** (`src/features.py`): hour, day of week, day, month, weekend, rush hour, junction one-hot, lag_1 / lag_24 / lag_168 (computed per junction)
3. **Models** (`src/train.py`): Linear Regression (baseline), Random Forest, XGBoost
4. **Evaluation**: MAE, RMSE, R2 using a time-based split (first 80% of timeline = train, last 20% = test)

## Results
Fill from `outputs/results.csv` after running `train.py`:

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | | | |
| Random Forest | | | |
| XGBoost | | | |

Per-junction metrics: `outputs/results_per_junction.csv`. Plots: `outputs/plots/`.

## Folder structure
```
data/raw/          original dataset (traffic.csv)
data/processed/    cleaned data
notebooks/         EDA + modelling notebook
src/               preprocess, features, train, predict
models/            saved best model (traffic_model.pkl)
outputs/           results + plots
app.py             Streamlit demo
```

## How to run
```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
streamlit run app.py
```

## Integration with the main web app
```python
from src.predict import predict_traffic
# junction, hour, day_of_week(0=Mon), month, day, lag_1, lag_24, lag_168
predict_traffic(1, 9, 1, 6, 15, 50, 52, 48)
```
