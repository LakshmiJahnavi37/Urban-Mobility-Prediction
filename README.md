# Urban Mobility Prediction Platform

An integrated AI-driven platform for intelligent urban transportation and transit analytics.

---

## AI305 Use Case: Public Transport Daily Passenger Demand Prediction

- **Module Name:** Demand Intelligence — Public Transport Daily Passenger Demand Forecasting
- **Module Owner:** 24B11AI305 (O. Bheemeswari)
- **Team:** Team 4 (Urban Mobility Prediction)
- **Institution:** Aditya University
- **Branch:** `Public-passenger-Demand-Prediction_AI305`

---

## 1. Overview & Problem Statement

Urban transit agencies and bus operators must plan daily service schedules, driver shifts, and fleet allocations in advance. Without accurate forecasts, routes face overcrowding on high-demand days or wasted operational expenditure with underutilized buses.

The **AI305 Passenger Demand Prediction Module** delivers an end-to-end regression machine learning pipeline that forecasts **daily passenger boardings** (`rides` / `passenger_count`) for specific transit corridors and service dates. Dispatchers and transit planners can utilize these predictions to optimize vehicle allocations, adjust headway frequencies, and prepare reserve fleet capacity.

---

## 2. Selected Official Dataset: Chicago CTA Bus Routes Daily Totals

The module is built and evaluated upon the official open dataset from the **City of Chicago Data Portal**:
- **Dataset Title:** CTA - Ridership - Bus Routes - Daily Totals by Route
- **Dataset Webpage:** [Chicago Data Portal (jyb9-n7fm)](https://data.cityofchicago.org/Transportation/CTA-Ridership-Bus-Routes-Daily-Totals-by-Route/jyb9-n7fm)
- **Direct CSV Download:** `https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv?accessType=DOWNLOAD`
- **SODA API URL:** `https://data.cityofchicago.org/resource/jyb9-n7fm.csv?$limit=25000&$order=date%20DESC`
- **License / Terms:** City of Chicago Open Data Terms of Use (Public Domain / Open Data).

### 2.1 Actual Dataset Columns & Structure
The CTA dataset records daily aggregate boardings per route with the following four native columns:
1. `route` *(String)*: Transit route / corridor identifier (e.g. `3`, `4`, `9`, `20`, `49`, `X9`, `J14`).
2. `date` *(Date / Timestamp)*: Calendar service date (`YYYY-MM-DD`).
3. `daytype` *(String)*: CTA schedule categorization:
   - `W`: Weekday (Regular schedule)
   - `A`: Saturday (Reduced weekend schedule)
   - `U`: Sunday or Official City Holiday (Holiday schedule)
4. `rides` *(Integer)*: Total recorded daily passenger boardings on that route.

### 2.2 Ingested Dataset Profile
- **Total Records:** 10,000 authentic transit observations
- **Date Range:** May 6, 2026 to July 31, 2026
- **Unique Bus Routes:** 127 transit corridors across Chicago
- **Daily Ridership Stats:** Mean: 4,951.6 rides | Median: 3,923.0 rides | Min: 0 rides | Max: 25,627.0 rides
- **Missing Values:** 0 null values across all columns

---

## 3. Daily Temporal Modeling & Zero Data Leakage Architecture

Because the official dataset provides daily aggregates (not hourly timestamps), the modeling pipeline strictly enforces **daily route-level forecasting** without unsupported hourly assumptions:

### 3.1 Feature Engineering
1. **Calendar Features (Strictly from Date and Daytype):**
   - `day_of_week`: Day index (0=Monday to 6=Sunday via `Series.dt.day_of_week`)
   - `day`: Day of month (1-31)
   - `month`: Calendar month (1-12)
   - `year`: Service year
   - `is_weekend`: Binary flag (1 if `daytype` is `A` or `U`, or `day_of_week >= 5`)
   - `is_holiday_or_sunday`: Binary flag (1 if `daytype == 'U'`)

2. **Historical Lag Features (Strictly Grouped by Route via `shift` Operations):**
   - `lag_1d_rides`: Prior day's ridership on the same route (`shift(1)`).
   - `trailing_3d_rides`: Trailing 3-day average ridership on the same route (`shift(1).rolling(3).mean()`).
   - `lag_7d_rides`: Prior week same-day ridership on the same route (`shift(7)`).
   - `rolling_mean_7d_rides`: Trailing 7-day average ridership on the same route (`shift(1).rolling(7).mean()`).
   - **Zero Leakage:** All shift and rolling operations exclude the current day's observation (`shift(1)`). Future targets are never accessible to lag features.

3. **Safe Inference & Cold-Start Handling:**
   - At prediction time, historical lag features are auto-populated strictly from historical observations recorded *prior* to the target date.
   - If a route is new or lag features are unavailable/excluded, the pipeline sets them to `NaN`, where the `SimpleImputer(strategy="median")` (fitted on training data) safely fills them without pipeline errors or lookahead leakage.

### 3.2 Chronological Train/Test Splitting
In accordance with time-series forecasting integrity, data is split **chronologically by date**:
- **Training Set (Earlier 80% Dates):** 7,867 records spanning `2026-05-06` to `2026-07-13`.
- **Held-Out Test Set (Latest 20% Dates):** 2,133 records spanning `2026-07-14` to `2026-07-31`.
- **Principle:** Future observations are strictly sequestered and never used to fit preprocessors or train models.

---

## 4. Model Training & Measured Evaluation Metrics

Both models were trained on the training partition and evaluated on the unseen held-out chronological test partition. No metrics are invented or hard-coded.

| Model | Held-out MAE (Rides) | Held-out RMSE (Rides) | Held-out $R^2$ Score | Status |
|---|---|---|---|---|
| **Baseline (Linear Regression)** | 375.5992 | 551.8677 | 0.9853 | Baseline Benchmark |
| **Random Forest Regressor** | **306.8332** | **527.9468** | **0.9866** | **Champion Model** |

### Metric Formulations:
- **MAE (Mean Absolute Error):** $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| = 306.8$ daily passenger boardings.
- **RMSE (Root Mean Squared Error):** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} = 527.9$ daily passenger boardings.
- **$R^2$ Score (Coefficient of Determination):** $R^2 = 0.9866$ (explains 98.66% of held-out daily variance).

---

## 5. Saved Artifacts

Artifacts are persisted in `models/`:
- `models/best_model.joblib`: Fitted Random Forest Regressor.
- `models/preprocessor.joblib`: Fitted `ColumnTransformer` (SimpleImputer, StandardScaler, OneHotEncoder).
- `models/baseline_model.joblib`: Fitted Linear Regression model.
- `models/rf_model.joblib`: Fitted Random Forest Regressor.
- `models/metrics_report.json`: Chronological evaluation report with exact measured MAE, RMSE, and $R^2$.
- `models/route_history.json`: Historical route baselines strictly extracted from observations for live inference.

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### Installation
```bash
git clone https://github.com/LakshmiJahnavi37/Urban-Mobility-Prediction.git
cd Urban-Mobility-Prediction
git checkout Public-passenger-Demand-Prediction_AI305

pip install -r requirements.txt
```

---

## 7. Running the Automated Tests

Run the test suite to verify data validation, leak-free lag calculation, chronological splitting, metrics calculation, and inference:
```bash
python -m pytest tests -v
```

---

## 8. Running the Streamlit Web Application

Launch the interactive web application locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Key Application Features:
1. **Live Daily Demand Prediction:**
   - Select transit route, service date, and day schedule type (`W`, `A`, `U`).
   - Choose historical lag mode: Auto-calculate from history before date, enter custom prior-day & trailing-3-day boardings, or exclude lag features.
   - Generates predicted daily boardings, corridor volume tier, dispatch recommendation, and required bus fleet estimation (~560 daily passenger capacity/bus).
2. **Dataset & Chronological Model Training:**
   - Fetch authentic Chicago CTA records directly from the Chicago Data Portal.
   - Profile row count, date range, unique routes, and target distribution.
   - Execute chronological model training and view live metrics table.
3. **Evaluation & Comparative Diagnostics:**
   - Review held-out MAE, RMSE, and $R^2$ scores and visual comparison charts.
4. **Schema Documentation:**
   - Detailed schema specification and downloadable template CSV.

---

## 9. Model Training via CLI

To train models directly from the terminal on an authentic dataset:
```bash
python -m ai305_passenger_demand.train --dataset-path data/passenger_demand.csv --test-size 0.2
```

---

## 10. Deployment on Streamlit Community Cloud

1. Commit and push to your team repository:
   ```bash
   git push team Public-passenger-Demand-Prediction_AI305
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Select `app.py` as the entry point and deploy.