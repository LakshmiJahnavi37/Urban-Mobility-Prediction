# AI305: Public Transport Passenger Demand Prediction
**Module Specification, Data Architecture & Run Guide**

- **Module Owner:** 24B11AI305 (O. Bheemeswari)  
- **Team:** Team 4 (Urban Mobility Prediction Platform)  
- **Institution:** Aditya University  
- **Current Branch:** `Public-passenger-Demand-Prediction_AI305`

---

## 1. Problem Statement & Objective
In urban transportation networks, public transit operators face significant uncertainty in estimating route-level passenger demand. Static schedules lead to severe vehicle overcrowding during morning/evening commute peaks and underutilized, costly empty runs during off-peak hours.

The **AI305 Demand Intelligence Module** builds a machine learning regression pipeline to predict continuous passenger demand (`passenger_count`) for specified routes, hours, and contextual weather/event conditions. This enables transit authorities to proactively dispatch standby buses, adjust vehicle dispatch frequency, and coordinate with traffic signal controllers before congestion builds.

---

## 2. Dataset Inspection & Schema Specification

### 2.1 Inspection Report
Inspection of the initial repository commits, local working tree, and external drives confirmed that **no real public transport passenger demand dataset existed** in the repository. In accordance with project integrity standards:
- No artificial training data has been fabricated.
- No synthetic evaluation numbers or fake accuracy metrics have been passed off as real.
- The pipeline architecture, data loader, preprocessor, model training, persistence, test suite, and interactive Streamlit UI are completely implemented and ready for dataset ingestion.

### 2.2 Proposed Dataset Format
The training dataset must be a CSV file placed at `data/passenger_demand.csv` (or uploaded directly via the Streamlit UI). A reference schema template is provided in `data/passenger_demand_template.csv`.

#### Schema Definition:
| Column Name | Data Type | Requirement | Description & Allowed Values |
|---|---|---|---|
| `passenger_count` | Integer / Float | **Target (Mandatory)** | Number of passengers boarding/demanding transit during the time window (non-negative). |
| `route_id` | String / Categorical | Feature | Unique transit corridor/line identifier (e.g. `Route_101`, `Route_201`). |
| `timestamp` | Datetime String | Feature | ISO timestamp `YYYY-MM-DD HH:MM:SS` from which `hour`, `day_of_week`, `month`, and `is_rush_hour` are derived. |
| `weather_condition` | String / Categorical | Feature | Weather state: `Clear`, `Partly Cloudy`, `Rainy`, `Foggy`, `Stormy`. |
| `temperature_c` | Float | Feature | Ambient temperature in Celsius (e.g. `15.0` to `45.0`). |
| `is_holiday` | Binary (0 / 1) | Feature | `1` if the date is an official public holiday, else `0`. |
| `is_weekend` | Binary (0 / 1) | Feature | `1` for Saturday/Sunday, else `0` (auto-derived if omitted). |
| `is_special_event` | Binary (0 / 1) | Feature | `1` if a major sports match, concert, or festival occurs near the route corridor. |
| `lag_1h_passenger_count` | Numeric | Feature | Observed passenger demand in the immediately preceding hour. |
| `lag_24h_passenger_count` | Numeric | Feature | Observed passenger demand at the same hour on the previous day. |
| `rolling_avg_3h` | Numeric | Feature | 3-hour rolling average passenger demand. |

---

## 3. Machine Learning Architecture & Leak-Free Preprocessing

```
Raw CSV Dataset (data/passenger_demand.csv)
                      │
                      ▼
            Data Validation & Loader
                      │
                      ▼
         Train / Held-Out Test Split (80% / 20%)
           ┌──────────┴──────────┐
           ▼                     ▼
        X_train               X_test
           │                     │
           ▼                     │
 Fit Preprocessor (leak-free)    │
           │                     │
           ├─────────────────────┤
           ▼                     ▼
  X_train_transformed   X_test_transformed
           │                     │
           ▼                     ▼
┌──────────────────────────────┐ │
│ 1. Baseline Linear Regression│ │
│ 2. Random Forest Regressor   │ │
└──────────────┬───────────────┘ │
               │                 │
               ▼                 ▼
          Model Evaluation (MAE, RMSE, R²)
                         │
                         ▼
             Model & Pipeline Persistence
             (models/best_model.joblib)
             (models/preprocessor.joblib)
                         │
                         ▼
        Fast Inference & Streamlit Web UI
```

### 3.1 Leakage Prevention
To guarantee that model evaluation strictly reflects real-world generalization:
1. `train_test_split(test_size=0.2, random_state=42)` is performed **before** any fitting.
2. Numerical imputers (`SimpleImputer(strategy='median')`) and categorical encoders (`OneHotEncoder(handle_unknown='ignore')`) are **fit strictly on `X_train`**.
3. `X_test` and future production inputs are only transformed through the fitted pipeline without recalculating statistics.

---

## 4. Model Training & Evaluation Metrics

### 4.1 Models Trained
1. **Baseline Model:** `LinearRegression()` — Provides an interpretable, linear reference benchmark.
2. **Non-linear Ensemble:** `RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42)` — Captures complex non-linear interactions across weather, calendar, and lag ridership.

### 4.2 Metrics Evaluated on Held-Out Test Set
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **Coefficient of Determination ($R^2$ Score):**
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

---

## 5. Reproducible Setup & Run Instructions

### 5.1 Installation
Install required dependencies:
```bash
pip install -r requirements.txt
```

### 5.2 Running Automated Unit Tests
Run the test suite to verify data validation, leak-free preprocessing, metrics computation, and model saving:
```bash
python -m unittest tests/test_ai305_pipeline.py -v
```

### 5.3 Training the Models (CLI)
Once your real dataset is placed at `data/passenger_demand.csv`:
```bash
python -m ai305_passenger_demand.train --dataset-path data/passenger_demand.csv
```
This will train both models, compare test metrics, select the best model, and save:
- `models/preprocessor.joblib`
- `models/best_model.joblib`
- `models/metrics_report.json`

### 5.4 Launching the Interactive Streamlit Dashboard
Launch the interactive web UI:
```bash
streamlit run app.py
```
From the dashboard, you can:
1. View dataset schema and download the template CSV.
2. Upload your dataset file with one click and trigger automated training.
3. Compare model performance metrics and charts.
4. Input route parameters (weather, hour, route, lag demand) to obtain instant predicted passenger demand and transit dispatch recommendations.
