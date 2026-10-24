"""
AI305: Public Transport Daily Passenger Demand Prediction Dashboard
Part of Team 4 Urban-Mobility-Prediction Platform
Interactive Streamlit Web Application (Configured for Daily Route-Level Forecasting)
"""

import os
# Safeguard Windows OpenBLAS memory allocation
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from datetime import datetime, date
import streamlit as st
import pandas as pd
import numpy as np

import importlib
import ai305_passenger_demand.data_loader
import ai305_passenger_demand.preprocessor
import ai305_passenger_demand.predict
importlib.reload(ai305_passenger_demand.data_loader)
importlib.reload(ai305_passenger_demand.preprocessor)
importlib.reload(ai305_passenger_demand.predict)

from ai305_passenger_demand.data_loader import (
    load_and_validate_dataset,
    get_dataset_summary,
    TARGET_COLUMN
)
from ai305_passenger_demand.train import train_models_pipeline, DEFAULT_MODEL_DIR
from ai305_passenger_demand.predict import (
    PassengerDemandPredictor,
    ModelNotTrainedError,
    HistoricalLookaheadError
)
from ai305_passenger_demand.metrics import format_metrics_comparison
from ai305_passenger_demand.fetch_dataset import download_cta_dataset


# Page Configuration
st.set_page_config(
    page_title="Daily Passenger Demand Prediction | AI305",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, cards, and typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #38bdf8;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e2e8f0;
        margin-bottom: 0.9rem;
        line-height: 1.4;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.12);
        color: #f1f5f9;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    
    .result-box-green {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1rem;
    }
    
    .result-box-blue {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #7dd3fc;
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1rem;
    }
    
    .result-box-orange {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border: 1px solid #fed7aa;
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1rem;
    }
    
    .result-box-red {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1rem;
    }
    
    .status-badge-ok {
        color: #16a34a;
        font-weight: 600;
        background: #dcfce7;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }
    
    .status-badge-warn {
        color: #d97706;
        font-weight: 600;
        background: #fef3c7;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def check_dataset_status():
    """Checks whether local dataset exists and returns path and availability."""
    default_path = "data/passenger_demand.csv"
    uploaded_path = "data/uploaded_passenger_demand.csv"
    
    if os.path.exists(uploaded_path):
        return uploaded_path, True
    elif os.path.exists(default_path):
        return default_path, True
    return default_path, False


def main():
    predictor = PassengerDemandPredictor()
    active_dataset_path, dataset_exists = check_dataset_status()
    models_ready = predictor.is_ready()

    # ==========================================
    # SIDEBAR: PLATFORM OVERVIEW & CONTROLS
    # ==========================================
    with st.sidebar:
        st.markdown("### 🚦 Platform Metadata")
        st.markdown("""
        - **Module:** AI305 Demand Intelligence
        - **Use Case:** Public Transport Daily Passenger Demand
        - **Dataset:** Chicago CTA Bus Routes Daily Totals
        - **Owner:** 24B11AI305 (O. Bheemeswari)
        - **Platform:** Urban-Mobility-Prediction (Team 4)
        - **Institution:** Aditya University
        """)
        
        st.markdown("---")
        st.markdown("### 📡 System Status")
        if dataset_exists:
            st.markdown('Dataset: <span class="status-badge-ok">Available</span>', unsafe_allow_html=True)
        else:
            st.markdown('Dataset: <span class="status-badge-warn">Awaiting CSV</span>', unsafe_allow_html=True)
            
        if models_ready:
            st.markdown('Model Engine: <span class="status-badge-ok">Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('Model Engine: <span class="status-badge-warn">Not Trained</span>', unsafe_allow_html=True)

        st.markdown("---")
        selected_nav = st.radio(
            "Navigation",
            options=[
                "🔮 Live Daily Demand Prediction",
                "📂 Dataset & Model Training",
                "📊 Evaluation & Metrics",
                "📋 Schema & Documentation"
            ],
            index=0
        )
        
        st.markdown("---")
        st.caption("AI305 Daily Transit Ridership Forecasting Engine • v1.1.0")

    # ==========================================
    # HERO HEADER
    # ==========================================
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🚌 Public Transport Daily Passenger Demand Prediction</div>
        <div class="hero-subtitle">
            AI-driven daily route-level transit ridership forecasting for fleet dispatch optimization, service frequency planning, and proactive capacity management.
        </div>
        <div>
            <span class="badge-pill">Use Case: AI305</span>
            <span class="badge-pill">Owner: 24B11AI305 (O. Bheemeswari)</span>
            <span class="badge-pill">Dataset: Chicago CTA Bus Daily Totals</span>
            <span class="badge-pill">Branch: Public-passenger-Demand-Prediction_AI305</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # VIEW 1: LIVE DAILY DEMAND PREDICTION
    # ==========================================
    if selected_nav == "🔮 Live Daily Demand Prediction":
        st.subheader("🔮 Predict Daily Route Passenger Demand")
        st.write("Forecast total daily passenger boardings for a specific transit route and service date.")

        if not models_ready:
            st.warning(
                "⚠️ **No trained model found on disk.**\n\n"
                "In strict accordance with project integrity standards, live predictions require a machine learning "
                "pipeline trained on authentic public transit ridership records. "
                "Please navigate to the **'📂 Dataset & Model Training'** section to ingest the Chicago CTA dataset "
                "and train the Baseline Linear Regression and Random Forest Regressors."
            )

        # Safely extract available routes from predictor route history
        route_hist = getattr(predictor, "route_history", {}) or {}
        available_routes = sorted(list(route_hist.keys())) if route_hist else [
            "3", "4", "9", "20", "22", "49", "53", "66", "79", "85", "95", "X9", "X49", "J14"
        ]

        with st.form("daily_prediction_form"):
            col_in1, col_in2 = st.columns(2)

            with col_in1:
                st.markdown("#### 📍 Route & Schedule Context")
                route_id = st.selectbox(
                    "Bus Route Number",
                    options=available_routes,
                    index=0,
                    help="Official Chicago CTA transit route or corridor identifier."
                )
                
                selected_date = st.date_input("Service Date", value=date.today())
                
                # Determine default daytype from calendar
                weekday_num = selected_date.weekday()
                default_dt_idx = 1 if weekday_num == 5 else (2 if weekday_num == 6 else 0)
                
                daytype = st.selectbox(
                    "Day Schedule Type",
                    options=["W (Weekday Regular)", "A (Saturday Reduced)", "U (Sunday / Holiday)"],
                    index=default_dt_idx,
                    help="CTA schedule type: W=Weekday, A=Saturday, U=Sunday or Official Holiday."
                )
                daytype_code = daytype.split()[0]

            with col_in2:
                st.markdown("#### 📈 Historical Trailing Ridership")
                hist_stats = route_hist.get(route_id, {})
                last_obs_str = hist_stats.get("last_observed_date", "2026-07-31")
                try:
                    last_obs_date = datetime.strptime(str(last_obs_str), "%Y-%m-%d").date()
                except Exception:
                    last_obs_date = date(2026, 7, 31)
                    
                is_historical = selected_date <= last_obs_date
                
                # Check if historical lags can be reconstructed strictly from records before selected_date
                reconstructed_lags = None
                if is_historical:
                    reconstructed_lags = predictor.reconstruct_historical_lags(route_id, str(selected_date))
                    if reconstructed_lags:
                        suggested_lag1 = reconstructed_lags["lag_1d_rides"]
                        suggested_lag3 = reconstructed_lags["trailing_3d_rides"]
                        suggested_lag7 = reconstructed_lags["lag_7d_rides"]
                    else:
                        suggested_lag1 = 5000.0
                        suggested_lag3 = 5000.0
                        suggested_lag7 = 5000.0
                else:
                    suggested_lag1 = hist_stats.get("latest_rides", 5500.0)
                    suggested_lag3 = hist_stats.get("latest_3d_avg", suggested_lag1)
                    suggested_lag7 = hist_stats.get("latest_7d_avg", 5400.0)
                
                lag_mode = st.radio(
                    "Lag Features Selection",
                    options=[
                        "Auto-calculate from historical observations before date",
                        "Enter custom prior-day & trailing-3-day boardings",
                        "Exclude prior lag features (cold start / no recent history)"
                    ],
                    index=0,
                    help="Select how prior-day and trailing-3-day ridership features are supplied."
                )
                
                if lag_mode == "Auto-calculate from historical observations before date":
                    if is_historical:
                        if reconstructed_lags:
                            st.info(
                                f"🕒 **Historical Date Protection Active (`{selected_date}` $\le$ `{last_obs_date}`):** "
                                f"Reconstructed strictly from observations prior to `{selected_date}` (as of {reconstructed_lags['last_available_date']}): "
                                f"Prior Day: **{int(round(suggested_lag1))}** rides | "
                                f"Trailing 3-Day: **{int(round(suggested_lag3))}** rides | "
                                f"Trailing 7-Day: **{int(round(suggested_lag7))}** rides. "
                                f"*(Zero July 31 baseline observations used.)*"
                            )
                            lag_1d_val = suggested_lag1
                            trailing_3d_val = suggested_lag3
                            lag_7d_val = suggested_lag7
                            exclude_lags_val = False
                        else:
                            st.warning(
                                f"⚠️ **Lookahead Protection Alert:** No historical records exist in the dataset strictly prior to `{selected_date}` for Route **{route_id}**. "
                                f"To prevent future data leakage, July 31 values are blocked. Please switch to **'Enter custom prior-day boardings'** or **'Exclude prior lag features'**."
                            )
                            lag_1d_val = None
                            trailing_3d_val = None
                            lag_7d_val = None
                            exclude_lags_val = False
                    else:
                        st.info(
                            f"Route **{route_id}** historical baselines (as of {last_obs_str}): "
                            f"Prior Day: **{int(round(suggested_lag1))}** rides | "
                            f"Trailing 3-Day: **{int(round(suggested_lag3))}** rides | "
                            f"Trailing 7-Day: **{int(round(suggested_lag7))}** rides."
                        )
                        lag_1d_val = suggested_lag1
                        trailing_3d_val = suggested_lag3
                        lag_7d_val = suggested_lag7
                        exclude_lags_val = False
                elif lag_mode == "Enter custom prior-day & trailing-3-day boardings":
                    lag_1d_val = st.number_input(
                        "Prior Day Passenger Boardings (Lag 1-Day)",
                        min_value=0.0, max_value=50000.0, value=float(suggested_lag1), step=50.0,
                        help="Actual passenger boardings observed on this route on the immediately preceding day."
                    )
                    trailing_3d_val = st.number_input(
                        "Trailing 3-Day Mean Boardings",
                        min_value=0.0, max_value=50000.0, value=float(suggested_lag3), step=50.0,
                        help="Rolling average daily ridership over the trailing 3 days."
                    )
                    lag_7d_val = suggested_lag7
                    exclude_lags_val = False
                else:
                    st.caption("ℹ️ Prior lag features excluded. Imputer will fill baseline from training data.")
                    lag_1d_val = None
                    trailing_3d_val = None
                    lag_7d_val = None
                    exclude_lags_val = True

            predict_submitted = st.form_submit_button("🚀 Forecast Daily Passenger Demand", type="primary", use_container_width=True)

        if predict_submitted:
            if not models_ready:
                st.error("Cannot predict: Model is not yet trained. Please provide a dataset and train in 'Dataset & Model Training'.")
            else:
                input_payload = {
                    "route_id": route_id,
                    "date": str(selected_date),
                    "daytype": daytype_code,
                    "lag_1d_rides": lag_1d_val,
                    "trailing_3d_rides": trailing_3d_val,
                    "lag_7d_rides": lag_7d_val,
                    "rolling_mean_7d_rides": lag_7d_val,
                    "exclude_lag_features": exclude_lags_val
                }
                
                try:
                    with st.spinner("Executing leak-free feature transformation and running trained regressor..."):
                        res = predictor.predict_single(input_payload)
                        predicted_count = res["predicted_passenger_count"]
                        tier = res["demand_tier"]
                        rec = res["operational_recommendation"]
                        buses_needed = res["estimated_buses_required"]
                        
                        box_class = "result-box-blue"
                        if "Low" in tier:
                            box_class = "result-box-green"
                        elif "Arterial" in tier:
                            box_class = "result-box-orange"
                        elif "Trunk" in tier:
                            box_class = "result-box-red"
                            
                        st.markdown(f"""
                        <div class="{box_class}">
                            <div style="font-size: 0.9rem; color: #475569; text-transform: uppercase; font-weight: 600;">
                                Daily Ridership Forecast for Route {route_id} • Service Date: {selected_date} ({daytype_code})
                            </div>
                            <div style="font-size: 3.2rem; font-weight: 700; color: #0f172a; margin: 0.3rem 0;">
                                {int(round(predicted_count)):,} <span style="font-size: 1.4rem; font-weight: 400; color: #64748b;">daily passenger boardings</span>
                            </div>
                            <div style="margin-bottom: 0.8rem;">
                                <strong>Corridor Classification:</strong> <span style="font-weight: 600; color: #0284c7;">{tier}</span>
                            </div>
                            <div style="background: white; border-radius: 8px; padding: 0.9rem; border-left: 5px solid #0284c7;">
                                <strong>🚍 Dispatch Recommendation:</strong> {rec}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Fleet Requirement Card
                        st.markdown("#### 🚌 Fleet Requirement & Corridor Load Analysis")
                        c_f1, c_f2, c_f3 = st.columns(3)
                        c_f1.metric("Predicted Boardings", f"{int(round(predicted_count)):,}")
                        c_f2.metric("Estimated Fleet Needed", f"{buses_needed} Buses", help="Calculated based on ~560 daily passenger capacity per standard transit bus.")
                        c_f3.metric("Schedule Profile", f"Schedule {daytype_code}")
                        
                except HistoricalLookaheadError as hle:
                    st.error(f"🛡️ **Lookahead Protection Blocked Request:** {str(hle)}")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    # ==========================================
    # VIEW 2: DATASET & MODEL TRAINING
    # ==========================================
    elif selected_nav == "📂 Dataset & Model Training":
        st.subheader("📂 Dataset Ingestion & Chronological ML Training")
        st.write(
            "Ingest the official **Chicago CTA Bus Daily Totals** dataset or upload your own CSV "
            "to execute leak-free chronological model training."
        )

        template_dataset_path = "data/passenger_demand_template.csv"
        
        st.markdown("#### 🌐 Authentic Open Transit Dataset Source")
        st.write(
            "Dataset: **CTA - Ridership - Bus Routes - Daily Totals by Route** from the **City of Chicago Open Data Portal** "
            "(`data.cityofchicago.org`, Public Domain / Open Access)."
        )
        
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            if st.button("📥 Fetch Chicago CTA Bus Ridership Dataset (~10,000-25,000 Real Records)", type="primary"):
                with st.spinner("Downloading authentic ridership data from data.cityofchicago.org..."):
                    try:
                        saved_path = download_cta_dataset()
                        st.success(f"✅ Successfully downloaded authentic CTA dataset to `{saved_path}`!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Download failed: {err}. You can download the file manually from https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv?accessType=DOWNLOAD and upload below.")

        st.markdown("---")
        uploaded_file = st.file_uploader("Or Upload Custom Transit Ridership Dataset (.CSV)", type=["csv"])
        
        target_csv_path = None
        if uploaded_file is not None:
            os.makedirs("data", exist_ok=True)
            save_path = os.path.join("data", "uploaded_passenger_demand.csv")
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            target_csv_path = save_path
            st.success(f"✅ Uploaded dataset saved to `{save_path}`")
        elif dataset_exists:
            target_csv_path = active_dataset_path
            st.info(f"ℹ️ Active dataset ready at: `{target_csv_path}`")
        else:
            st.info(
                f"ℹ️ No dataset file found at `data/passenger_demand.csv` yet. "
                "Click the button above to fetch the authentic Chicago CTA dataset, "
                "or upload your CSV file."
            )

        if target_csv_path and os.path.exists(target_csv_path):
            try:
                df = load_and_validate_dataset(target_csv_path)
                summary = get_dataset_summary(df)
                
                st.markdown("#### 📋 Dataset Profile & Health Verification")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Daily Records", f"{summary['total_rows']:,}")
                c2.metric("Date Span", summary["date_range"])
                c3.metric("Unique Routes", summary["unique_routes"])
                c4.metric("Mean Daily Rides", f"{summary['target_stats']['mean']:,.1f}" if summary['target_stats'] else "N/A")
                
                with st.expander("🔍 Preview Processed Dataset Records (First 10 Rows)", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                with st.expander("📊 Target Distribution & Missing Values", expanded=False):
                    st.write("**Target Statistics (Daily Passenger Boardings):**", summary["target_stats"])
                    st.write("**Missing Values per Column:**", summary["missing_values_per_col"])

                st.markdown("---")
                st.markdown("#### ⚙️ Chronological Model Training")
                st.write(
                    "Splits data chronologically so future dates are **never used to predict past dates**. "
                    "Fits preprocessor strictly on the training dates."
                )
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    test_pct = st.slider("Held-out Chronological Test Fraction (%)", min_value=10, max_value=35, value=20, step=5)
                with col_t2:
                    rand_seed = st.number_input("Random State Seed", value=42, step=1)
                
                if st.button("⚡ Execute Chronological Training Pipeline", type="primary"):
                    with st.spinner("Executing chronological train/test split, fitting preprocessor, and training regressors..."):
                        report = train_models_pipeline(
                            dataset_path=target_csv_path,
                            output_dir=DEFAULT_MODEL_DIR,
                            test_size=(test_pct / 100.0),
                            random_state=rand_seed
                        )
                        
                        st.success(f"🎉 Training completed! Selected Champion Model: **{report['best_model']}**")
                        st.write(f"• **Train Dates:** {report['train_date_range']} ({report['train_records']:,} records)")
                        st.write(f"• **Held-out Test Dates:** {report['test_date_range']} ({report['test_records']:,} records)")
                        
                        st.markdown("### 🏆 Held-Out Test Evaluation Results")
                        m = report["metrics"]
                        res_table = [
                            {
                                "Model": name,
                                "MAE (Rides)": m[name]["mae"],
                                "RMSE (Rides)": m[name]["rmse"],
                                "R² Score": m[name]["r2"]
                            }
                            for name in m
                        ]
                        st.table(pd.DataFrame(res_table))
                        st.rerun()

            except Exception as e:
                st.error(f"Dataset error: {str(e)}")

    # ==========================================
    # VIEW 3: MODEL EVALUATION & METRICS
    # ==========================================
    elif selected_nav == "📊 Evaluation & Metrics":
        st.subheader("📊 Model Evaluation & Comparative Metrics")
        st.write("Rigorous evaluation on held-out chronological test dates. No metrics are invented or hard-coded.")

        report_file = os.path.join(DEFAULT_MODEL_DIR, "metrics_report.json")
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                rep = json.load(f)
                
            best_name = rep.get("best_model", "Unknown")
            best_metrics = rep.get("metrics", {}).get(best_name, {})
            
            st.markdown("#### 🏆 Champion Model Summary")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Champion Model", best_name)
            col_m2.metric("Test MAE", f"{best_metrics.get('mae', 'N/A'):,} rides")
            col_m3.metric("Test RMSE", f"{best_metrics.get('rmse', 'N/A'):,} rides")
            col_m4.metric("Test R² Score", best_metrics.get("r2", "N/A"))
            
            st.markdown("#### 📋 Full Model Comparison Table")
            comp_rows = []
            chart_data = {"Model": [], "MAE": [], "RMSE": [], "R2": []}
            for m_name, vals in rep.get("metrics", {}).items():
                comp_rows.append({
                    "Model": m_name,
                    "Mean Absolute Error (MAE)": vals.get("mae"),
                    "Root Mean Squared Error (RMSE)": vals.get("rmse"),
                    "R² Score (Variance Explained)": vals.get("r2")
                })
                chart_data["Model"].append(m_name)
                chart_data["MAE"].append(vals.get("mae", 0.0))
                chart_data["RMSE"].append(vals.get("rmse", 0.0))
                chart_data["R2"].append(vals.get("r2", 0.0))
                
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
            
            st.markdown("#### 📉 Error Comparison (MAE & RMSE in Daily Boardings)")
            df_chart = pd.DataFrame(chart_data).set_index("Model")[["MAE", "RMSE"]]
            st.bar_chart(df_chart)
            
            st.markdown("""
            ---
            ### 📐 Metric Formulations & Explanations
            - **MAE (Mean Absolute Error):**
              $$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^n |y_i - \\hat{y}_i|$$
              *Directly measures the average daily passenger boarding error per route.*
            - **RMSE (Root Mean Squared Error):**
              $$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (y_i - \\hat{y}_i)^2}$$
              *Penalizes large errors heavily, essential for detecting significant unpredicted corridor spikes.*
            - **R² Score (Coefficient of Determination):**
              $$R^2 = 1 - \\frac{\\sum_{i=1}^n (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^n (y_i - \\bar{y})^2}$$
              *Represents the fraction of daily ridership variance explained by the model.*
            """)
        else:
            st.info(
                "ℹ️ **No trained model report found yet.**\n\n"
                "Once the Chicago CTA dataset is ingested and training is executed in the '📂 Dataset & Model Training' tab, "
                "the real MAE, RMSE, and R² results, comparison charts, and diagnostic summaries will appear here automatically."
            )

    # ==========================================
    # VIEW 4: SCHEMA & DOCUMENTATION
    # ==========================================
    elif selected_nav == "📋 Schema & Documentation":
        st.subheader("📋 Dataset Schema Specification & Architecture")
        
        st.markdown("""
        ### Target Column
        - **`rides` / `passenger_count`** *(Integer)*: Total daily recorded passenger boardings for that specific route on that day.

        ### Required Input Features
        | Feature Name | Type | Description | Example Values |
        |---|---|---|---|
        | `route` / `route_id` | String | Transit line or route identifier | `3`, `4`, `20`, `X9` |
        | `date` | Date / Datetime | Service date (YYYY-MM-DD or MM/DD/YYYY) | `2026-07-31` |
        | `daytype` | Categorical | Schedule category | `W` (Weekday), `A` (Saturday), `U` (Sunday/Holiday) |
        | `lag_1d_rides` | Numeric | Prior day's ridership on the same route | `5420` |
        | `trailing_3d_rides` | Numeric | Trailing 3-day average ridership | `5390.0` |
        | `lag_7d_rides` | Numeric | Same day last week ridership on the same route | `5510` |
        | `rolling_mean_7d_rides` | Numeric | Trailing 7-day average ridership | `5380.5` |

        ### Zero Leakage Chronological Pipeline Architecture
        1. **Data Ingestion:** Loads daily records and extracts calendar features (`day_of_week`, `day`, `month`, `year`).
        2. **Shift Operations:** Calculates historical prior-day and trailing-7-day lag features using strictly `shift(1)` operations.
        3. **Chronological Splitting:** Splits by service date (earlier 80% train, latest 20% test). Future dates are never used to predict past dates.
        4. **Leak-Free Preprocessing:** Imputation, One-Hot Encoding, and Standard Scaling fit strictly on training dates.
        5. **Model Layer:** Baseline Linear Regression vs. Random Forest Regressor.
        6. **Held-Out Evaluation:** Measures real MAE, RMSE, and R² on unseen future test dates.
        """)

        if os.path.exists("data/passenger_demand_template.csv"):
            with open("data/passenger_demand_template.csv", "r") as f:
                template_csv = f.read()
            st.download_button(
                label="📥 Download CTA Daily Dataset Schema Template (CSV)",
                data=template_csv,
                file_name="cta_passenger_demand_template.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
