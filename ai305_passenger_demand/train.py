"""
Model Training & Chronological Evaluation Pipeline for AI305 Public Transport Demand.
Trains a Baseline Linear Regression model and a Random Forest Regressor on daily ridership.
Applies chronological train/test splitting to prevent lookahead target leakage,
computes held-out MAE, RMSE, and R2, and saves model artifacts for production inference.
"""

from typing import Dict, Any, Tuple, Optional
import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from ai305_passenger_demand.data_loader import (
    load_and_validate_dataset,
    TARGET_COLUMN,
    DatasetValidationError
)
from ai305_passenger_demand.preprocessor import DemandFeaturePreprocessor
from ai305_passenger_demand.metrics import (
    calculate_regression_metrics,
    format_metrics_comparison
)


DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_models_pipeline(
    dataset_path: str,
    output_dir: str = DEFAULT_MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes leak-free chronological training and evaluation:
    1. Loads dataset and derives daily calendar + shift lag features.
    2. Sorts data chronologically and splits by time (earlier 80% train, latest 20% test).
    3. Fits preprocessor strictly on training split.
    4. Trains Baseline Linear Regression and Random Forest Regressor.
    5. Evaluates real MAE, RMSE, and R2 on unseen future test dates.
    6. Saves artifacts and route historical stats for inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[AI305 Pipeline] Loading daily transit dataset from: {dataset_path}")
    df = load_and_validate_dataset(dataset_path)
    
    if len(df) < 20:
        raise DatasetValidationError(
            f"Dataset has only {len(df)} records. Need at least 20 records for chronological splitting."
        )
        
    # Chronological Ordering
    df = df.sort_values(by=["date", "route_id"]).reset_index(drop=True)
    
    # Chronological Split (prevents predicting past using future data)
    unique_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(unique_dates)
    split_idx = int(n_dates * (1.0 - test_size))
    split_date = unique_dates.iloc[split_idx]
    
    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"[AI305 Pipeline] Total records: {len(df)} across {n_dates} unique service days.")
    print(f"  • Training Split: {len(train_df)} records ({str(train_df['date'].min().date())} to {str(train_df['date'].max().date())})")
    print(f"  • Test Split:     {len(test_df)} records ({str(test_df['date'].min().date())} to {str(test_df['date'].max().date())})")
    
    y_train = train_df[TARGET_COLUMN].values
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    
    y_test = test_df[TARGET_COLUMN].values
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    
    # Fit preprocessor strictly on X_train (Zero Data Leakage)
    print("\n[AI305 Pipeline] Fitting feature preprocessor strictly on training dates...")
    preprocessor = DemandFeaturePreprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Train Model 1: Baseline Linear Regression
    print("[AI305 Pipeline] Training Model 1: Baseline Linear Regression...")
    baseline_model = LinearRegression()
    baseline_model.fit(X_train_transformed, y_train)
    y_pred_baseline = np.clip(baseline_model.predict(X_test_transformed), a_min=0, a_max=None)
    metrics_baseline = calculate_regression_metrics(y_test, y_pred_baseline)
    
    # Train Model 2: Random Forest Regressor
    print("[AI305 Pipeline] Training Model 2: Random Forest Regressor (n_estimators=100)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=1
    )
    rf_model.fit(X_train_transformed, y_train)
    y_pred_rf = np.clip(rf_model.predict(X_test_transformed), a_min=0, a_max=None)
    metrics_rf = calculate_regression_metrics(y_test, y_pred_rf)
    
    # Model Comparison & Selection
    metrics_comparison = {
        "Baseline (Linear Regression)": metrics_baseline,
        "Random Forest Regressor": metrics_rf
    }
    
    print("\n" + "=" * 62)
    print("CHRONOLOGICAL HELD-OUT TEST EVALUATION RESULTS:")
    print(format_metrics_comparison(metrics_comparison))
    print("=" * 62 + "\n")
    
    if metrics_rf["r2"] >= metrics_baseline["r2"]:
        best_model_name = "Random Forest Regressor"
        best_model = rf_model
        best_metrics = metrics_rf
    else:
        best_model_name = "Baseline (Linear Regression)"
        best_model = baseline_model
        best_metrics = metrics_baseline
        
    print(f"[AI305 Pipeline] Selected Champion: {best_model_name} (R2={best_metrics['r2']}, RMSE={best_metrics['rmse']})")
    
    # Extract latest historical route observations for live inference
    route_history = {}
    for r_id, group in df.groupby("route_id"):
        sorted_g = group.sort_values(by="date")
        latest_row = sorted_g.iloc[-1]
        route_history[str(r_id)] = {
            "median_rides": float(group[TARGET_COLUMN].median()),
            "last_observed_date": str(latest_row["date"].date()),
            "latest_rides": float(latest_row[TARGET_COLUMN]),
            "latest_3d_avg": float(sorted_g[TARGET_COLUMN].tail(3).mean()),
            "latest_7d_avg": float(sorted_g[TARGET_COLUMN].tail(7).mean())
        }
        
    # Save artifacts to disk
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    rf_model_path = os.path.join(output_dir, "rf_model.joblib")
    baseline_model_path = os.path.join(output_dir, "baseline_model.joblib")
    report_path = os.path.join(output_dir, "metrics_report.json")
    history_path = os.path.join(output_dir, "route_history.json")
    
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(best_model, best_model_path)
    joblib.dump(rf_model, rf_model_path)
    joblib.dump(baseline_model, baseline_model_path)
    
    report_data = {
        "dataset_path": os.path.abspath(dataset_path),
        "total_records": len(df),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_date_range": f"{str(train_df['date'].min().date())} to {str(train_df['date'].max().date())}",
        "test_date_range": f"{str(test_df['date'].min().date())} to {str(test_df['date'].max().date())}",
        "best_model": best_model_name,
        "metrics": metrics_comparison,
        "feature_names": preprocessor.feature_names_out_
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(route_history, f, indent=2)
        
    print(f"[AI305 Pipeline] Preprocessor saved to: {preprocessor_path}")
    print(f"[AI305 Pipeline] Best model saved to: {best_model_path}")
    print(f"[AI305 Pipeline] Metrics report saved to: {report_path}")
    print(f"[AI305 Pipeline] Route history saved to: {history_path}")
    
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI305 Public Transport Daily Passenger Demand Models.")
    parser.add_argument("--dataset-path", type=str, default="data/passenger_demand.csv", help="Path to CSV dataset")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to save artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out chronological test fraction")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"\n[ERROR] Dataset not found at: '{args.dataset_path}'")
        print("In accordance with project integrity requirements, training requires an authentic dataset.")
        print(f"Please place your dataset file at '{args.dataset_path}' or fetch via `python -m ai305_passenger_demand.fetch_dataset`.\n")
        exit(1)
        
    train_models_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        test_size=args.test_size
    )
