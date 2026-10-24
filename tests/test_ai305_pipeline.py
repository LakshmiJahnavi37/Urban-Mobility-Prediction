"""
Unit and integration tests for AI305 Public Transport Daily Passenger Demand Prediction.
Configured for daily route-level transit forecasting without hourly assumptions.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from ai305_passenger_demand.data_loader import (
    load_and_validate_dataset,
    extract_daily_calendar_features,
    compute_daily_lag_features,
    DatasetValidationError,
    TARGET_COLUMN
)
from ai305_passenger_demand.preprocessor import DemandFeaturePreprocessor
from ai305_passenger_demand.metrics import calculate_regression_metrics
from ai305_passenger_demand.predict import (
    PassengerDemandPredictor,
    ModelNotTrainedError,
    HistoricalLookaheadError
)
from ai305_passenger_demand.train import train_models_pipeline


class TestAI305DailyDemandPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up temporary directories and small test datasets matching CTA daily schema."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Authentic daily route ridership sample (24 chronological days across 2 routes)
        dates_r3 = pd.date_range("2026-06-01", periods=15, freq="D")
        dates_r4 = pd.date_range("2026-06-01", periods=15, freq="D")
        
        records = []
        for d in dates_r3:
            dt = "A" if d.weekday() == 5 else ("U" if d.weekday() == 6 else "W")
            records.append({"route": "3", "date": d.strftime("%Y-%m-%d"), "daytype": dt, "rides": 5000 + int(d.day * 50)})
            
        for d in dates_r4:
            dt = "A" if d.weekday() == 5 else ("U" if d.weekday() == 6 else "W")
            records.append({"route": "4", "date": d.strftime("%Y-%m-%d"), "daytype": dt, "rides": 7000 + int(d.day * 80)})
            
        self.sample_df = pd.DataFrame(records)
        self.sample_csv_path = os.path.join(self.temp_dir, "cta_daily_sample.csv")
        self.sample_df.to_csv(self.sample_csv_path, index=False)
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_metrics_calculation(self):
        """Test that MAE, RMSE, and R2 compute exact mathematical values."""
        y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        y_pred = np.array([1200.0, 1800.0, 3200.0, 3800.0])  # errors: +200, -200, +200, -200
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics["mae"], 200.0, places=2)
        self.assertAlmostEqual(metrics["rmse"], 200.0, places=2)
        self.assertAlmostEqual(metrics["r2"], 0.968, places=3)

    def test_daily_calendar_feature_extraction(self):
        """Test extraction of authentic daily calendar indicators without hourly assumptions."""
        df_loaded = load_and_validate_dataset(self.sample_csv_path)
        
        self.assertIn("day_of_week", df_loaded.columns)
        self.assertIn("day", df_loaded.columns)
        self.assertIn("month", df_loaded.columns)
        self.assertIn("year", df_loaded.columns)
        self.assertIn("is_weekend", df_loaded.columns)
        self.assertIn("is_holiday_or_sunday", df_loaded.columns)
        
        # Verify no hourly assumptions exist
        self.assertNotIn("hour", df_loaded.columns)
        self.assertNotIn("is_rush_hour", df_loaded.columns)
        
        # 2026-06-01 is Monday (day_of_week = 0)
        row_mon = df_loaded[df_loaded["date"] == "2026-06-01"].iloc[0]
        self.assertEqual(row_mon["day_of_week"], 0)
        self.assertEqual(row_mon["is_weekend"], 0)

    def test_daily_lag_features_zero_leakage(self):
        """Test that lag features use shift operations and do not leak current target."""
        df_loaded = load_and_validate_dataset(self.sample_csv_path)
        
        self.assertIn("lag_1d_rides", df_loaded.columns)
        self.assertIn("trailing_3d_rides", df_loaded.columns)
        self.assertIn("lag_7d_rides", df_loaded.columns)
        self.assertIn("rolling_mean_7d_rides", df_loaded.columns)
        
        # For route 3 on second day (2026-06-02), lag_1d_rides must equal first day's rides (2026-06-01)
        r3_data = df_loaded[df_loaded["route_id"] == "3"].sort_values(by="date")
        day1_rides = r3_data.iloc[0]["passenger_count"]
        day2_lag = r3_data.iloc[1]["lag_1d_rides"]
        self.assertEqual(day1_rides, day2_lag)

    def test_data_loader_validation_missing_target(self):
        """Test that missing rides / passenger_count column raises DatasetValidationError."""
        bad_df = self.sample_df.drop(columns=["rides"])
        bad_csv = os.path.join(self.temp_dir, "bad.csv")
        bad_df.to_csv(bad_csv, index=False)
        
        with self.assertRaises(DatasetValidationError):
            load_and_validate_dataset(bad_csv)

    def test_preprocessor_leak_free_transformation(self):
        """Test fitting on X_train and transforming without leaking test data."""
        df_loaded = load_and_validate_dataset(self.sample_csv_path)
        preprocessor = DemandFeaturePreprocessor()
        
        train_df = df_loaded.iloc[:20]
        test_df = df_loaded.iloc[20:]
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        self.assertEqual(X_train_trans.shape[1], X_test_trans.shape[1])
        self.assertGreater(X_train_trans.shape[0], 0)

    def test_predictor_uninitialized_raises_error(self):
        """Test that predictor raises ModelNotTrainedError when artifacts do not exist."""
        empty_dir = os.path.join(self.temp_dir, "empty_models")
        os.makedirs(empty_dir, exist_ok=True)
        predictor = PassengerDemandPredictor(model_dir=empty_dir)
        self.assertFalse(predictor.is_ready())
        
        with self.assertRaises(ModelNotTrainedError):
            predictor.predict_single({"route_id": "3", "date": "2026-10-25"})

    def test_end_to_end_chronological_training_and_prediction(self):
        """Test chronological split, model training, artifact saving, and live inference."""
        models_out = os.path.join(self.temp_dir, "saved_models")
        
        results = train_models_pipeline(
            dataset_path=self.sample_csv_path,
            output_dir=models_out,
            test_size=0.25,
            random_state=42
        )
        
        self.assertIn("metrics", results)
        self.assertIn("Baseline (Linear Regression)", results["metrics"])
        self.assertIn("Random Forest Regressor", results["metrics"])
        self.assertIn("train_date_range", results)
        self.assertIn("test_date_range", results)
        
        # Verify saved artifacts
        self.assertTrue(os.path.exists(os.path.join(models_out, "best_model.joblib")))
        self.assertTrue(os.path.exists(os.path.join(models_out, "preprocessor.joblib")))
        self.assertTrue(os.path.exists(os.path.join(models_out, "metrics_report.json")))
        self.assertTrue(os.path.exists(os.path.join(models_out, "route_history.json")))
        
        # Test Predictor loads artifacts and produces non-negative daily forecast
        predictor = PassengerDemandPredictor(model_dir=models_out)
        self.assertTrue(predictor.is_ready())
        
        single_input = {
            "route_id": "3",
            "date": "2026-06-20",
            "daytype": "W"
        }
        
        pred_res = predictor.predict_single(single_input)
        self.assertIn("predicted_passenger_count", pred_res)
        self.assertGreater(pred_res["predicted_passenger_count"], 0)
        self.assertIn("demand_tier", pred_res)
        self.assertIn("estimated_buses_required", pred_res)
        
        # Test prediction when lag features are excluded at prediction time
        cold_start_input = {
            "route_id": "3",
            "date": "2026-06-25",
            "daytype": "W",
            "exclude_lag_features": True
        }
        pred_cold = predictor.predict_single(cold_start_input)
        self.assertIn("predicted_passenger_count", pred_cold)
        self.assertGreater(pred_cold["predicted_passenger_count"], 0)

    def test_historical_prediction_cannot_use_future_observations_june_15(self):
        """
        Verify that a prediction for June 15, 2026 cannot use observations from July 31, 2026.
        Tests both lag reconstruction strictly before June 15 and rejection of future lookahead.
        """
        # Create a mock dataset spanning June 1 through July 31, 2026
        dates = pd.date_range("2026-06-01", "2026-07-31", freq="D")
        records = []
        for d in dates:
            dt = "A" if d.weekday() == 5 else ("U" if d.weekday() == 6 else "W")
            # Distinct rides: 5200 on June 14, 9999 on July 31
            if d.strftime("%Y-%m-%d") == "2026-06-14":
                rides = 5200
            elif d.strftime("%Y-%m-%d") == "2026-07-31":
                rides = 9999
            else:
                rides = 5000 + int(d.day * 10)
            records.append({"route": "3", "date": d.strftime("%Y-%m-%d"), "daytype": dt, "rides": rides})
            
        history_df = pd.DataFrame(records)
        history_csv = os.path.join(self.temp_dir, "june_july_history.csv")
        history_df.to_csv(history_csv, index=False)
        
        # Train pipeline on this history
        models_out = os.path.join(self.temp_dir, "models_lookahead_test")
        train_models_pipeline(dataset_path=history_csv, output_dir=models_out, test_size=0.2, random_state=42)
        
        # Load predictor with route_history where last_observed_date is 2026-07-31 (latest_rides = 9999)
        predictor = PassengerDemandPredictor(model_dir=models_out, dataset_path=history_csv)
        self.assertEqual(predictor.route_history["3"]["last_observed_date"], "2026-07-31")
        self.assertEqual(predictor.route_history["3"]["latest_rides"], 9999.0)
        
        # 1. Test lag reconstruction for June 15, 2026:
        # Must strictly use observations BEFORE June 15 (i.e. June 14: 5200 rides), NEVER July 31 (9999 rides)
        reconstructed = predictor.reconstruct_historical_lags("3", "2026-06-15")
        self.assertIsNotNone(reconstructed)
        self.assertEqual(reconstructed["lag_1d_rides"], 5200.0)
        self.assertNotEqual(reconstructed["lag_1d_rides"], 9999.0)
        self.assertEqual(reconstructed["last_available_date"], "2026-06-14")
        
        # 2. Predict for June 15, 2026:
        # Prediction must succeed using reconstructed lag without leaking July 31
        pred_res = predictor.predict_single({"route_id": "3", "date": "2026-06-15", "daytype": "W"})
        self.assertIn("predicted_passenger_count", pred_res)
        self.assertGreater(pred_res["predicted_passenger_count"], 0)
        
        # 3. Test lookahead guard when historical data prior to requested date is unavailable:
        # Request for historical date without prior data and without custom lags must raise HistoricalLookaheadError
        predictor_isolated = PassengerDemandPredictor(
            model_dir=models_out,
            dataset_path=os.path.join(self.temp_dir, "nonexistent.csv")
        )
        with self.assertRaises(HistoricalLookaheadError):
            predictor_isolated.predict_single({"route_id": "3", "date": "2026-06-15", "daytype": "W"})
            
        # 4. Request for future date (after July 31, e.g. 2026-08-15) preserves valid workflow using latest history
        pred_future = predictor.predict_single({"route_id": "3", "date": "2026-08-15", "daytype": "W"})
        self.assertIn("predicted_passenger_count", pred_future)
        self.assertGreater(pred_future["predicted_passenger_count"], 0)


if __name__ == "__main__":
    unittest.main()
