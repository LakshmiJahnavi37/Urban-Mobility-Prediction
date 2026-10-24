"""
Inference Module for AI305 Public Transport Daily Passenger Demand Prediction.
Loads saved preprocessor, model, and route historical baselines to generate forecasts
without lookahead target leakage.
"""

from typing import Dict, Any, Union, List, Tuple, Optional
import os
import json
import joblib
import pandas as pd
import numpy as np

from ai305_passenger_demand.preprocessor import DemandFeaturePreprocessor


DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ModelNotTrainedError(Exception):
    """Raised when inference is attempted before models have been trained and saved."""
    pass


class HistoricalLookaheadError(ValueError):
    """Raised when an inference request for a historical date would leak future observations."""
    pass


class PassengerDemandPredictor:
    """
    Loads persisted pipeline artifacts and produces daily route-level demand forecasts
    with strict temporal guards preventing future lookahead leakage.
    """
    
    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        dataset_path: Optional[str] = None,
        historical_df: Optional[pd.DataFrame] = None
    ):
        self.model_dir = model_dir
        self.dataset_path = dataset_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "passenger_demand.csv"
        )
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        self.best_model_path = os.path.join(model_dir, "best_model.joblib")
        self.route_history_path = os.path.join(model_dir, "route_history.json")
        self.preprocessor: Optional[DemandFeaturePreprocessor] = None
        self.model = None
        self.route_history: Dict[str, Any] = {}
        self._historical_df: Optional[pd.DataFrame] = historical_df
        self._load_artifacts()
        
    def is_ready(self) -> bool:
        """Checks if model artifacts are loaded and ready for prediction."""
        return self.preprocessor is not None and self.model is not None
        
    def _load_artifacts(self) -> None:
        """Loads joblib artifacts and route history from disk if they exist."""
        if os.path.exists(self.preprocessor_path) and os.path.exists(self.best_model_path):
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
                self.model = joblib.load(self.best_model_path)
                if os.path.exists(self.route_history_path):
                    with open(self.route_history_path, "r", encoding="utf-8") as f:
                        self.route_history = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed loading existing model artifacts: {e}")
                self.preprocessor = None
                self.model = None
        else:
            self.preprocessor = None
            self.model = None

    def _get_historical_df(self) -> Optional[pd.DataFrame]:
        """Loads and caches the historical dataset for lag reconstruction if available."""
        if self._historical_df is not None:
            return self._historical_df
            
        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                df = df.rename(columns={c: c.strip().lower() for c in df.columns})
                # Normalize column names
                if "rides" in df.columns and "passenger_count" not in df.columns:
                    df = df.rename(columns={"rides": "passenger_count"})
                if "route" in df.columns and "route_id" not in df.columns:
                    df = df.rename(columns={"route": "route_id"})
                df["route_id"] = df["route_id"].astype(str).str.strip()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
                df = df.dropna(subset=["date", "passenger_count"])
                self._historical_df = df
                return self._historical_df
            except Exception:
                return None
        return None

    def reconstruct_historical_lags(
        self,
        route_id: str,
        target_date: Union[str, pd.Timestamp]
    ) -> Optional[Dict[str, float]]:
        """
        Reconstructs lag features strictly from historical records observed BEFORE target_date.
        Guarantees zero future target leakage for historical service predictions.
        
        Returns:
            Dictionary with lag_1d_rides, trailing_3d_rides, lag_7d_rides, rolling_mean_7d_rides,
            and last_available_date; or None if no records exist before target_date.
        """
        df = self._get_historical_df()
        if df is None or df.empty:
            return None
            
        ts_target = pd.to_datetime(target_date)
        # Filter for the same route and STRICTLY prior dates
        prior_records = df[(df["route_id"] == str(route_id).strip()) & (df["date"] < ts_target)]
        
        if prior_records.empty:
            return None
            
        sorted_records = prior_records.sort_values(by="date").reset_index(drop=True)
        
        # 1-day prior observation (latest available strictly before target_date)
        latest_prior = sorted_records.iloc[-1]
        lag_1d = float(latest_prior["passenger_count"])
        
        # Trailing 3-day average of prior observations
        trailing_3d = float(sorted_records["passenger_count"].tail(3).mean())
        
        # Trailing 7-day average of prior observations
        rolling_7d = float(sorted_records["passenger_count"].tail(7).mean())
        
        # 7-day prior observation (7th prior record or fallback to trailing 7d)
        if len(sorted_records) >= 7:
            lag_7d = float(sorted_records.iloc[-7]["passenger_count"])
        else:
            lag_7d = rolling_7d
            
        return {
            "lag_1d_rides": lag_1d,
            "trailing_3d_rides": trailing_3d,
            "lag_7d_rides": lag_7d,
            "rolling_mean_7d_rides": rolling_7d,
            "last_available_date": str(latest_prior["date"].date())
        }
        
    def _categorize_daily_demand(self, count: float) -> Tuple[str, str, str, int]:
        """
        Categorizes daily route passenger ridership into operational corridor tiers.
        Assumes standard urban transit bus with ~70 passenger capacity making ~8 trips/day (~560 riders/bus).
        
        Returns:
            Tuple of (corridor_tier, status_color, operational_recommendation, estimated_buses_required)
        """
        bus_daily_capacity = 560
        buses_needed = max(1, int(np.ceil(count / bus_daily_capacity)))
        
        if count < 2500:
            return (
                "Low-Density Community Route",
                "green",
                f"Standard community feeder schedule. Deploy {buses_needed} standard transit buses.",
                buses_needed
            )
        elif count < 6500:
            return (
                "Moderate Regular Corridor",
                "blue",
                f"Standard city corridor schedule. Deploy {buses_needed} buses with regular 12-15 min headways.",
                buses_needed
            )
        elif count < 12000:
            return (
                "High-Density Arterial Corridor",
                "orange",
                f"High passenger volume corridor. Deploy {buses_needed} buses with frequent 6-8 min peak headways.",
                buses_needed
            )
        else:
            return (
                "Heavy Transit Trunk Line",
                "red",
                f"Heavy demand alert! Deploy {buses_needed} buses including articulated/express units and signal priority.",
                buses_needed
            )
            
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes a daily demand forecast for a route on a specified service date.
        
        Guarantees ZERO lookahead leakage:
        - For future service dates (after historical dataset max date): uses route_history.json baselines.
        - For historical service dates (on or before route_history last_observed_date): NEVER uses
          route_history.json's latest values. Instead, reconstructs lags strictly from observations
          dated before the service date, uses custom supplied lags, or imputes cold-start if excluded.
          If no prior observations exist and lags are not supplied, raises HistoricalLookaheadError.
        
        Args:
            input_data: Dictionary containing route_id, date, daytype, and optional lag ridership.
            
        Returns:
            Dictionary with forecasted daily passenger count, corridor tier, and bus fleet recommendation.
        """
        if not self.is_ready():
            raise ModelNotTrainedError(
                "AI305 model artifacts were not found in 'models/' directory. "
                "The model has not been trained yet because an authentic dataset was not provided. "
                "Please provide a dataset and run the training pipeline first."
            )
            
        route_id = str(input_data.get("route_id", "General")).strip()
        hist = self.route_history.get(route_id, {})
        
        target_date_str = str(input_data.get("date", "2026-10-25"))
        try:
            target_date = pd.to_datetime(target_date_str)
        except Exception:
            target_date = pd.Timestamp.now()
            
        last_obs_date_str = hist.get("last_observed_date")
        last_obs_date = pd.to_datetime(last_obs_date_str) if last_obs_date_str else None
        
        is_historical_request = (
            last_obs_date is not None and target_date.date() <= last_obs_date.date()
        )
        
        # Safe handling of lag features without lookahead leakage
        exclude_lags = input_data.get("exclude_lag_features", False)
        
        if exclude_lags:
            # Features not available at prediction time: exclude and let preprocessor impute baseline
            lag_1d = np.nan
            trailing_3d = np.nan
            lag_7d = np.nan
            rolling_7d = np.nan
        else:
            # Check if user explicitly provided custom lag values
            user_lag1 = input_data.get("lag_1d_rides")
            user_lag3 = input_data.get("trailing_3d_rides")
            user_lag7 = input_data.get("lag_7d_rides")
            user_roll7 = input_data.get("rolling_mean_7d_rides")
            
            has_custom_lags = (
                user_lag1 is not None and isinstance(user_lag1, (int, float)) and user_lag1 > 0
            )
            
            if has_custom_lags:
                lag_1d = float(user_lag1)
                trailing_3d = float(user_lag3) if (user_lag3 is not None and user_lag3 > 0) else lag_1d
                lag_7d = float(user_lag7) if (user_lag7 is not None and user_lag7 > 0) else trailing_3d
                rolling_7d = float(user_roll7) if (user_roll7 is not None and user_roll7 > 0) else trailing_3d
            elif is_historical_request:
                # STRICT LOOKAHEAD SAFETY: For dates on or before last_observed_date (e.g. 2026-06-15 vs 2026-07-31),
                # NEVER use route_history.json's latest_rides from the future!
                reconstructed = self.reconstruct_historical_lags(route_id, target_date)
                if reconstructed is not None:
                    lag_1d = reconstructed["lag_1d_rides"]
                    trailing_3d = reconstructed["trailing_3d_rides"]
                    lag_7d = reconstructed["lag_7d_rides"]
                    rolling_7d = reconstructed["rolling_mean_7d_rides"]
                else:
                    raise HistoricalLookaheadError(
                        f"Cannot predict for historical service date '{target_date.date()}' on route '{route_id}': "
                        f"No historical records exist strictly prior to this date. "
                        f"To prevent lookahead target leakage, future observations from '{last_obs_date.date()}' "
                        f"cannot be used. Please enter custom prior-day boardings or enable 'exclude_lag_features=True'."
                    )
            else:
                # Safe filling of historical lag strictly from observations available before target date
                # For future dates (target_date > last_obs_date), last_obs_date is strictly in the past
                lag_1d = hist.get("latest_rides", hist.get("median_rides", 4500.0))
                trailing_3d = hist.get("latest_3d_avg", lag_1d)
                lag_7d = hist.get("latest_7d_avg", hist.get("median_rides", 4500.0))
                rolling_7d = hist.get("latest_7d_avg", trailing_3d)
                
        record = {
            "route_id": route_id,
            "date": target_date_str,
            "daytype": input_data.get("daytype", "W"),
            "lag_1d_rides": lag_1d,
            "trailing_3d_rides": trailing_3d,
            "lag_7d_rides": lag_7d,
            "rolling_mean_7d_rides": rolling_7d
        }
        
        df_input = pd.DataFrame([record])
        X_trans = self.preprocessor.transform(df_input)
        
        raw_pred = self.model.predict(X_trans)[0]
        pred_count = max(0.0, float(raw_pred))
        
        tier, color, rec, buses = self._categorize_daily_demand(pred_count)
        
        return {
            "predicted_passenger_count": round(pred_count, 1),
            "predicted_passenger_count_int": int(round(pred_count)),
            "demand_tier": tier,
            "badge_color": color,
            "operational_recommendation": rec,
            "estimated_buses_required": buses,
            "route_id": route_id,
            "date": str(record["date"]),
            "daytype": record["daytype"]
        }
