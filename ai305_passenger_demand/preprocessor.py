"""
Data Preprocessing Pipeline for Daily Public Transport Passenger Demand Prediction.
Guarantees zero data leakage by strictly fitting transformers on training data.
Transforms route identifiers, daily schedule daytypes, and calendar/lag indicators.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai305_passenger_demand.data_loader import extract_daily_calendar_features


# Feature definitions for daily CTA ridership forecasting
CATEGORICAL_FEATURES = ["route_id", "daytype"]

NUMERICAL_FEATURES = [
    "day_of_week",
    "day",
    "month",
    "year",
    "is_weekend",
    "is_holiday_or_sunday",
    "lag_1d_rides",
    "trailing_3d_rides",
    "lag_7d_rides",
    "rolling_mean_7d_rides"
]


class DemandFeaturePreprocessor:
    """
    Leak-free feature preprocessing pipeline for daily route-level transit ridership.
    """
    
    def __init__(self):
        self.categorical_features = CATEGORICAL_FEATURES
        self.numerical_features = NUMERICAL_FEATURES
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None
        self._is_fitted = False
        
    def _build_column_transformer(self) -> ColumnTransformer:
        """
        Builds the ColumnTransformer containing numerical imputation/scaling
        and categorical imputation/one-hot encoding.
        """
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_features),
                ("cat", cat_pipeline, self.categorical_features)
            ],
            remainder="drop"
        )
        
    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all expected daily features exist with correct numeric/string types.
        """
        df_processed = extract_daily_calendar_features(df)
        
        # Ensure numerical columns exist
        for col in self.numerical_features:
            if col not in df_processed.columns:
                df_processed[col] = np.nan
            else:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
                
        # Ensure categorical columns exist
        for col in self.categorical_features:
            if col not in df_processed.columns:
                df_processed[col] = "Unknown"
            else:
                df_processed[col] = df_processed[col].astype(str).str.strip().fillna("Unknown")
                
        return df_processed
        
    def fit(self, X: pd.DataFrame, y=None) -> "DemandFeaturePreprocessor":
        """
        Fits the preprocessor strictly on training data X_train to prevent leakage.
        """
        prepared_df = self._prepare_df(X)
        self.column_transformer = self._build_column_transformer()
        self.column_transformer.fit(prepared_df)
        
        try:
            self.feature_names_out_ = list(self.column_transformer.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = []
            
        self._is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms new data (e.g. held-out test split or production inference).
        """
        if not self._is_fitted or self.column_transformer is None:
            raise RuntimeError("DemandFeaturePreprocessor must be fitted before calling transform().")
            
        prepared_df = self._prepare_df(X)
        return self.column_transformer.transform(prepared_df)
        
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Fits on X and returns transformed array.
        """
        return self.fit(X, y).transform(X)
        
    def transform_single_record(self, record: Dict[str, Any]) -> np.ndarray:
        """
        Convenience method to preprocess a single daily route input record for inference.
        """
        df = pd.DataFrame([record])
        return self.transform(df)
