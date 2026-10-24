"""
Evaluation metrics calculation module for regression models.
Calculates MAE, RMSE, and R2 on held-out test predictions.
No metrics are invented or hard-coded.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes real regression evaluation metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R2 (Coefficient of Determination)
    
    Args:
        y_true: Ground truth target values
        y_pred: Model predictions
        
    Returns:
        Dictionary containing real calculated metrics rounded to 4 decimal places.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    # root_mean_squared_error is standard; using sqrt(mean_squared_error) for cross-version compatibility
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }


def format_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Formats a comparative string table of model evaluation metrics.
    
    Args:
        metrics_dict: Dictionary mapping model names to metric dictionaries.
        
    Returns:
        Formatted ASCII comparison table.
    """
    header = f"{'Model':<25} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}"
    divider = "-" * len(header)
    rows = [divider, header, divider]
    
    for model_name, m in metrics_dict.items():
        row = f"{model_name:<25} | {m.get('mae', 0.0):<10.4f} | {m.get('rmse', 0.0):<10.4f} | {m.get('r2', 0.0):<10.4f}"
        rows.append(row)
        
    rows.append(divider)
    return "\n".join(rows)
