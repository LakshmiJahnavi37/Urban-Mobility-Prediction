import os
import joblib, pandas as pd

_path = os.path.join(os.path.dirname(__file__), "..", "models", "traffic_model.pkl")
_bundle = None

def _load():
    global _bundle
    if _bundle is None:
        if not os.path.exists(_path):
            raise FileNotFoundError("Model not found. Run: python src/train.py")
        _bundle = joblib.load(_path)
    return _bundle

def predict_traffic(junction, hour, day_of_week, month, day, lag_1, lag_24, lag_168):
    """Predict number of vehicles for one junction (1-4) at a given hour."""
    b = _load()
    row = {f: 0 for f in b["features"]}
    row.update({
        "hour": hour, "day_of_week": day_of_week, "day": day, "month": month,
        "is_weekend": int(day_of_week >= 5),
        "is_rush_hour": int(hour in [8, 9, 17, 18, 19]),
        "lag_1": lag_1, "lag_24": lag_24, "lag_168": lag_168,
        f"Junction_{junction}": 1,
    })
    return float(b["model"].predict(pd.DataFrame([row])[b["features"]])[0])
