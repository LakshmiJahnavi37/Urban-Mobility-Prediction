import os, sys
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(__file__))
from features import add_features, get_X_y, DATE_COL

df = add_features(pd.read_csv("data/processed/traffic_clean.csv"))
X, y = get_X_y(df)

# time-based split: first 80% of the timeline = train, last 20% = test (no shuffling)
cutoff = df[DATE_COL].quantile(0.8)
train_idx = df[DATE_COL] < cutoff
X_train, X_test = X[train_idx], X[~train_idx]
y_train, y_test = y[train_idx], y[~train_idx]
print(f"Cutoff: {cutoff} | train={len(X_train)} test={len(X_test)}")

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, random_state=42),
}

results, preds = [], {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    preds[name] = p
    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, p), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, p))), 2),
        "R2": round(r2_score(y_test, p), 4),
    })

res = pd.DataFrame(results)
print(res.to_string(index=False))
os.makedirs("outputs/plots", exist_ok=True)
res.to_csv("outputs/results.csv", index=False)

best = res.sort_values("R2", ascending=False).iloc[0]["Model"]
print("Best model:", best)

# per-junction metrics for best model
test_df = df[~train_idx].copy()
test_df["pred"] = preds[best]
rows = []
for j in range(1, 5):
    col = f"Junction_{j}"
    if col in test_df and test_df[col].sum() > 0:
        s = test_df[test_df[col] == 1]
        rows.append({"Junction": j,
                     "MAE": round(mean_absolute_error(s["Vehicles"], s["pred"]), 2),
                     "R2": round(r2_score(s["Vehicles"], s["pred"]), 4)})
pj = pd.DataFrame(rows)
print(pj.to_string(index=False))
pj.to_csv("outputs/results_per_junction.csv", index=False)

os.makedirs("models", exist_ok=True)
joblib.dump({"model": models[best], "features": list(X.columns), "name": best},
            "models/traffic_model.pkl")

# plot 1: actual vs predicted (first week of test set, junction 1)
s = test_df[test_df["Junction_1"] == 1].head(168)
plt.figure(figsize=(11, 4))
plt.plot(s["Vehicles"].values, label="Actual")
plt.plot(s["pred"].values, label="Predicted")
plt.title(f"Actual vs Predicted - Junction 1, first test week ({best})")
plt.xlabel("Hour"); plt.ylabel("Vehicles"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/plots/actual_vs_predicted.png", dpi=150); plt.close()

# plot 2: model comparison
plt.figure(figsize=(6, 4))
plt.bar(res["Model"], res["R2"], color=["#888", "#4c9", "#49c"])
plt.title("Model comparison (R2)"); plt.ylim(0, 1); plt.tight_layout()
plt.savefig("outputs/plots/model_comparison.png", dpi=150); plt.close()

# plot 3: feature importance
mdl = models[best]
if hasattr(mdl, "feature_importances_"):
    imp = pd.Series(mdl.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 6)); imp.plot.barh(); plt.title(f"Feature importance ({best})")
    plt.tight_layout(); plt.savefig("outputs/plots/feature_importance.png", dpi=150); plt.close()
print("Saved model + plots + results")
