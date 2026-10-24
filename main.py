
import os
import glob
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 1. Load dataset

dataset_path = (
    r"E:\DriveReady\Main_Project\Urban-Mobility-Prediction"
    r"\dataset\TrafficTwoMonth.csv"
)

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

df.columns = df.columns.str.strip()

df.columns = [
    "Day of the Week"
    if col.lower() == "day of the week"
    else col
    for col in df.columns
]

required_columns = [
    "Time", "Date", "Day of the Week",
    "CarCount", "BikeCount", "BusCount",
    "TruckCount", "Total", "Traffic Situation"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.drop_duplicates().copy()
df = df.dropna(subset=["Traffic Situation"]).copy()

count_columns = [
    "CarCount", "BikeCount", "BusCount",
    "TruckCount", "Total"
]

for col in count_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=count_columns).copy()

df["Traffic Situation"] = (
    df["Traffic Situation"].astype(str).str.strip().str.lower()
)

date_values = pd.to_datetime(
    df["Date"], errors="coerce", dayfirst=True
)

time_values = pd.to_datetime(
    df["Time"].astype(str).str.strip(),
    format="mixed",
    errors="coerce"
)

if date_values.isna().any() or time_values.isna().any():
    raise ValueError(
        "Some date/time values could not be parsed. "
        "Please check their format."
    )

df["Hour"] = time_values.dt.hour
df["Minute"] = time_values.dt.minute
df["DayOfWeek"] = date_values.dt.dayofweek
df["Month"] = date_values.dt.month
df["DayOfMonth"] = date_values.dt.day
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

features = [
    "CarCount", "BikeCount", "BusCount",
    "TruckCount", "Total",
    "Hour", "Minute", "DayOfWeek",
    "Month", "DayOfMonth", "IsWeekend"
]

X = df[features]
y = df["Traffic Situation"]

print("\nTarget class distribution:")
print(y.value_counts())

df["DateTime"] = (
    date_values.dt.normalize()
    + pd.to_timedelta(time_values.dt.hour, unit="h")
    + pd.to_timedelta(time_values.dt.minute, unit="m")
    + pd.to_timedelta(time_values.dt.second, unit="s")
)

df = df.sort_values("DateTime").reset_index(drop=True)

X = df[features]
y = df["Traffic Situation"]

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

if y_train.nunique() < 2:
    raise ValueError(
        "Training data has fewer than two classes. "
        "Check the chronological split."
    )

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)
print("Training completed successfully!")

y_pred = model.predict(X_test)

print("\n========== EVALUATION RESULTS ==========")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

print("\nTraining and evaluation completed.")
