import pandas as pd

TARGET = "Vehicles"
DATE_COL = "DateTime"
GROUP_COL = "Junction"

def add_features(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    # time features
    df["hour"] = df[DATE_COL].dt.hour
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["day"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([8, 9, 17, 18, 19]).astype(int)

    # lag features, computed separately for each junction
    grp = df.groupby(GROUP_COL)[TARGET]
    df["lag_1"] = grp.shift(1)       # 1 hour ago
    df["lag_24"] = grp.shift(24)     # same hour yesterday
    df["lag_168"] = grp.shift(168)   # same hour last week

    # junction one-hot -> Junction_1 ... Junction_4
    df = pd.get_dummies(df, columns=[GROUP_COL], prefix="Junction", dtype=int)

    return df.dropna().reset_index(drop=True)

def get_X_y(df):
    X = df.drop(columns=[TARGET, DATE_COL])
    return X, df[TARGET]
