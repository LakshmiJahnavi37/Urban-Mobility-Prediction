import pandas as pd

TARGET = "Vehicles"
DATE_COL = "DateTime"
GROUP_COL = "Junction"

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # ID is just a unique key (date + junction) - not useful for modelling
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.drop_duplicates(subset=[DATE_COL, GROUP_COL])
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0]
    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = clean_data(load_data("data/raw/traffic.csv"))
    df.to_csv("data/processed/traffic_clean.csv", index=False)
    print("Saved cleaned data:", df.shape)
    print(df.groupby(GROUP_COL)[DATE_COL].agg(["min", "max", "count"]))
