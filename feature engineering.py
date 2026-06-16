# %%
import pandas as pd
import numpy as np
import boto3
import io
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client("s3")

# Load 2024 departure demand
obj = s3.get_object(Bucket="insy684", Key="processed-data/2024_departure_demand_15min.csv")
df = pd.read_csv(io.BytesIO(obj["Body"].read()))
df["time_15min"] = pd.to_datetime(df["time_15min"])

print(df.shape)
print(df.head())
print(df.dtypes)

# %%
# Step 1: Temporal features
df["hour"] = df["time_15min"].dt.hour
df["minute"] = df["time_15min"].dt.minute
df["dayofweek"] = df["time_15min"].dt.dayofweek  # 0=Monday, 6=Sunday
df["month"] = df["time_15min"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["is_rush_hour"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 9)) |
    ((df["hour"] >= 17) & (df["hour"] <= 19))
).astype(int)

# 15-minute slot index (0-95, representing each 15-min window in a day)
df["slot_of_day"] = df["hour"] * 4 + df["minute"] // 15

print(df[["time_15min", "hour", "minute", "dayofweek", "month", 
          "is_weekend", "is_rush_hour", "slot_of_day"]].head(10))

# %%
# Step 2: Historical demand features
# Based on 2024 data: average demand by station + dayofweek + slot_of_day
historical_avg = (
    df.groupby(["station_name", "dayofweek", "slot_of_day"])["demand"]
    .mean()
    .reset_index()
    .rename(columns={"demand": "hist_avg_demand"})
)

# Merge back to main dataframe
df = df.merge(historical_avg, on=["station_name", "dayofweek", "slot_of_day"], how="left")

print(df[["station_name", "time_15min", "demand", "hist_avg_demand"]].head(10))
print("\nNull count:", df["hist_avg_demand"].isnull().sum())

# %%
# Step 3: Lag features
# Sort by station and time first to ensure correct lag calculation
df = df.sort_values(["station_name", "time_15min"]).reset_index(drop=True)

# Create lag features per station
df["lag_1"] = df.groupby("station_name")["demand"].shift(1)   # 15 min ago
df["lag_4"] = df.groupby("station_name")["demand"].shift(4)   # 1 hour ago
df["lag_96"] = df.groupby("station_name")["demand"].shift(96)  # 1 day ago
df["lag_672"] = df.groupby("station_name")["demand"].shift(672) # 1 week ago

# Rolling mean (past 4 slots = past 1 hour)
df["rolling_mean_4"] = (
    df.groupby("station_name")["demand"]
    .transform(lambda x: x.shift(1).rolling(4).mean())
)

print(df[["station_name", "time_15min", "demand", 
          "lag_1", "lag_4", "lag_96", "lag_672", "rolling_mean_4"]].head(20))
print("\nNull counts:")
print(df[["lag_1", "lag_4", "lag_96", "lag_672", "rolling_mean_4"]].isnull().sum())

# %%
# Step 4: Merge weather data
obj_weather = s3.get_object(Bucket="insy684", Key="weather-data/2024_weather_15min.csv")
df_weather = pd.read_csv(io.BytesIO(obj_weather["Body"].read()))
df_weather["time"] = pd.to_datetime(df_weather["time"])

# Merge on time
df = df.merge(df_weather, left_on="time_15min", right_on="time", how="left")
df = df.drop(columns=["time"])

print(df.shape)
print(df[["time_15min", "temperature_2m", "precipitation", 
          "wind_speed_10m", "relative_humidity_2m", "weather_code"]].head())
print("\nWeather null counts:")
print(df[["temperature_2m", "precipitation", "wind_speed_10m", 
          "relative_humidity_2m", "weather_code"]].isnull().sum())

# %%
# Step 5: Handle missing values
# Fill weather nulls with forward fill
weather_cols = ["temperature_2m", "precipitation", "wind_speed_10m", 
                "relative_humidity_2m", "weather_code"]
df[weather_cols] = df[weather_cols].fillna(method="ffill")

# Drop rows where lag features are NaN (first week of data)
# Keep lag_1, lag_4 as required; drop rows where lag_96 or lag_672 is NaN
df_clean = df.dropna(subset=["lag_1", "lag_4", "lag_96"])
df_clean = df_clean.fillna({"lag_672": df_clean["lag_96"], 
                             "rolling_mean_4": df_clean["lag_1"]})

print("Shape after cleaning:", df_clean.shape)
print("Remaining nulls:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Upload to S3
csv_buffer = io.StringIO()
df_clean.to_csv(csv_buffer, index=False)
s3.put_object(Bucket="insy684", Key="processed-data/2024_departure_features.csv", 
              Body=csv_buffer.getvalue())
print("✅ Upload complete")

# %%
def run_feature_engineering(demand_key, weather_key, output_key, historical_df=None):
    """
    demand_key: S3 key for demand data
    weather_key: S3 key for weather data
    output_key: S3 key for output
    historical_df: if provided, use this as historical average base (for 2025 data)
    """
    print(f"Loading {demand_key}...")
    obj = s3.get_object(Bucket="insy684", Key=demand_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    df["time_15min"] = pd.to_datetime(df["time_15min"])

    # Step 1: Temporal features
    df["hour"] = df["time_15min"].dt.hour
    df["minute"] = df["time_15min"].dt.minute
    df["dayofweek"] = df["time_15min"].dt.dayofweek
    df["month"] = df["time_15min"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9)) |
        ((df["hour"] >= 17) & (df["hour"] <= 19))
    ).astype(int)
    df["slot_of_day"] = df["hour"] * 4 + df["minute"] // 15

    # Step 2: Historical demand features
    # Use provided historical_df (2024 base) or self
    base_df = historical_df if historical_df is not None else df
    historical_avg = (
        base_df.groupby(["station_name", "dayofweek", "slot_of_day"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "hist_avg_demand"})
    )
    df = df.merge(historical_avg, on=["station_name", "dayofweek", "slot_of_day"], how="left")

    # Step 3: Lag features
    df = df.sort_values(["station_name", "time_15min"]).reset_index(drop=True)
    df["lag_1"] = df.groupby("station_name")["demand"].shift(1)
    df["lag_4"] = df.groupby("station_name")["demand"].shift(4)
    df["lag_96"] = df.groupby("station_name")["demand"].shift(96)
    df["lag_672"] = df.groupby("station_name")["demand"].shift(672)
    df["rolling_mean_4"] = (
        df.groupby("station_name")["demand"]
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    # Step 4: Merge weather
    obj_weather = s3.get_object(Bucket="insy684", Key=weather_key)
    df_weather = pd.read_csv(io.BytesIO(obj_weather["Body"].read()))
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    df = df.merge(df_weather, left_on="time_15min", right_on="time", how="left")
    df = df.drop(columns=["time"])

    # Step 5: Handle missing values
    weather_cols = ["temperature_2m", "precipitation", "wind_speed_10m",
                    "relative_humidity_2m", "weather_code"]
    df[weather_cols] = df[weather_cols].ffill()
    df = df.dropna(subset=["lag_1", "lag_4", "lag_96"])
    df = df.fillna({"lag_672": df["lag_96"], "rolling_mean_4": df["lag_1"]})

    # Upload to S3
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket="insy684", Key=output_key, Body=csv_buffer.getvalue())
    print(f"✅ Done: {output_key}, shape: {df.shape}")
    return df

# %%
# 2024 arrival
run_feature_engineering(
    demand_key="processed-data/2024_arrival_demand_15min.csv",
    weather_key="weather-data/2024_weather_15min.csv",
    output_key="processed-data/2024_arrival_features.csv"
)

# Load 2024 departure as historical base for 2025
obj = s3.get_object(Bucket="insy684", Key="processed-data/2024_departure_demand_15min.csv")
hist_dep = pd.read_csv(io.BytesIO(obj["Body"].read()))
hist_dep["time_15min"] = pd.to_datetime(hist_dep["time_15min"])
hist_dep["dayofweek"] = hist_dep["time_15min"].dt.dayofweek
hist_dep["slot_of_day"] = hist_dep["time_15min"].dt.hour * 4 + hist_dep["time_15min"].dt.minute // 15

obj = s3.get_object(Bucket="insy684", Key="processed-data/2024_arrival_demand_15min.csv")
hist_arr = pd.read_csv(io.BytesIO(obj["Body"].read()))
hist_arr["time_15min"] = pd.to_datetime(hist_arr["time_15min"])
hist_arr["dayofweek"] = hist_arr["time_15min"].dt.dayofweek
hist_arr["slot_of_day"] = hist_arr["time_15min"].dt.hour * 4 + hist_arr["time_15min"].dt.minute // 15

# 2025 May departure
run_feature_engineering(
    demand_key="processed-data/2025_may_departure_demand_15min.csv",
    weather_key="weather-data/2025-may_weather_15min.csv",
    output_key="processed-data/2025_may_departure_features.csv",
    historical_df=hist_dep
)

# 2025 May arrival
run_feature_engineering(
    demand_key="processed-data/2025_may_arrival_demand_15min.csv",
    weather_key="weather-data/2025-may_weather_15min.csv",
    output_key="processed-data/2025_may_arrival_features.csv",
    historical_df=hist_arr
)

# 2025 Oct departure
run_feature_engineering(
    demand_key="processed-data/2025_oct_departure_demand_15min.csv",
    weather_key="weather-data/2025-oct_weather_15min.csv",
    output_key="processed-data/2025_oct_departure_features.csv",
    historical_df=hist_dep
)

# 2025 Oct arrival
run_feature_engineering(
    demand_key="processed-data/2025_oct_arrival_demand_15min.csv",
    weather_key="weather-data/2025-oct_weather_15min.csv",
    output_key="processed-data/2025_oct_arrival_features.csv",
    historical_df=hist_arr
)

# %%



