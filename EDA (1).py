# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import io
import os
from dotenv import load_dotenv

load_dotenv()


s3 = boto3.client("s3")

# 2024 departure demand
obj = s3.get_object(Bucket="insy684", Key="processed-data/2024_departure_demand_15min.csv")
df = pd.read_csv(io.BytesIO(obj["Body"].read()))
df["time_15min"] = pd.to_datetime(df["time_15min"])

print("Shape:", df.shape)
print("\nsummary：")
print(df["demand"].describe())
print("\n missing value：")
print(df.isnull().sum())

# %%
df["hour"] = df["time_15min"].dt.hour
df["dayofweek"] = df["time_15min"].dt.dayofweek  # 0=Monday
df["month"] = df["time_15min"].dt.month

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Demand distribution
axes[0,0].hist(df["demand"], bins=50, color="steelblue", edgecolor="white")
axes[0,0].set_title("Demand Distribution")
axes[0,0].set_xlabel("Demand")
axes[0,0].set_ylabel("Count")

# 2. Average demand by hour
hourly = df.groupby("hour")["demand"].mean()
axes[0,1].plot(hourly.index, hourly.values, marker="o", color="steelblue")
axes[0,1].set_title("Average Demand by Hour")
axes[0,1].set_xlabel("Hour")
axes[0,1].set_ylabel("Avg Demand")

# 3. Average demand by day of week
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily = df.groupby("dayofweek")["demand"].mean()
axes[1,0].bar(days, daily.values, color="steelblue")
axes[1,0].set_title("Average Demand by Day of Week")
axes[1,0].set_ylabel("Avg Demand")

# 4. Average demand by month
monthly = df.groupby("month")["demand"].mean()
axes[1,1].bar(monthly.index, monthly.values, color="steelblue")
axes[1,1].set_title("Average Demand by Month")
axes[1,1].set_xlabel("Month")
axes[1,1].set_ylabel("Avg Demand")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.show()

# %%
# Load arrival demand for comparison
obj_arr = s3.get_object(Bucket="insy684", Key="processed-data/2024_arrival_demand_15min.csv")
df_arr = pd.read_csv(io.BytesIO(obj_arr["Body"].read()))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Top 15 stations by departure demand
top_dep = df.groupby("station_name")["demand"].sum().nlargest(15)
axes[0,0].barh(top_dep.index[::-1], top_dep.values[::-1], color="steelblue")
axes[0,0].set_title("Top 15 Stations by Total Departure Demand")
axes[0,0].set_xlabel("Total Demand")

# 2. Top 15 stations by arrival demand
top_arr = df_arr.groupby("station_name")["demand"].sum().nlargest(15)
axes[0,1].barh(top_arr.index[::-1], top_arr.values[::-1], color="coral")
axes[0,1].set_title("Top 15 Stations by Total Arrival Demand")
axes[0,1].set_xlabel("Total Demand")

# 3. Departure vs Arrival demand by hour
dep_hourly = df.copy()
dep_hourly["hour"] = pd.to_datetime(dep_hourly["time_15min"]).dt.hour
arr_hourly = df_arr.copy()
arr_hourly["hour"] = pd.to_datetime(arr_hourly["time_15min"]).dt.hour

dep_by_hour = dep_hourly.groupby("hour")["demand"].mean()
arr_by_hour = arr_hourly.groupby("hour")["demand"].mean()

axes[1,0].plot(dep_by_hour.index, dep_by_hour.values, marker="o", label="Departure", color="steelblue")
axes[1,0].plot(arr_by_hour.index, arr_by_hour.values, marker="o", label="Arrival", color="coral")
axes[1,0].set_title("Departure vs Arrival Demand by Hour")
axes[1,0].set_xlabel("Hour")
axes[1,0].set_ylabel("Avg Demand")
axes[1,0].legend()

# 4. Departure vs Arrival demand by day of week
dep_hourly["dayofweek"] = pd.to_datetime(dep_hourly["time_15min"]).dt.dayofweek
arr_hourly["dayofweek"] = pd.to_datetime(arr_hourly["time_15min"]).dt.dayofweek

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dep_by_day = dep_hourly.groupby("dayofweek")["demand"].mean()
arr_by_day = arr_hourly.groupby("dayofweek")["demand"].mean()

x = range(7)
width = 0.35
axes[1,1].bar([i - width/2 for i in x], dep_by_day.values, width, label="Departure", color="steelblue")
axes[1,1].bar([i + width/2 for i in x], arr_by_day.values, width, label="Arrival", color="coral")
axes[1,1].set_title("Departure vs Arrival Demand by Day of Week")
axes[1,1].set_xticks(list(x))
axes[1,1].set_xticklabels(days)
axes[1,1].set_ylabel("Avg Demand")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("eda_stations.png", dpi=150)
plt.show()

# %%
# Load weather data
obj_weather = s3.get_object(Bucket="insy684", Key="weather-data/2024_weather_15min.csv")
df_weather = pd.read_csv(io.BytesIO(obj_weather["Body"].read()))
df_weather["time"] = pd.to_datetime(df_weather["time"])

print(df_weather.columns.tolist())
print(df_weather.head())

# %%
# Aggregate demand to hourly level for merging with weather
df["time_15min"] = pd.to_datetime(df["time_15min"])
demand_15min = df.groupby("time_15min")["demand"].mean().reset_index()

# Merge with weather data
merged = pd.merge(demand_15min, df_weather, left_on="time_15min", right_on="time", how="inner")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Temperature vs demand
axes[0,0].scatter(merged["temperature_2m"], merged["demand"], alpha=0.1, color="steelblue", s=1)
axes[0,0].set_title("Temperature vs Demand")
axes[0,0].set_xlabel("Temperature (°C)")
axes[0,0].set_ylabel("Avg Demand")

# 2. Precipitation vs demand
axes[0,1].scatter(merged["precipitation"], merged["demand"], alpha=0.1, color="coral", s=1)
axes[0,1].set_title("Precipitation vs Demand")
axes[0,1].set_xlabel("Precipitation (mm)")
axes[0,1].set_ylabel("Avg Demand")

# 3. Wind speed vs demand
axes[1,0].scatter(merged["wind_speed_10m"], merged["demand"], alpha=0.1, color="green", s=1)
axes[1,0].set_title("Wind Speed vs Demand")
axes[1,0].set_xlabel("Wind Speed (km/h)")
axes[1,0].set_ylabel("Avg Demand")

# 4. Average demand by weather code (top 10 most common)
top_codes = merged["weather_code"].value_counts().nlargest(10).index
weather_demand = merged[merged["weather_code"].isin(top_codes)].groupby("weather_code")["demand"].mean().sort_values()
axes[1,1].barh(weather_demand.index.astype(str), weather_demand.values, color="steelblue")
axes[1,1].set_title("Avg Demand by Weather Code (Top 10)")
axes[1,1].set_xlabel("Avg Demand")
axes[1,1].set_ylabel("Weather Code")

plt.tight_layout()
plt.savefig("eda_weather.png", dpi=150)
plt.show()

# %%



