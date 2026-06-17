import pandas as pd
import boto3
import requests
import zipfile
import io
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client("s3")
BUCKET = os.getenv("S3_BUCKET", "insy684")

# ── 1. Download & Upload Raw BIXI Trip Data ────────────────────────────────

urls = {
    "2024": "https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2025/01/DonneesOuvertes2024_010203040506070809101112.zip",
    "2025": "https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2026/02/DonneesOuvertes2025_010203040506070809101112.zip",
    "2026": "https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2026/02/DonneesOuvertes2026_01020304.zip",
}

for year, url in urls.items():
    print(f"Downloading {year} trip data...")
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                s3_key = f"bixi-data/{year}/{filename}"
                s3.upload_fileobj(f, BUCKET, s3_key)
                print(f"  Uploaded: {s3_key}")

# ── 2. Split & Aggregate Demand at 15-min Level ────────────────────────────

def process_year(s3_key, year_label, months=None):
    """
    Load trip data from S3, split into departure/arrival,
    optionally filter by months, aggregate at 15-min level, upload.
    """
    print(f"\nLoading {s3_key}...")
    obj = s3.get_object(Bucket=BUCKET, Key=s3_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    # Convert timestamps
    df["STARTTIMEMS"] = pd.to_datetime(df["STARTTIMEMS"], unit="ms", utc=True).dt.tz_localize(None)
    df["ENDTIMEMS"] = pd.to_datetime(df["ENDTIMEMS"], unit="ms", utc=True).dt.tz_localize(None)

    # Filter months if specified (e.g. May=5, October=10)
    if months:
        df = df[df["STARTTIMEMS"].dt.month.isin(months)]

    for trip_type in ["departure", "arrival"]:
        if trip_type == "departure":
            subset = df[["STARTSTATIONNAME", "STARTTIMEMS",
                         "STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE"]].copy()
            subset.columns = ["station_name", "datetime", "latitude", "longitude"]
        else:
            subset = df[["ENDSTATIONNAME", "ENDTIMEMS",
                         "ENDSTATIONLATITUDE", "ENDSTATIONLONGITUDE"]].copy()
            subset.columns = ["station_name", "datetime", "latitude", "longitude"]

        # Floor to 15-min window
        subset["time_15min"] = subset["datetime"].dt.floor("15min")

        # Aggregate demand
        demand = (
            subset.groupby(["station_name", "time_15min", "latitude", "longitude"])
            .size()
            .reset_index(name="demand")
        )

        # Upload
        out_key = f"processed-data/{year_label}_{trip_type}_demand_15min.csv"
        buf = io.StringIO()
        demand.to_csv(buf, index=False)
        s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
        print(f"  Uploaded: {out_key}, shape: {demand.shape}")

# 2024 full year
process_year(
    s3_key="bixi-data/2024/DonneesOuvertes2024_010203040506070809101112.csv",
    year_label="2024"
)

# 2025 May (validation) and October (test)
process_year(
    s3_key="bixi-data/2025/DonneesOuvertes2025_010203040506070809101112.csv",
    year_label="2025_may",
    months=[5]
)
process_year(
    s3_key="bixi-data/2025/DonneesOuvertes2025_010203040506070809101112.csv",
    year_label="2025_oct",
    months=[10]
)

print("\nAll done!")