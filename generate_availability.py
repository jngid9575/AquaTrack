import os
import numpy as np
import pandas as pd

np.random.seed(42)

zones = [
    "Vijay Nagar",
    "Palasia",
    "Bhawarkuan",
    "Khajrana",
    "Bengali Square",
    "Rau",
    "Rajendra Nagar",
    "Annapurna",
    "Sudama Nagar",
    "Aerodrome",
    "Kanadia",
    "Silicon City"
]

records = []

start_date = pd.Timestamp("2025-01-01")

for zone in zones:
    for i in range(365):

        date = start_date + pd.Timedelta(days=i)

        # Determine season
        month = date.month

        if month in [6, 7, 8, 9]:
            season = "Monsoon"
            rainfall = np.clip(np.random.normal(8, 5), 0, 20)
        elif month in [10, 11, 12, 1, 2]:
            season = "Winter"
            rainfall = np.clip(np.random.normal(1.5, 1.5), 0, 8)
        else:
            season = "Summer"
            rainfall = np.clip(np.random.normal(0.5, 0.8), 0, 5)

        # Generate realistic project-level supply hours
        base_supply = 8

        # More rainfall generally improves availability
        rainfall_effect = rainfall * 0.12

        # Summer generally has lower availability
        if season == "Summer":
            seasonal_effect = -1.5
        elif season == "Monsoon":
            seasonal_effect = 1.0
        else:
            seasonal_effect = 0.3

        noise = np.random.normal(0, 0.7)

        supply_hours = (
            base_supply
            + rainfall_effect
            + seasonal_effect
            + noise
        )

        supply_hours = np.clip(supply_hours, 2, 12)

        records.append([
            zone,
            date,
            round(supply_hours, 2),
            round(rainfall, 2),
            season
        ])


columns = [
    "zone",
    "date",
    "supply_hours",
    "rainfall",
    "season"
]

df = pd.DataFrame(records, columns=columns)

os.makedirs("data", exist_ok=True)

output_file = "data/availability.csv"

df.to_csv(output_file, index=False)

print("Water-availability dataset created successfully.")
print(f"File: {output_file}")
print(f"Total records: {len(df)}")
print(f"Zones: {df['zone'].nunique()}")

print("\nSeason distribution:")
print(df["season"].value_counts())

print("\nFirst 5 records:")
print(df.head())