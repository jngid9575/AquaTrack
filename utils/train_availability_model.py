import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# AQUATRACK - AVAILABILITY FORECASTING MODEL
# ============================================================

# Load availability dataset
data_file = "data/availability.csv"
df = pd.read_csv(data_file)


# ------------------------------------------------------------
# Validate required columns
# ------------------------------------------------------------

required_columns = [
    "zone",
    "date",
    "supply_hours",
    "rainfall",
    "season"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in availability.csv: {missing_columns}"
    )


# ------------------------------------------------------------
# Prepare date information
# ------------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

# Sort by zone and date so that previous-day supply
# can be calculated correctly.
df = df.sort_values(["zone", "date"]).reset_index(drop=True)


# ------------------------------------------------------------
# Create historical supply-hours feature
# ------------------------------------------------------------

# Previous day's supply hours represent historical
# availability information available before the prediction.
df["previous_supply_hours"] = (
    df.groupby("zone")["supply_hours"].shift(1)
)


# The first record of every zone has no previous-day
# supply value, so remove those records.
df = df.dropna(
    subset=["previous_supply_hours"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Convert season into numerical values
# ------------------------------------------------------------

season_mapping = {
    "Summer": 0,
    "Monsoon": 1,
    "Winter": 2
}

df["season_code"] = df["season"].map(season_mapping)


# Check for unknown season values
if df["season_code"].isna().any():
    unknown_seasons = df.loc[
        df["season_code"].isna(),
        "season"
    ].unique()

    raise ValueError(
        f"Unknown season values found: {unknown_seasons}"
    )


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

features = [
    "previous_supply_hours",
    "rainfall",
    "season_code"
]


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

target = "supply_hours"


X = df[features]
y = df[target]


# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# Create Linear Regression model
# ------------------------------------------------------------

model = LinearRegression()


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

model.fit(X_train, y_train)


# ------------------------------------------------------------
# Make predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ------------------------------------------------------------
# Create models directory
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)


# ------------------------------------------------------------
# Save trained model
# ------------------------------------------------------------

model_file = "models/availability_linear_regression.joblib"

joblib.dump(model, model_file)


# ------------------------------------------------------------
# Display training results
# ------------------------------------------------------------

print("=" * 60)
print("AquaTrack Availability Forecasting Model")
print("=" * 60)

print("Linear Regression availability model trained successfully.")

print(f"Total records used: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print()
print("Features used:")
for feature in features:
    print(f" - {feature}")

print()
print(f"Mean Absolute Error: {mae:.2f} hours")
print(f"R2 Score: {r2:.4f}")

print()
print(f"Model saved to: {model_file}")

print("=" * 60)