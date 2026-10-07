import pandas as pd
import joblib


# ============================================================
# AQUATRACK - AVAILABILITY MODEL TEST
# ============================================================

model_file = "models/availability_linear_regression.joblib"

model = joblib.load(model_file)


# ------------------------------------------------------------
# Test Input
# ------------------------------------------------------------

previous_supply_hours = 8.0
rainfall = 2.0
season = "Summer"


# ------------------------------------------------------------
# Convert Season
# ------------------------------------------------------------

season_mapping = {
    "Summer": 0,
    "Monsoon": 1,
    "Winter": 2
}


if season not in season_mapping:

    raise ValueError(
        f"Invalid season: {season}"
    )


season_code = season_mapping[season]


# ------------------------------------------------------------
# Create Test DataFrame
# ------------------------------------------------------------

sample = pd.DataFrame(
    [{
        "previous_supply_hours": previous_supply_hours,
        "rainfall": rainfall,
        "season_code": season_code
    }]
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

prediction = model.predict(sample)[0]

prediction = max(
    0.0,
    min(24.0, float(prediction))
)


# ------------------------------------------------------------
# Availability Classification
# ------------------------------------------------------------

if prediction < 5:

    status = "Scarcity Risk"

elif prediction < 8:

    status = "Moderate Availability"

else:

    status = "Good Availability"


# ------------------------------------------------------------
# Display Result
# ------------------------------------------------------------

print("=" * 50)
print("AquaTrack Availability Model Test")
print("=" * 50)

print(
    f"Previous Supply Hours: "
    f"{previous_supply_hours:.2f}"
)

print(
    f"Rainfall: "
    f"{rainfall:.2f} mm"
)

print(
    f"Season: "
    f"{season}"
)

print(
    f"Forecast Supply: "
    f"{prediction:.2f} hours"
)

print(
    f"Availability Status: "
    f"{status}"
)

print("=" * 50)