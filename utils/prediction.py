import os
import joblib
import pandas as pd
from functools import lru_cache


# ============================================================
# AQUATRACK - PREDICTION FUNCTIONS
# ============================================================

# Model paths
QUALITY_MODEL_FILE = "models/water_quality_rf.joblib"
AVAILABILITY_MODEL_FILE = "models/availability_linear_regression.joblib"


# ------------------------------------------------------------
# Load Water Quality Model (cached — loaded from disk only once)
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_quality_model():
    if not os.path.exists(QUALITY_MODEL_FILE):
        raise FileNotFoundError(
            f"Water quality model not found: {QUALITY_MODEL_FILE}"
        )

    return joblib.load(QUALITY_MODEL_FILE)


# ------------------------------------------------------------
# Load Availability Model (cached — loaded from disk only once)
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_availability_model():
    if not os.path.exists(AVAILABILITY_MODEL_FILE):
        raise FileNotFoundError(
            f"Availability model not found: {AVAILABILITY_MODEL_FILE}"
        )

    return joblib.load(AVAILABILITY_MODEL_FILE)


# ------------------------------------------------------------
# Water Quality Prediction
# ------------------------------------------------------------

def predict_water_quality(data):
    """
    Predict water-quality risk.

    Expected input:
    pH, TDS, turbidity, hardness,
    chloride, fluoride, nitrate
    """

    model = load_quality_model()

    input_data = pd.DataFrame([{
        "pH": data["pH"],
        "TDS": data["TDS"],
        "turbidity": data["turbidity"],
        "hardness": data["hardness"],
        "chloride": data["chloride"],
        "fluoride": data["fluoride"],
        "nitrate": data["nitrate"]
    }])

    prediction = model.predict(input_data)

    return prediction[0]


# ------------------------------------------------------------
# Water Availability Forecast
# ------------------------------------------------------------

def forecast_water_availability(
    previous_supply_hours,
    rainfall,
    season
):
    """
    Forecast water supply hours using:

    1. Previous-day supply hours
    2. Rainfall
    3. Season
    """

    model = load_availability_model()

    season_mapping = {
        "Summer": 0,
        "Monsoon": 1,
        "Winter": 2
    }

    if season not in season_mapping:
        raise ValueError(
            f"Invalid season: {season}. "
            f"Use Summer, Monsoon, or Winter."
        )

    season_code = season_mapping[season]

    input_data = pd.DataFrame([{
        "previous_supply_hours": previous_supply_hours,
        "rainfall": rainfall,
        "season_code": season_code
    }])

    prediction = model.predict(input_data)

    forecast_hours = float(prediction[0])

    # Keep forecast within a realistic daily range
    forecast_hours = max(0.0, min(24.0, forecast_hours))

    # --------------------------------------------------------
    # Availability classification
    # --------------------------------------------------------

    if forecast_hours < 5:
        status = "Scarcity Risk"
    elif forecast_hours < 8:
        status = "Moderate Availability"
    else:
        status = "Good Availability"

    return {
        "forecast_hours": round(forecast_hours, 2),
        "status": status
    }