from prediction import (
    predict_water_quality,
    forecast_water_availability
)


# ============================================================
# Test Water Quality Prediction
# ============================================================

quality_data = {
    "pH": 7.2,
    "TDS": 350,
    "turbidity": 0.8,
    "hardness": 180,
    "chloride": 200,
    "fluoride": 0.7,
    "nitrate": 25
}

risk = predict_water_quality(quality_data)

print("Water Quality Risk:", risk)


# ============================================================
# Test Water Availability Forecast
# ============================================================

previous_supply_hours = 8.0
rainfall = 2.0
season = "Summer"

availability_result = forecast_water_availability(
    previous_supply_hours,
    rainfall,
    season
)

print(
    "Water Availability Forecast:",
    availability_result["forecast_hours"],
    "hours"
)

print(
    "Availability Status:",
    availability_result["status"]
)