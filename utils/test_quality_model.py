import joblib
import pandas as pd


# Load trained Random Forest model
model = joblib.load("models/water_quality_rf.joblib")


# Example new water-quality sample
sample = pd.DataFrame([{
    "pH": 7.2,
    "TDS": 350,
    "turbidity": 0.8,
    "hardness": 180,
    "chloride": 200,
    "fluoride": 0.7,
    "nitrate": 25
}])


# Predict risk
prediction = model.predict(sample)[0]


print("AquaTrack Water Quality Prediction")
print("-----------------------------------")
print("pH:", sample["pH"].iloc[0])
print("TDS:", sample["TDS"].iloc[0])
print("Turbidity:", sample["turbidity"].iloc[0])
print("Hardness:", sample["hardness"].iloc[0])
print("Chloride:", sample["chloride"].iloc[0])
print("Fluoride:", sample["fluoride"].iloc[0])
print("Nitrate:", sample["nitrate"].iloc[0])

print("\nPredicted Risk Level:", prediction)