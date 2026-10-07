import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load labeled dataset
data_file = "data/water_quality_labeled.csv"
df = pd.read_csv(data_file)


# Features used by the Random Forest model
features = [
    "pH",
    "TDS",
    "turbidity",
    "hardness",
    "chloride",
    "fluoride",
    "nitrate"
]

X = df[features]
y = df["risk_level"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


# Test the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# Create models directory if needed
os.makedirs("models", exist_ok=True)


# Save trained model
model_file = "models/water_quality_rf.joblib"
joblib.dump(model, model_file)


print("Random Forest model trained successfully.")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Model saved to: {model_file}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))