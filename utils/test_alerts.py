from alerts import generate_alert
from alerts import generate_recommendations


risk = "High Risk"
availability = "Moderate Availability"


alerts = generate_alert(
    risk,
    availability
)

recommendations = generate_recommendations(
    risk,
    availability
)


print("AquaTrack Alerts")
print("----------------")

for alert in alerts:
    print(
        alert["severity"],
        "|",
        alert["type"],
        "|",
        alert["message"]
    )


print("\nRecommendations")
print("----------------")

for recommendation in recommendations:
    print("-", recommendation)