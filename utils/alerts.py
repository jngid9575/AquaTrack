def generate_alert(risk_level, availability_status):
    """Generate an alert based on water quality and availability."""

    alerts = []

    if risk_level == "High Risk":
        alerts.append({
            "type": "Water Quality",
            "severity": "High",
            "message": "High water-quality risk detected. Further testing is recommended."
        })

    elif risk_level == "Moderate Risk":
        alerts.append({
            "type": "Water Quality",
            "severity": "Medium",
            "message": "Moderate water-quality risk detected. Monitoring is recommended."
        })

    if availability_status == "Scarcity Risk":
        alerts.append({
            "type": "Water Availability",
            "severity": "High",
            "message": "Possible water scarcity detected. Water conservation is recommended."
        })

    elif availability_status == "Moderate Availability":
        alerts.append({
            "type": "Water Availability",
            "severity": "Medium",
            "message": "Moderate water availability detected. Monitor supply conditions."
        })

    return alerts


def generate_recommendations(risk_level, availability_status):
    """Generate recommendations for the selected zone."""

    recommendations = []

    if risk_level == "High Risk":
        recommendations.append(
            "Conduct detailed water-quality testing before drinking use."
        )

    elif risk_level == "Moderate Risk":
        recommendations.append(
            "Increase water-quality monitoring frequency."
        )

    else:
        recommendations.append(
            "Continue regular water-quality monitoring."
        )

    if availability_status == "Scarcity Risk":
        recommendations.append(
            "Promote water conservation and monitor supply levels."
        )

    elif availability_status == "Moderate Availability":
        recommendations.append(
            "Monitor consumption and maintain efficient water usage."
        )

    else:
        recommendations.append(
            "Maintain normal water-management practices."
        )

    return recommendations