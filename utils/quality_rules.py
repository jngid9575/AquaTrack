import pandas as pd


# Reference limits used by the AquaTrack prototype.
# These values are used for project-level screening,
# not as a replacement for official laboratory testing.

LIMITS = {
    "pH_min": 6.5,
    "pH_max": 8.5,
    "TDS_max": 500,
    "turbidity_max": 1,
    "hardness_max": 200,
    "chloride_max": 250,
    "fluoride_max": 1.0,
    "nitrate_max": 45,
}


def check_parameter_limits(row):
    """Check water-quality parameters against prototype limits."""

    violations = []

    if not (LIMITS["pH_min"] <= row["pH"] <= LIMITS["pH_max"]):
        violations.append("pH")

    if row["TDS"] > LIMITS["TDS_max"]:
        violations.append("TDS")

    if row["turbidity"] > LIMITS["turbidity_max"]:
        violations.append("Turbidity")

    if row["hardness"] > LIMITS["hardness_max"]:
        violations.append("Hardness")

    if row["chloride"] > LIMITS["chloride_max"]:
        violations.append("Chloride")

    if row["fluoride"] > LIMITS["fluoride_max"]:
        violations.append("Fluoride")

    if row["nitrate"] > LIMITS["nitrate_max"]:
        violations.append("Nitrate")

    return violations


def classify_water_quality(row):
    """
    Assign a project-level risk category.

    0 violations  -> Safe
    1-2 violations -> Moderate Risk
    3+ violations -> High Risk
    """

    violations = check_parameter_limits(row)

    if len(violations) == 0:
        risk = "Safe"
    elif len(violations) <= 2:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return risk


def add_risk_labels(input_file="data/water_quality.csv",
                    output_file="data/water_quality_labeled.csv"):
    """Read water-quality data, classify it, and save labeled data."""

    df = pd.read_csv(input_file)

    df["risk_level"] = df.apply(classify_water_quality, axis=1)

    df.to_csv(output_file, index=False)

    print("Water-quality risk labels created successfully.")
    print(f"Input records: {len(df)}")
    print(f"Output file: {output_file}")
    print("\nRisk distribution:")
    print(df["risk_level"].value_counts())


if __name__ == "__main__":
    add_risk_labels()