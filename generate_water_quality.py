import pandas as pd
import numpy as np


# ============================================================
# AQUATRACK WATER QUALITY DATA GENERATOR
# ============================================================

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


# ============================================================
# GENERATE 100 OBSERVATIONS PER ZONE
# ============================================================

for zone in zones:

    for day in range(100):

        date = pd.Timestamp(
            "2025-01-01"
        ) + pd.Timedelta(
            days=day
        )


        # ----------------------------------------------------
        # Create three broad project-level water conditions
        # ----------------------------------------------------

        condition = np.random.choice(
            [
                "Safe",
                "Moderate Risk",
                "High Risk"
            ],
            p=[
                0.30,
                0.40,
                0.30
            ]
        )


        # ----------------------------------------------------
        # SAFE CONDITION
        # ----------------------------------------------------

        if condition == "Safe":

            ph = np.random.normal(
                7.2,
                0.35
            )

            tds = np.random.normal(
                300,
                70
            )

            turbidity = np.random.normal(
                0.55,
                0.20
            )

            hardness = np.random.normal(
                150,
                30
            )

            chloride = np.random.normal(
                150,
                40
            )

            fluoride = np.random.normal(
                0.65,
                0.15
            )

            nitrate = np.random.normal(
                20,
                8
            )


        # ----------------------------------------------------
        # MODERATE RISK CONDITION
        # ----------------------------------------------------

        elif condition == "Moderate Risk":

            ph = np.random.normal(
                7.7,
                0.65
            )

            tds = np.random.normal(
                480,
                120
            )

            turbidity = np.random.normal(
                1.2,
                0.55
            )

            hardness = np.random.normal(
                210,
                50
            )

            chloride = np.random.normal(
                270,
                70
            )

            fluoride = np.random.normal(
                0.95,
                0.25
            )

            nitrate = np.random.normal(
                38,
                15
            )


        # ----------------------------------------------------
        # HIGH RISK CONDITION
        # ----------------------------------------------------

        else:

            ph = np.random.normal(
                8.4,
                0.55
            )

            tds = np.random.normal(
                700,
                180
            )

            turbidity = np.random.normal(
                2.5,
                1.0
            )

            hardness = np.random.normal(
                300,
                70
            )

            chloride = np.random.normal(
                400,
                100
            )

            fluoride = np.random.normal(
                1.3,
                0.35
            )

            nitrate = np.random.normal(
                65,
                20
            )


        # ----------------------------------------------------
        # Keep values physically reasonable
        # ----------------------------------------------------

        ph = np.clip(
            ph,
            5.0,
            10.0
        )

        tds = np.clip(
            tds,
            50,
            1200
        )

        turbidity = np.clip(
            turbidity,
            0.05,
            8.0
        )

        hardness = np.clip(
            hardness,
            50,
            500
        )

        chloride = np.clip(
            chloride,
            30,
            700
        )

        fluoride = np.clip(
            fluoride,
            0.1,
            2.5
        )

        nitrate = np.clip(
            nitrate,
            2,
            120
        )


        records.append(
            {
                "zone": zone,
                "date": date.strftime(
                    "%Y-%m-%d"
                ),
                "pH": round(
                    ph,
                    2
                ),
                "TDS": round(
                    tds,
                    2
                ),
                "turbidity": round(
                    turbidity,
                    2
                ),
                "hardness": round(
                    hardness,
                    2
                ),
                "chloride": round(
                    chloride,
                    2
                ),
                "fluoride": round(
                    fluoride,
                    2
                ),
                "nitrate": round(
                    nitrate,
                    2
                )
            }
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

data = pd.DataFrame(
    records
)


# ============================================================
# SAVE DATASET
# ============================================================

data.to_csv(
    "data/water_quality.csv",
    index=False
)


print(
    "Water-quality dataset created successfully."
)

print(
    "File: data/water_quality.csv"
)

print(
    f"Total records: {len(data)}"
)

print(
    f"Zones: {data['zone'].nunique()}"
)