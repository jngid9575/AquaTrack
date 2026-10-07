import sqlite3
import pandas as pd


DATABASE_NAME = "aquatrack.db"


def import_water_quality():
    connection = sqlite3.connect(DATABASE_NAME)

    df = pd.read_csv("data/water_quality.csv")

    zone_map = pd.read_sql_query(
        "SELECT zone_id, zone_name FROM zones",
        connection
    )

    for _, row in df.iterrows():

        zone_id = zone_map.loc[
            zone_map["zone_name"] == row["zone"],
            "zone_id"
        ].iloc[0]

        connection.execute(
            """
            INSERT INTO water_quality
            (
                zone_id,
                date,
                ph,
                tds,
                turbidity,
                hardness,
                chloride,
                fluoride,
                nitrate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(zone_id),
                row["date"],
                row["pH"],
                row["TDS"],
                row["turbidity"],
                row["hardness"],
                row["chloride"],
                row["fluoride"],
                row["nitrate"]
            )
        )

    connection.commit()
    connection.close()

    print("Water-quality data imported successfully.")


def import_availability():
    connection = sqlite3.connect(DATABASE_NAME)

    df = pd.read_csv("data/availability.csv")

    zone_map = pd.read_sql_query(
        "SELECT zone_id, zone_name FROM zones",
        connection
    )

    for _, row in df.iterrows():

        zone_id = zone_map.loc[
            zone_map["zone_name"] == row["zone"],
            "zone_id"
        ].iloc[0]

        connection.execute(
            """
            INSERT INTO availability
            (
                zone_id,
                date,
                supply_hours,
                rainfall,
                season
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(zone_id),
                row["date"],
                row["supply_hours"],
                row["rainfall"],
                row["season"]
            )
        )

    connection.commit()
    connection.close()

    print("Availability data imported successfully.")


if __name__ == "__main__":

    import_water_quality()
    import_availability()

    print("All AquaTrack data imported successfully.")