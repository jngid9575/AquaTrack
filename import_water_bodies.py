from pathlib import Path
import csv
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "aquatrack.db"
CSV_PATH = BASE_DIR / "data" / "water_body_supply_data.csv"


def import_water_bodies():
    if not DATABASE_PATH.exists():
        print("ERROR: aquatrack.db was not found.")
        return

    if not CSV_PATH.exists():
        print("ERROR: data/water_body_supply_data.csv was not found.")
        return

    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    imported_bodies = 0
    skipped_bodies = 0
    imported_supply = 0
    skipped_supply = 0

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:

            # Find existing project-defined zone
            cursor.execute(
                """
                SELECT zone_id
                FROM zones
                WHERE zone_name = ?
                """,
                (row["zone_name"],)
            )

            zone = cursor.fetchone()

            if zone is None:
                print(f"WARNING: Zone not found: {row['zone_name']}")
                continue

            zone_id = zone[0]

            # Check whether water body already exists
            cursor.execute(
                """
                SELECT water_body_id
                FROM water_bodies
                WHERE name = ?
                """,
                (row["water_body_name"],)
            )

            existing_body = cursor.fetchone()

            if existing_body:
                water_body_id = existing_body[0]
                skipped_bodies += 1
            else:
                cursor.execute(
                    """
                    INSERT INTO water_bodies
                    (
                        name,
                        water_body_type,
                        location,
                        latitude,
                        longitude,
                        description,
                        active
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        row["water_body_name"],
                        row["water_body_type"],
                        row["location"],
                        float(row["latitude"]),
                        float(row["longitude"]),
                        row["description"],
                    )
                )

                water_body_id = cursor.lastrowid
                imported_bodies += 1

            # Check whether colony/society relationship already exists
            cursor.execute(
                """
                SELECT supply_id
                FROM water_body_supply
                WHERE water_body_id = ?
                  AND zone_id = ?
                  AND colony_name = ?
                  AND society_name = ?
                """,
                (
                    water_body_id,
                    zone_id,
                    row["colony_name"],
                    row["society_name"],
                )
            )

            existing_supply = cursor.fetchone()

            if existing_supply:
                skipped_supply += 1
            else:
                cursor.execute(
                    """
                    INSERT INTO water_body_supply
                    (
                        water_body_id,
                        zone_id,
                        colony_name,
                        society_name,
                        supply_description,
                        active
                    )
                    VALUES (?, ?, ?, ?, ?, 1)
                    """,
                    (
                        water_body_id,
                        zone_id,
                        row["colony_name"],
                        row["society_name"],
                        row["supply_description"],
                    )
                )

                imported_supply += 1

    conn.commit()

    # Final database counts
    cursor.execute("SELECT COUNT(*) FROM water_bodies")
    total_bodies = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM water_body_supply")
    total_supply = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM water_body_reports")
    total_reports = cursor.fetchone()[0]

    conn.close()

    print()
    print("========================================")
    print("AquaTrack Water Body Import")
    print("========================================")
    print(f"New water bodies imported : {imported_bodies}")
    print(f"Existing water bodies     : {skipped_bodies}")
    print(f"New supply relationships  : {imported_supply}")
    print(f"Existing relationships    : {skipped_supply}")
    print()
    print(f"Total water bodies        : {total_bodies}")
    print(f"Total supply relationships: {total_supply}")
    print(f"Total water body reports  : {total_reports}")
    print("========================================")
    print("Import completed successfully.")


if __name__ == "__main__":
    import_water_bodies()