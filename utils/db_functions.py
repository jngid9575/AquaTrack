import sqlite3
import pandas as pd


DATABASE_NAME = "aquatrack.db"


def get_connection():
    """Create a database connection."""
    return sqlite3.connect(DATABASE_NAME)


def get_zones():
    """Return all project zones."""
    connection = get_connection()

    query = """
        SELECT zone_id, zone_name, latitude, longitude
        FROM zones
        ORDER BY zone_name
    """

    df = pd.read_sql_query(query, connection)

    connection.close()

    return df


def get_water_quality(zone_name=None):
    """Return water-quality records."""

    connection = get_connection()

    if zone_name:
        query = """
            SELECT
                z.zone_id,
                z.zone_name,
                w.date,
                w.ph,
                w.tds,
                w.turbidity,
                w.hardness,
                w.chloride,
                w.fluoride,
                w.nitrate
            FROM water_quality w
            JOIN zones z ON w.zone_id = z.zone_id
            WHERE z.zone_name = ?
            ORDER BY w.date
        """

        df = pd.read_sql_query(
            query,
            connection,
            params=(zone_name,)
        )

    else:
        query = """
            SELECT
                z.zone_id,
                z.zone_name,
                w.date,
                w.ph,
                w.tds,
                w.turbidity,
                w.hardness,
                w.chloride,
                w.fluoride,
                w.nitrate
            FROM water_quality w
            JOIN zones z ON w.zone_id = z.zone_id
            ORDER BY w.date
        """

        df = pd.read_sql_query(query, connection)

    connection.close()

    return df


def get_availability(zone_name=None):
    """Return water-availability records."""

    connection = get_connection()

    if zone_name:
        query = """
            SELECT
                z.zone_id,
                z.zone_name,
                a.date,
                a.supply_hours,
                a.rainfall,
                a.season
            FROM availability a
            JOIN zones z ON a.zone_id = z.zone_id
            WHERE z.zone_name = ?
            ORDER BY a.date
        """

        df = pd.read_sql_query(
            query,
            connection,
            params=(zone_name,)
        )

    else:
        query = """
            SELECT
                z.zone_id,
                z.zone_name,
                a.date,
                a.supply_hours,
                a.rainfall,
                a.season
            FROM availability a
            JOIN zones z ON a.zone_id = z.zone_id
            ORDER BY a.date
        """

        df = pd.read_sql_query(query, connection)

    connection.close()

    return df


def save_prediction(
    zone_name,
    prediction_date,
    risk_level=None,
    availability_forecast=None
):
    """Save a model prediction."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT zone_id
        FROM zones
        WHERE zone_name = ?
        """,
        (zone_name,)
    )

    result = cursor.fetchone()

    if result is None:
        connection.close()
        return False

    zone_id = result[0]

    cursor.execute(
        """
        INSERT INTO predictions
        (
            zone_id,
            prediction_date,
            risk_level,
            availability_forecast
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            zone_id,
            prediction_date,
            risk_level,
            availability_forecast
        )
    )

    connection.commit()
    connection.close()

    return True