import sqlite3
from pathlib import Path


# ============================================================
# AquaTrack Database Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "aquatrack.db"


# ============================================================
# Database Connection
# ============================================================

def get_connection():
    """
    Create and return a SQLite connection for AquaTrack.

    Foreign-key enforcement is enabled for relationships between
    water bodies, zones, residents, and water-body reports.
    """
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


# ============================================================
# Create Database Tables
# ============================================================

def create_tables():
    """
    Create all AquaTrack tables if they do not already exist.

    IMPORTANT:
    - Existing tables and records are NOT deleted.
    - Existing project data is preserved.
    - New reporting tables are added safely.
    - Resident and alert tables are included for the
      zone-based alert notification system.
    """

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # Users
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
        """
    )

    # --------------------------------------------------------
    # Project-defined Zones
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS zones (
            zone_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_name TEXT NOT NULL,
            latitude REAL,
            longitude REAL
        )
        """
    )

    # --------------------------------------------------------
    # Water Quality
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS water_quality (
            quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            pH REAL NOT NULL,
            TDS REAL NOT NULL,
            turbidity REAL NOT NULL,
            hardness REAL NOT NULL,
            chloride REAL NOT NULL,
            fluoride REAL NOT NULL,
            nitrate REAL NOT NULL,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id)
        )
        """
    )

    # --------------------------------------------------------
    # Water Availability
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS availability (
            availability_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            supply_hours REAL NOT NULL,
            rainfall REAL NOT NULL,
            season TEXT NOT NULL,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id)
        )
        """
    )

    # --------------------------------------------------------
    # ML Predictions
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            prediction_date TEXT NOT NULL,
            risk_level TEXT,
            availability_forecast REAL,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id)
        )
        """
    )

    # --------------------------------------------------------
    # Alerts
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id)
        )
        """
    )

    # ========================================================
    # WATER BODY REPORTING SYSTEM
    # ========================================================

    # --------------------------------------------------------
    # Water Bodies
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS water_bodies (
            water_body_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            water_body_type TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            description TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # Water Body -> Colony/Society Supply Relationships
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS water_body_supply (
            supply_id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_body_id INTEGER NOT NULL,
            zone_id INTEGER,
            colony_name TEXT NOT NULL,
            society_name TEXT,
            supply_description TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (water_body_id)
                REFERENCES water_bodies(water_body_id)
                ON DELETE CASCADE,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id)
                ON DELETE SET NULL
        )
        """
    )

    # --------------------------------------------------------
    # Water Body Issue Reports
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS water_body_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_body_id INTEGER,
            zone_id INTEGER,
            colony_name TEXT,
            society_name TEXT,
            issue_type TEXT NOT NULL,
            description TEXT NOT NULL,
            photo_path TEXT,
            latitude REAL,
            longitude REAL,
            status TEXT NOT NULL DEFAULT 'Pending',
            severity TEXT NOT NULL DEFAULT 'Medium',
            reporter_user_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (water_body_id)
                REFERENCES water_bodies(water_body_id)
                ON DELETE SET NULL,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id)
                ON DELETE SET NULL,
            FOREIGN KEY (reporter_user_id)
                REFERENCES users(user_id)
                ON DELETE SET NULL
        )
        """
    )

    # ========================================================
    # RESIDENT REGISTRATION + ZONE ALERT SYSTEM
    # ========================================================

    # --------------------------------------------------------
    # Resident Registration
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS residents (
            resident_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            name TEXT,
            phone TEXT,
            email TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id)
        )
        """
    )

    # --------------------------------------------------------
    # Resident Alert Log
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts_log (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            report_id INTEGER,
            resident_id INTEGER,
            message TEXT NOT NULL,
            channel TEXT NOT NULL,
            recipient TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Queued (in-app)',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (zone_id)
                REFERENCES zones(zone_id),
            FOREIGN KEY (resident_id)
                REFERENCES residents(resident_id)
        )
        """
    )

    # --------------------------------------------------------
    # Helpful indexes for reporting/search
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_water_body_supply_water_body
        ON water_body_supply(water_body_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_water_body_supply_zone
        ON water_body_supply(zone_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_water_body_reports_water_body
        ON water_body_reports(water_body_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_water_body_reports_zone
        ON water_body_reports(zone_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_water_body_reports_status
        ON water_body_reports(status)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_water_body_reports_created
        ON water_body_reports(created_at)
        """
    )

    # ========================================================
    # Project-defined Zone Seed Data
    # ========================================================

    # These are the same representative/project-defined zones
    # already used by AquaTrack. Existing rows are preserved.
    #
    # Coordinates are only fallback project markers and are NOT
    # presented as official municipal boundaries.

    zone_data = [
        ("Vijay Nagar", 22.7533, 75.8937),
        ("Palasia", 22.7247, 75.8839),
        ("Bhawarkuan", 22.6818, 75.8577),
        ("Khajrana", 22.7226, 75.9048),
        ("Bengali Square", 22.7167, 75.9000),
        ("Rau", 22.6333, 75.8000),
        ("Rajendra Nagar", 22.6756, 75.8178),
        ("Annapurna", 22.6886, 75.8233),
        ("Sudama Nagar", 22.6942, 75.8350),
        ("Aerodrome", 22.7217, 75.8011),
        ("Kanadia", 22.6869, 75.9192),
        ("Silicon City", 22.6469, 75.8661),
    ]

    for zone_name, latitude, longitude in zone_data:
        cursor.execute(
            """
            SELECT zone_id
            FROM zones
            WHERE zone_name = ?
            LIMIT 1
            """,
            (zone_name,)
        )

        existing = cursor.fetchone()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO zones (
                    zone_name,
                    latitude,
                    longitude
                )
                VALUES (?, ?, ?)
                """,
                (
                    zone_name,
                    latitude,
                    longitude
                )
            )

    connection.commit()
    connection.close()


# ============================================================
# Database Verification
# ============================================================

def verify_database():
    """
    Print a compact verification of AquaTrack database tables
    and existing record counts.
    """

    connection = get_connection()
    cursor = connection.cursor()

    def count_records(table_name):
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        )
        return cursor.fetchone()[0]

    print("AquaTrack Database Verification")
    print("--------------------------------")

    core_tables = [
        "users",
        "zones",
        "water_quality",
        "availability",
        "predictions",
        "alerts",
        "water_bodies",
        "water_body_supply",
        "water_body_reports",
        "residents",
        "alerts_log",
    ]

    for table in core_tables:
        print(
            f"{table}: {count_records(table)}"
        )

    print()
    print("Zones:")

    cursor.execute(
        """
        SELECT
            zone_id,
            zone_name
        FROM zones
        ORDER BY zone_id
        """
    )

    for row in cursor.fetchall():
        print(row)

    connection.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    create_tables()
    verify_database()