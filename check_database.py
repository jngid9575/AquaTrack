import sqlite3


DATABASE_NAME = "aquatrack.db"


connection = sqlite3.connect(DATABASE_NAME)
cursor = connection.cursor()


# Count zones
cursor.execute("SELECT COUNT(*) FROM zones")
zone_count = cursor.fetchone()[0]


# Count water-quality records
cursor.execute("SELECT COUNT(*) FROM water_quality")
quality_count = cursor.fetchone()[0]


# Count availability records
cursor.execute("SELECT COUNT(*) FROM availability")
availability_count = cursor.fetchone()[0]


print("AquaTrack Database Verification")
print("--------------------------------")
print("Zones:", zone_count)
print("Water-quality records:", quality_count)
print("Availability records:", availability_count)


# Show a few zones
print("\nZones:")
cursor.execute("SELECT zone_id, zone_name FROM zones ORDER BY zone_id")

for zone in cursor.fetchall():
    print(zone)


connection.close()