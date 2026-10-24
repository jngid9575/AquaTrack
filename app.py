"""
AquaTrack - Predictive Analytics System for Water Quality Assessment
and Scarcity Forecasting in Indore.

Single-file Streamlit application (decision-support prototype).
Bilingual: English and colloquial Hindi (Hinglish-style, not shuddh/formal Hindi).
Run with:  python -m streamlit run app.py
"""

import io
import re
import smtplib
import hashlib
import secrets
import sqlite3
import logging
import os
import shutil
from contextlib import closing
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
from pathlib import Path
from urllib.parse import urlencode

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from openpyxl import Workbook
from openpyxl.styles import Font
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

from utils.alerts import generate_alert, generate_recommendations
from utils.auth import authenticate_user
from utils.db_functions import (
    get_zones as db_get_zones,
    get_water_quality as db_get_water_quality,
    get_availability as db_get_availability,
)


LOGGER = logging.getLogger("aquatrack")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | AquaTrack | %(message)s")


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_NAME = str(BASE_DIR / "aquatrack.db")
QUALITY_MODEL_PATH = BASE_DIR / "models" / "water_quality_rf.joblib"
AVAILABILITY_MODEL_PATH = BASE_DIR / "models" / "availability_linear_regression.joblib"

QUALITY_FEATURES = ["pH", "TDS", "turbidity", "hardness", "chloride", "fluoride", "nitrate"]
DB_QUALITY_COLS = ["ph", "tds", "turbidity", "hardness", "chloride", "fluoride", "nitrate"]
DB_TO_MODEL = dict(zip(DB_QUALITY_COLS, QUALITY_FEATURES))

SEASON_CODES = {"Summer": 0, "Monsoon": 1, "Winter": 2}
RISK_ORDER = ["Safe", "Moderate Risk", "High Risk"]
STATUS_ORDER = ["Good Availability", "Moderate Availability", "Scarcity Risk"]

RISK_COLORS = {"Safe": "#16a34a", "Moderate Risk": "#f59e0b", "High Risk": "#dc2626"}
STATUS_COLORS = {
    "Good Availability": "#16a34a",
    "Moderate Availability": "#f59e0b",
    "Scarcity Risk": "#dc2626",
}
RISK_EMOJI = {"Safe": "🟢", "Moderate Risk": "🟠", "High Risk": "🔴", "No Data": "⚪"}
STATUS_EMOJI = {
    "Good Availability": "🟢",
    "Moderate Availability": "🟠",
    "Scarcity Risk": "🔴",
    "No Data": "⚪",
}
MARKER_COLORS = {"Safe": "green", "Moderate Risk": "orange", "High Risk": "red"}

PARAM_INFO = {
    "ph": {"label": "pH", "unit": "", "min": 6.5, "max": 8.5},
    "tds": {"label": "TDS", "unit": "mg/L", "max": 500},
    "turbidity": {"label": "Turbidity", "unit": "NTU", "max": 1.0},
    "hardness": {"label": "Hardness", "unit": "mg/L", "max": 200},
    "chloride": {"label": "Chloride", "unit": "mg/L", "max": 250},
    "fluoride": {"label": "Fluoride", "unit": "mg/L", "max": 1.0},
    "nitrate": {"label": "Nitrate", "unit": "mg/L", "max": 45},
}

INPUT_DEFAULTS = {
    "ph": 7.2, "tds": 350.0, "turbidity": 0.8, "hardness": 180.0,
    "chloride": 200.0, "fluoride": 0.7, "nitrate": 25.0,
}

DISCLAIMER = (
    "AquaTrack is a project-level decision-support prototype that uses "
    "project-generated synthetic data. It performs predictive screening only "
    "and does not replace laboratory testing or official municipal systems."
)


# ============================================================
# LOW-LEVEL DATABASE HELPERS
# ============================================================

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def db_query(sql, params=()):
    with closing(get_db_connection()) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def db_execute(sql, params=()):
    with closing(get_db_connection()) as connection:
        with connection:
            cursor = connection.execute(sql, params)
            return cursor.lastrowid


# ============================================================
# RESIDENT REGISTRATION + ZONE ALERT ENGINE
# ============================================================


WORKFLOW_STATUSES = [
    "Submitted", "Assigned", "In Progress",
    "Waiting for Verification", "Resolved", "Reopened",
]
WORKER_STATUSES = ["Available", "Busy", "Offline"]


def _validate_identifier(name: str) -> str:
    if not isinstance(name, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


def _column_names(table_name):
    try:
        table_name = _validate_identifier(table_name)
        return {row[1] for row in db_query(f"PRAGMA table_info({table_name})").itertuples(index=False, name=None)}
    except Exception:
        return set()


def _ensure_column(table_name, column_name, column_definition):
    table_name = _validate_identifier(table_name)
    column_name = _validate_identifier(column_name)
    columns = _column_names(table_name)
    if column_name not in columns:
        db_execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def hash_worker_password(password, salt=None):
    salt_bytes = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt_bytes, 180_000)
    return f"pbkdf2_sha256$180000${salt_bytes.hex()}${digest.hex()}"


def verify_worker_password(password, stored):
    try:
        scheme, rounds, salt_hex, digest_hex = str(stored).split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        calculated = hashlib.pbkdf2_hmac(
            "sha256", str(password).encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        ).hex()
        return secrets.compare_digest(calculated, digest_hex)
    except Exception:
        return False


def ensure_workflow_tables():
    """Add the citizen -> admin -> worker -> verification -> review workflow without destroying legacy data."""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aquatrack_workers (
                worker_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                phone TEXT,
                zone_id INTEGER,
                availability_status TEXT NOT NULL DEFAULT 'Available',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_status_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                old_status TEXT,
                new_status TEXT NOT NULL,
                changed_by_role TEXT NOT NULL,
                changed_by_id INTEGER,
                remarks TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (report_id) REFERENCES water_body_reports(report_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citizen_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                resident_id INTEGER,
                worker_id INTEGER,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                resolved INTEGER NOT NULL DEFAULT 1,
                comment TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(report_id, resident_id),
                FOREIGN KEY (report_id) REFERENCES water_body_reports(report_id),
                FOREIGN KEY (resident_id) REFERENCES residents(resident_id),
                FOREIGN KEY (worker_id) REFERENCES aquatrack_workers(worker_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aquatrack_notifications (
                notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient_role TEXT NOT NULL,
                recipient_id INTEGER,
                report_id INTEGER,
                message TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (report_id) REFERENCES water_body_reports(report_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aquatrack_audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                actor_id INTEGER,
                report_id INTEGER,
                details TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()

        # Migrate legacy citizen_reviews tables that enforced one review per report.
        # The intended rule is one review per resident per report.
        unique_report_only = False
        for index_row in cursor.execute("PRAGMA index_list(citizen_reviews)").fetchall():
            # PRAGMA index_list: seq, name, unique, origin, partial
            if int(index_row[2] or 0) != 1:
                continue
            index_name = str(index_row[1])
            columns = [
                str(row[2])
                for row in cursor.execute(f'PRAGMA index_info("{index_name}")').fetchall()
            ]
            if columns == ["report_id"]:
                unique_report_only = True
                break

        if unique_report_only:
            cursor.execute("PRAGMA foreign_keys=OFF")
            cursor.execute("ALTER TABLE citizen_reviews RENAME TO citizen_reviews_legacy")
            cursor.execute("""
                CREATE TABLE citizen_reviews (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    resident_id INTEGER,
                    worker_id INTEGER,
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    resolved INTEGER NOT NULL DEFAULT 1,
                    comment TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(report_id, resident_id),
                    FOREIGN KEY (report_id) REFERENCES water_body_reports(report_id),
                    FOREIGN KEY (resident_id) REFERENCES residents(resident_id),
                    FOREIGN KEY (worker_id) REFERENCES aquatrack_workers(worker_id)
                )
            """)
            cursor.execute("""
                INSERT INTO citizen_reviews
                    (review_id, report_id, resident_id, worker_id, rating, resolved, comment, created_at)
                SELECT review_id, report_id, resident_id, worker_id, rating, resolved, comment, created_at
                FROM citizen_reviews_legacy
            """)
            cursor.execute("DROP TABLE citizen_reviews_legacy")
            cursor.execute("PRAGMA foreign_keys=ON")
            connection.commit()

        # Extend the existing report table instead of replacing it.
        # IMPORTANT: these columns must exist BEFORE any query references them.
        # Older aquatrack.db files may predate the workflow_status column.
        for name, definition in [
            ("workflow_status", "TEXT DEFAULT 'Submitted'"),
            ("resident_id", "INTEGER"),
            ("assigned_worker_id", "INTEGER"),
            ("assigned_at", "TEXT"),
            ("started_at", "TEXT"),
            ("worker_completed_at", "TEXT"),
            ("admin_verified_at", "TEXT"),
            ("resolved_at", "TEXT"),
            ("reopened_at", "TEXT"),
            ("worker_remarks", "TEXT"),
            ("admin_remarks", "TEXT"),
            ("completion_evidence_path", "TEXT"),
        ]:
            _ensure_column("water_body_reports", name, definition)

        # Old builds could contain the now-unused Worker Completed state.
        # It represents the same point as Waiting for Verification, so normalize it.
        cursor.execute(
            "UPDATE water_body_reports "
            "SET workflow_status='Waiting for Verification' "
            "WHERE workflow_status='Worker Completed'"
        )

        # Admin notifications are a shared role-level inbox; do not bind them to admin #1.
        cursor.execute(
            "UPDATE aquatrack_notifications SET recipient_id=NULL "
            "WHERE recipient_role='admin'"
        )
        connection.commit()

        # Legacy rows are treated as submitted/resolved workflow rows.
        db_execute(
            "UPDATE water_body_reports SET workflow_status = "
            "CASE WHEN status = 'Resolved' THEN 'Resolved' "
            "ELSE COALESCE(workflow_status, 'Submitted') END "
            "WHERE workflow_status IS NULL OR TRIM(workflow_status) = ''"
        )
        connection.commit()
    except Exception:
        try:
            connection.rollback()
        except sqlite3.ProgrammingError:
            pass
        raise
    finally:
        connection.close()


def get_workers(active_only=False):
    sql = """
        SELECT w.worker_id, w.username, w.name, w.phone, w.zone_id,
               z.zone_name, w.availability_status, w.active, w.created_at,
               (SELECT COUNT(*) FROM water_body_reports r
                WHERE r.assigned_worker_id = w.worker_id
                  AND r.workflow_status IN ('Assigned','In Progress','Waiting for Verification')) AS active_jobs
        FROM aquatrack_workers w
        LEFT JOIN zones z ON w.zone_id = z.zone_id
    """
    if active_only:
        sql += " WHERE w.active = 1"
    sql += " ORDER BY w.name"
    return db_query(sql)


def authenticate_worker(username, password):
    row = db_query("SELECT worker_id, username, password_hash, name, phone, zone_id, availability_status, active FROM aquatrack_workers WHERE username = ? LIMIT 1", (str(username).strip(),))
    if row.empty:
        return None
    worker = row.iloc[0]
    if not int(worker["active"] or 0) or not verify_worker_password(password, worker["password_hash"]):
        return None
    return worker.to_dict()


def set_worker_availability(worker_id, status, actor_id=None):
    status = str(status).strip()
    if status not in WORKER_STATUSES:
        return False, "Invalid worker availability status."
    if status == "Available":
        active_jobs = db_query("SELECT COUNT(*) AS c FROM water_body_reports WHERE assigned_worker_id=? AND workflow_status IN ('Assigned','In Progress')", (int(worker_id),))
        if not active_jobs.empty and int(active_jobs.iloc[0]["c"]) > 0:
            return False, "A worker with active jobs cannot be marked available manually."
    db_execute("UPDATE aquatrack_workers SET availability_status=? WHERE worker_id=?", (status, int(worker_id)))
    audit_event("worker_availability_changed", "admin", actor_id, None, f"worker_id={int(worker_id)}; status={status}")
    return True, "Worker availability updated."


def update_worker_account(worker_id, username, name, phone, zone_id=None, password=None, actor_id=None):
    worker_id = int(worker_id)
    username = str(username or "").strip()
    name = str(name or "").strip()
    phone = str(phone or "").strip()
    if not username or not name:
        return False, "Worker name and username are required."
    existing = db_query("SELECT worker_id FROM aquatrack_workers WHERE username=? AND worker_id<>? LIMIT 1", (username, worker_id))
    if not existing.empty:
        return False, "A worker with this username already exists."
    if password:
        db_execute(
            "UPDATE aquatrack_workers SET username=?, name=?, phone=?, zone_id=?, password_hash=? WHERE worker_id=?",
            (username, name, phone, None if zone_id is None else int(zone_id), hash_worker_password(password), worker_id),
        )
    else:
        db_execute(
            "UPDATE aquatrack_workers SET username=?, name=?, phone=?, zone_id=? WHERE worker_id=?",
            (username, name, phone, None if zone_id is None else int(zone_id), worker_id),
        )
    audit_event("worker_updated", "admin", actor_id, None, f"worker_id={worker_id}; username={username}")
    return True, f"Worker #{worker_id} updated successfully."


def remove_worker_account(worker_id, actor_id=None):
    worker_id = int(worker_id)
    worker = db_query("SELECT name, active FROM aquatrack_workers WHERE worker_id=? LIMIT 1", (worker_id,))
    if worker.empty:
        return False, "Worker not found."
    active_jobs = db_query(
        "SELECT COUNT(*) AS c FROM water_body_reports WHERE assigned_worker_id=? AND workflow_status IN ('Assigned','In Progress','Waiting for Verification')",
        (worker_id,),
    )
    if int(active_jobs.iloc[0]["c"] or 0) > 0:
        return False, "This worker has active requests. Reassign or complete them before removing the worker."
    db_execute("UPDATE aquatrack_workers SET active=0, availability_status='Offline' WHERE worker_id=?", (worker_id,))
    audit_event("worker_removed", "admin", actor_id, None, f"worker_id={worker_id}; name={worker.iloc[0]['name']}")
    return True, f"Worker {worker.iloc[0]['name']} was removed from active workers."


def create_worker_account(username, password, name, phone, zone_id=None):
    username = str(username or "").strip().lower()
    name = str(name or "").strip()
    if len(username) < 3 or len(password or "") < 6 or not name:
        return False, "Username, name and a password of at least 6 characters are required."
    existing = db_query("SELECT worker_id FROM aquatrack_workers WHERE username = ?", (username,))
    if not existing.empty:
        return False, "A worker with this username already exists."
    worker_id = db_execute(
        "INSERT INTO aquatrack_workers (username,password_hash,name,phone,zone_id,availability_status) VALUES (?,?,?,?,?, 'Available')",
        (username, hash_worker_password(password), name, str(phone or "").strip(), None if zone_id is None else int(zone_id)),
    )
    return bool(worker_id), f"Worker #{worker_id} created successfully."


def add_notification(role, recipient_id, report_id, message):
    db_execute(
        "INSERT INTO aquatrack_notifications (recipient_role, recipient_id, report_id, message) VALUES (?,?,?,?)",
        (role, None if recipient_id is None else int(recipient_id), None if report_id is None else int(report_id), str(message)),
    )


def get_notifications(role, recipient_id, limit=30):
    if str(role) == "admin" and recipient_id is None:
        return db_query("""
            SELECT notification_id, report_id, message, is_read, created_at
            FROM aquatrack_notifications
            WHERE recipient_role = ?
            ORDER BY notification_id DESC LIMIT ?
        """, (str(role), int(limit)))
    return db_query("""
        SELECT notification_id, report_id, message, is_read, created_at
        FROM aquatrack_notifications
        WHERE recipient_role = ? AND recipient_id = ?
        ORDER BY notification_id DESC LIMIT ?
    """, (str(role), int(recipient_id), int(limit)))


def mark_notifications_read(role, recipient_id):
    with closing(get_db_connection()) as connection:
        if str(role) == "admin" and recipient_id is None:
            cursor = connection.execute(
                """
                UPDATE aquatrack_notifications
                SET is_read = 1
                WHERE recipient_role = ?
                  AND COALESCE(is_read, 0) = 0
                """,
                (str(role),),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE aquatrack_notifications
                SET is_read = 1
                WHERE recipient_role = ?
                  AND recipient_id = ?
                  AND COALESCE(is_read, 0) = 0
                """,
                (str(role), int(recipient_id)),
            )
        connection.commit()
        return cursor.rowcount


def get_service_requests(status=None, worker_id=None, resident_id=None):
    sql = """
        SELECT r.*,
               wb.name AS water_body,
               wb.location AS water_body_location,
               wb.latitude AS water_body_latitude,
               wb.longitude AS water_body_longitude,
               z.zone_name,
               w.name AS worker_name, w.username AS worker_username,
               rv.rating AS review_rating, rv.comment AS review_comment
        FROM water_body_reports r
        LEFT JOIN water_bodies wb ON r.water_body_id = wb.water_body_id
        LEFT JOIN zones z ON r.zone_id = z.zone_id
        LEFT JOIN aquatrack_workers w ON r.assigned_worker_id = w.worker_id
        LEFT JOIN citizen_reviews rv ON rv.review_id = (SELECT MAX(cr.review_id) FROM citizen_reviews cr WHERE cr.report_id = r.report_id)
        WHERE 1=1
    """
    params=[]
    if status:
        sql += " AND r.workflow_status = ?"
        params.append(status)
    if worker_id is not None:
        sql += " AND r.assigned_worker_id = ?"
        params.append(int(worker_id))
    if resident_id is not None:
        sql += " AND r.resident_id = ?"
        params.append(int(resident_id))
    sql += " ORDER BY r.report_id DESC"
    return db_query(sql, tuple(params))


STATUS_TRANSITIONS = {
    "Submitted": {"Assigned", "Reopened"},
    "Assigned": {"In Progress", "Reopened"},
    "In Progress": {"Waiting for Verification", "Reopened"},
    "Waiting for Verification": {"Resolved", "Reopened"},
    "Resolved": {"Reopened"},
    "Reopened": {"Assigned", "In Progress"},
}


def audit_event(action, actor_role, actor_id=None, report_id=None, details=""):
    try:
        db_execute(
            "INSERT INTO aquatrack_audit_log (action, actor_role, actor_id, report_id, details) VALUES (?,?,?,?,?)",
            (str(action), str(actor_role), None if actor_id is None else int(actor_id),
             None if report_id is None else int(report_id), str(details or "")),
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Audit event skipped: %s", error)


def sync_worker_availability(worker_id):
    if worker_id is None:
        return
    row = db_query(
        "SELECT active, availability_status FROM aquatrack_workers WHERE worker_id=?",
        (int(worker_id),),
    )
    if row.empty or int(row.iloc[0]["active"] or 0) != 1:
        return
    active_jobs = db_query(
        """SELECT COUNT(*) AS c FROM water_body_reports
           WHERE assigned_worker_id=?
             AND workflow_status IN ('Assigned','In Progress')""",
        (int(worker_id),),
    )
    count = int(active_jobs.iloc[0]["c"]) if not active_jobs.empty else 0
    current = str(row.iloc[0]["availability_status"] or "Available")
    if current == "Offline":
        return
    new_status = "Busy" if count > 0 else "Available"
    if new_status != current:
        db_execute("UPDATE aquatrack_workers SET availability_status=? WHERE worker_id=?", (new_status, int(worker_id)))


def validate_status_transition(old_status, new_status, role):
    old_status = str(old_status or "Submitted")
    new_status = str(new_status or "").strip()
    if new_status not in WORKFLOW_STATUSES:
        return False, "Invalid workflow status."
    if new_status == old_status:
        return True, "No status change."
    if new_status not in STATUS_TRANSITIONS.get(old_status, set()):
        return False, f"Invalid workflow transition: {old_status} → {new_status}."
    if role == "worker" and new_status not in {"In Progress", "Waiting for Verification"}:
        return False, "Workers can only start work or submit completion for verification."
    if role == "admin" and new_status == "Waiting for Verification":
        return False, "Only the assigned worker can submit work for verification."
    return True, ""


ALLOWED_REPORT_FIELDS = {
    "workflow_status", "assigned_at", "started_at", "worker_completed_at",
    "resolved_at", "admin_verified_at", "reopened_at", "worker_remarks",
    "admin_remarks", "completion_evidence_path",
}


def change_request_status(report_id, new_status, role, actor_id=None, remarks=""):
    current = db_query(
        "SELECT workflow_status, assigned_worker_id, resident_id FROM water_body_reports WHERE report_id = ?",
        (int(report_id),),
    )
    if current.empty:
        return False, "Request not found."
    row = current.iloc[0]
    old_status = str(row.get("workflow_status") or "Submitted")
    allowed, reason = validate_status_transition(old_status, new_status, role)
    if not allowed:
        return False, reason
    if new_status == old_status:
        return True, "Status is already set to that value."
    if role == "worker":
        if actor_id is None or pd.isna(row.get("assigned_worker_id")) or int(row["assigned_worker_id"]) != int(actor_id):
            return False, "This request is not assigned to the current worker."
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    fields = {"workflow_status": new_status}
    if new_status == "Assigned": fields["assigned_at"] = now
    if new_status == "In Progress": fields["started_at"] = now
    if new_status == "Waiting for Verification": fields["worker_completed_at"] = now
    if new_status == "Resolved": fields["resolved_at"] = now; fields["admin_verified_at"] = now
    if new_status == "Reopened": fields["reopened_at"] = now
    if role == "worker" and remarks:
        fields["worker_remarks"] = str(remarks).strip()
    if role == "admin" and remarks:
        fields["admin_remarks"] = str(remarks).strip()
    invalid_fields = set(fields) - ALLOWED_REPORT_FIELDS
    if invalid_fields:
        raise ValueError(f"Invalid report fields: {sorted(invalid_fields)}")
    assignments = ", ".join(f"{_validate_identifier(k)} = ?" for k in fields)
    db_execute(
        f"UPDATE water_body_reports SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE report_id = ?",
        tuple(fields.values()) + (int(report_id),),
    )
    db_execute(
        "INSERT INTO request_status_history (report_id, old_status, new_status, changed_by_role, changed_by_id, remarks) VALUES (?,?,?,?,?,?)",
        (int(report_id), old_status, new_status, role, actor_id, remarks),
    )
    clear_data_caches()
    audit_event("status_changed", role, actor_id, report_id, f"{old_status} -> {new_status}; {remarks}")
    resident_id = row.get("resident_id")
    worker_id = row.get("assigned_worker_id")
    if resident_id is not None and not pd.isna(resident_id):
        add_notification("resident", int(resident_id), int(report_id), f"Request #{report_id} status changed to {new_status}.")
        if new_status == "Resolved":
            add_notification("resident", int(resident_id), int(report_id), f"Request #{report_id} has been resolved. You can now share your experience and leave a review.")
    if worker_id is not None and not pd.isna(worker_id):
        add_notification("worker", int(worker_id), int(report_id), f"Request #{report_id} status changed to {new_status}.")
        sync_worker_availability(int(worker_id))
    return True, "Status updated."


def assign_worker(report_id, worker_id, actor_id=None):
    worker = db_query(
        "SELECT worker_id, name, availability_status, active FROM aquatrack_workers WHERE worker_id = ?",
        (int(worker_id),),
    )
    if worker.empty or int(worker.iloc[0]["active"] or 0) != 1:
        return False, "Worker not found or inactive."
    if str(worker.iloc[0]["availability_status"]) == "Offline":
        return False, "This worker is currently offline."
    current = db_query(
        "SELECT workflow_status, resident_id, assigned_worker_id FROM water_body_reports WHERE report_id = ?",
        (int(report_id),),
    )
    if current.empty:
        return False, "Request not found."
    old = str(current.iloc[0]["workflow_status"] or "Submitted")
    if old not in {"Submitted", "Reopened", "Assigned"}:
        return False, f"Request cannot be assigned from {old}."
    old_worker = current.iloc[0]["assigned_worker_id"]
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    db_execute(
        "UPDATE water_body_reports SET assigned_worker_id=?, assigned_at=?, workflow_status='Assigned', updated_at=CURRENT_TIMESTAMP WHERE report_id=?",
        (int(worker_id), now, int(report_id)),
    )
    if old_worker is not None and not pd.isna(old_worker) and int(old_worker) != int(worker_id):
        sync_worker_availability(int(old_worker))
    sync_worker_availability(int(worker_id))
    db_execute(
        "INSERT INTO request_status_history (report_id, old_status, new_status, changed_by_role, changed_by_id, remarks) VALUES (?,?,?,?,?,?)",
        (int(report_id), old, "Assigned", "admin", actor_id, f"Assigned to {worker.iloc[0]['name']}"),
    )
    clear_data_caches()
    audit_event("worker_assigned", "admin", actor_id, report_id, f"Worker {worker.iloc[0]['name']} ({int(worker_id)}) assigned")
    resident_id = current.iloc[0]["resident_id"]
    if resident_id is not None and not pd.isna(resident_id):
        add_notification("resident", int(resident_id), int(report_id), f"Request #{report_id} has been assigned to a field worker.")
    add_notification("worker", int(worker_id), int(report_id), f"You have been assigned request #{report_id}.")
    return True, f"Request #{report_id} assigned successfully."


def complete_worker_request(report_id, worker_id, remarks, evidence_path=None):
    row = db_query(
        "SELECT assigned_worker_id, workflow_status, resident_id FROM water_body_reports WHERE report_id=?",
        (int(report_id),),
    )
    if row.empty or int(row.iloc[0]["assigned_worker_id"] or -1) != int(worker_id):
        return False, "This request is not assigned to this worker."
    if str(row.iloc[0]["workflow_status"]) != "In Progress":
        return False, "Start the request before submitting completion."
    if not str(remarks or "").strip():
        return False, "Work remarks are required before completion."
    db_execute(
        "UPDATE water_body_reports SET workflow_status='Waiting for Verification', worker_completed_at=CURRENT_TIMESTAMP, worker_remarks=?, completion_evidence_path=?, updated_at=CURRENT_TIMESTAMP WHERE report_id=?",
        (str(remarks).strip(), evidence_path, int(report_id)),
    )
    db_execute(
        "INSERT INTO request_status_history (report_id, old_status, new_status, changed_by_role, changed_by_id, remarks) VALUES (?,?,?,?,?,?)",
        (int(report_id), str(row.iloc[0]["workflow_status"]), "Waiting for Verification", "worker", int(worker_id), remarks),
    )
    clear_data_caches()
    audit_event("work_completed", "worker", worker_id, report_id, str(remarks).strip())
    sync_worker_availability(int(worker_id))
    add_notification("admin", None, int(report_id), f"Worker completed request #{report_id}; verification is required.")
    resident_id = row.iloc[0]["resident_id"]
    if resident_id is not None and not pd.isna(resident_id):
        add_notification("resident", int(resident_id), int(report_id), f"Request #{report_id} has been completed by the field worker and is awaiting verification.")
    return True, "Completion submitted for admin verification."


def get_reviews():
    return db_query("""
        SELECT rv.review_id, rv.report_id, rv.rating, rv.resolved, rv.comment, rv.created_at,
               z.zone_name, w.name AS worker_name, r.issue_type
        FROM citizen_reviews rv
        JOIN water_body_reports r ON rv.report_id = r.report_id
        LEFT JOIN zones z ON r.zone_id = z.zone_id
        LEFT JOIN aquatrack_workers w ON rv.worker_id = w.worker_id
        ORDER BY rv.review_id DESC
    """)


def submit_review(report_id,resident_id,rating,comment):
    req=db_query(
        "SELECT report_id,assigned_worker_id,workflow_status FROM water_body_reports WHERE report_id=? AND resident_id=?",
        (int(report_id),int(resident_id)),
    )
    if req.empty:
        return False,"You can only review your own requests."
    if str(req.iloc[0]["workflow_status"])!="Resolved":
        return False,"This request is not ready for review."
    if not db_query(
        "SELECT review_id FROM citizen_reviews WHERE report_id=? AND resident_id=? LIMIT 1",
        (int(report_id),int(resident_id)),
    ).empty:
        return False,"You have already reviewed this request."
    rating=int(rating); comment=str(comment or '').strip()
    if not 1<=rating<=5: return False,"Rating must be between 1 and 5."
    if len(comment)>1000: return False,"Review comment is too long (maximum 1000 characters)."
    worker_id=req.iloc[0]['assigned_worker_id']
    try:
        db_execute(
            "INSERT INTO citizen_reviews (report_id,resident_id,worker_id,rating,resolved,comment) VALUES (?,?,?,?,?,?)",
            (int(report_id),int(resident_id),None if pd.isna(worker_id) else int(worker_id),rating,1,comment),
        )
    except sqlite3.IntegrityError:
        return False,"You have already reviewed this request."
    clear_data_caches()
    audit_event('review_submitted','resident',resident_id,report_id,f'rating={rating}')
    add_notification('admin',None,int(report_id),f'Citizen submitted a {rating}/5 review for request #{report_id}.')
    return True,"Thank you. Your review has been recorded."


def ensure_workflow_seed_admin_notifications():
    # Admin notification recipient id is a logical single admin inbox; no secret/account data is stored here.
    return None


def ensure_alert_tables():
    """Create/migrate resident-alert tables without changing existing AquaTrack tables."""
    with closing(get_db_connection()) as connection:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS residents (
                resident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id INTEGER NOT NULL,
                name TEXT,
                phone TEXT,
                email TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
            )
        """)

        resident_columns = {row[1] for row in cursor.execute("PRAGMA table_info(residents)").fetchall()}
        if "username" not in resident_columns: cursor.execute("ALTER TABLE residents ADD COLUMN username TEXT")
        if "password_hash" not in resident_columns: cursor.execute("ALTER TABLE residents ADD COLUMN password_hash TEXT")
        if "active" not in resident_columns: cursor.execute("ALTER TABLE residents ADD COLUMN active INTEGER NOT NULL DEFAULT 1")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citizen_password_resets (
                reset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                resident_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (resident_id) REFERENCES residents(resident_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citizen_password_resets_token ON citizen_password_resets(token_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citizen_password_resets_resident ON citizen_password_resets(resident_id)")

        cursor.execute("""
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
                FOREIGN KEY (zone_id) REFERENCES zones(zone_id),
                FOREIGN KEY (resident_id) REFERENCES residents(resident_id)
            )
        """)

        columns = {row[1] for row in cursor.execute("PRAGMA table_info(alerts_log)").fetchall()}
        if "resident_id" not in columns:
            cursor.execute("ALTER TABLE alerts_log ADD COLUMN resident_id INTEGER")

        connection.commit()


def register_resident(zone_id, name, phone, email):
    zone_id = int(zone_id)
    name = (name or "").strip()
    phone = (phone or "").strip()
    email = (email or "").strip().lower()

    existing = db_query(
        """
        SELECT resident_id
        FROM residents
        WHERE zone_id = ?
          AND (
              (? <> '' AND phone = ?)
              OR (? <> '' AND email = ?)
          )
        LIMIT 1
        """,
        (zone_id, phone, phone, email, email),
    )

    if not existing.empty:
        resident_id = int(existing.iloc[0]["resident_id"])
        db_execute(
            """
            UPDATE residents
            SET name = ?, phone = ?, email = ?
            WHERE resident_id = ?
            """,
            (name, phone, email, resident_id),
        )
        return resident_id, False

    resident_id = db_execute(
        """
        INSERT INTO residents (zone_id, name, phone, email)
        VALUES (?, ?, ?, ?)
        """,
        (zone_id, name, phone, email),
    )
    return resident_id, True


def get_zone_residents(zone_id):
    return db_query(
        """
        SELECT resident_id, zone_id, name, phone, email, username, active, created_at
        FROM residents
        WHERE zone_id = ?
        ORDER BY resident_id DESC
        """,
        (int(zone_id),),
    )


def get_resident(resident_id):
    return db_query(
        """
        SELECT resident_id, zone_id, name, phone, email, created_at
        FROM residents
        WHERE resident_id = ?
        """,
        (int(resident_id),),
    )


def authenticate_resident(identifier, password):
    identifier = str(identifier or "").strip().lower()
    row = db_query("SELECT resident_id, username, password_hash, name, phone, email, zone_id, active FROM residents WHERE username=? OR email=? LIMIT 1", (identifier, identifier))
    if row.empty: return None
    resident=row.iloc[0]
    if int(resident.get("active",1) or 0)!=1 or not verify_worker_password(password, str(resident.get("password_hash") or "")): return None
    return resident.to_dict()

def register_citizen_account(username,password,name,phone,email,zone_id):
    username=str(username or "").strip().lower(); name=str(name or "").strip(); phone=str(phone or "").strip(); email=str(email or "").strip().lower()
    if len(username)<4: return False,"Username must be at least 4 characters."
    if len(password)<6: return False,"Password must be at least 6 characters."
    if not name: return False,"Name is required."
    if not db_query("SELECT resident_id FROM residents WHERE username=? LIMIT 1",(username,)).empty: return False,"That username is already in use."
    ph=hash_worker_password(password); existing=db_query("SELECT resident_id FROM residents WHERE (?<>'' AND phone=?) OR (?<>'' AND email=?) LIMIT 1",(phone,phone,email,email))
    if not existing.empty:
        rid=int(existing.iloc[0]["resident_id"]); db_execute("UPDATE residents SET username=?,password_hash=?,name=?,phone=?,email=?,zone_id=?,active=1 WHERE resident_id=?",(username,ph,name,phone,email,int(zone_id),rid)); return True,rid
    rid=db_execute("INSERT INTO residents (zone_id,name,phone,email,username,password_hash,active) VALUES (?,?,?,?,?,?,1)",(int(zone_id),name,phone,email,username,ph)); return True,rid

def update_citizen_profile(resident_id,name,phone,email,zone_id):
    db_execute("UPDATE residents SET name=?,phone=?,email=?,zone_id=? WHERE resident_id=?",(str(name or '').strip(),str(phone or '').strip(),str(email or '').strip().lower(),int(zone_id),int(resident_id)))

def _reset_token_hash(token):
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()

def _get_app_base_url():
    try:
        configured_url = st.secrets.get("app", {}).get("base_url") or os.getenv("AQUATRACK_BASE_URL")
        if configured_url:
            return str(configured_url).rstrip("/")
        headers = st.context.headers
        proto = str(headers.get("X-Forwarded-Proto") or headers.get("x-forwarded-proto") or "http").split(",")[0].strip()
        host = str(headers.get("X-Forwarded-Host") or headers.get("x-forwarded-host") or headers.get("Host") or "localhost:8501").split(",")[0].strip()
        return f"{proto}://{host}"
    except Exception:
        return "http://localhost:8501"

def create_citizen_password_reset(email):
    email = str(email or "").strip().lower()
    if "@" not in email:
        return False, "Please enter a valid email address."
    resident = db_query("SELECT resident_id, name, email, active FROM residents WHERE lower(email)=? LIMIT 1", (email,))
    # Do not reveal whether an email exists.
    generic = "If an account exists for that email, a password reset link has been sent."
    if resident.empty or int(resident.iloc[0].get("active", 1) or 0) != 1:
        return True, generic

    resident_id = int(resident.iloc[0]["resident_id"])
    token = secrets.token_urlsafe(32)
    token_hash = _reset_token_hash(token)
    db_execute("DELETE FROM citizen_password_resets WHERE resident_id=? OR expires_at < CURRENT_TIMESTAMP", (resident_id,))
    db_execute("INSERT INTO citizen_password_resets (resident_id,token_hash,expires_at) VALUES (?,?,datetime('now','+30 minutes'))", (resident_id, token_hash))

    reset_url = f"{_get_app_base_url()}/?reset_token={token}"
    name = str(resident.iloc[0].get("name") or "Citizen").strip() or "Citizen"
    body = (
        f"Hello {name},\n\n"
        "We received a request to reset your AquaTrack citizen account password.\n\n"
        f"Reset your password using this link (valid for 30 minutes):\n{reset_url}\n\n"
        "If you did not request this, you can safely ignore this email.\n\n"
        "AquaTrack"
    )
    delivered, message = send_email_alert(email, "AquaTrack — Reset Your Password", body)
    if not delivered:
        db_execute("DELETE FROM citizen_password_resets WHERE token_hash=?", (token_hash,))
        return False, f"We could not send the reset email. {message}"
    return True, generic

def get_password_reset(token):
    token = str(token or "").strip()
    if not token:
        return None
    row = db_query("""
        SELECT pr.reset_id, pr.resident_id, r.email, r.name
        FROM citizen_password_resets pr
        JOIN residents r ON r.resident_id = pr.resident_id
        WHERE pr.token_hash=? AND pr.used_at IS NULL AND pr.expires_at > CURRENT_TIMESTAMP AND r.active=1
        LIMIT 1
    """, (_reset_token_hash(token),))
    return None if row.empty else row.iloc[0].to_dict()

def complete_citizen_password_reset(token, new_password):
    record = get_password_reset(token)
    if not record:
        return False, "This password reset link is invalid or has expired."
    new_password = str(new_password or "")
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    db_execute("UPDATE residents SET password_hash=? WHERE resident_id=?", (hash_worker_password(new_password), int(record["resident_id"])))
    db_execute("UPDATE citizen_password_resets SET used_at=CURRENT_TIMESTAMP WHERE reset_id=?", (int(record["reset_id"]),))
    return True, "Password updated successfully. You can now log in."

def create_citizen_report(resident_id,zone_id,issue_type,description,severity,water_body_id=None,colony_name='',society_name='',latitude=None,longitude=None,photo_path=None):
    rid=db_execute("""INSERT INTO water_body_reports (water_body_id,zone_id,colony_name,society_name,issue_type,description,photo_path,latitude,longitude,status,severity,reporter_user_id,workflow_status,resident_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",(None if water_body_id is None else int(water_body_id),int(zone_id),str(colony_name or ''),str(society_name or ''),str(issue_type),str(description),photo_path,latitude,longitude,'Pending',str(severity),None,'Submitted',int(resident_id)))
    add_notification('admin',None,int(rid),f'New citizen report #{rid} has been submitted and requires review.')
    add_notification('resident',int(resident_id),int(rid),f'Your report #{rid} has been submitted successfully.')
    clear_data_caches()
    audit_event('report_submitted','resident',resident_id,rid,f'issue_type={issue_type}; severity={severity}')
    return int(rid)

def get_smtp_config():
    """Read and validate AquaTrack SMTP settings from Streamlit secrets.

    Expected secrets.toml section:
        [smtp]
        host = "smtp.gmail.com"
        port = 587
        username = "your-email@gmail.com"
        password = "GOOGLE_APP_PASSWORD"
        sender_name = "AquaTrack Support"
        use_tls = true

    The Gmail password itself must NOT be used here. For Gmail, use a
    Google App Password with 2-Step Verification enabled.
    """
    try:
        smtp_cfg = st.secrets.get("smtp")
    except Exception as error:  # noqa: BLE001
        return None, f"Unable to read Streamlit SMTP secrets: {error}"

    if not smtp_cfg:
        return None, "SMTP is not configured. Add [smtp] to .streamlit/secrets.toml."

    required = ("host", "port", "username", "password")
    missing = [key for key in required if not str(smtp_cfg.get(key, "")).strip()]
    if missing:
        return None, "SMTP configuration is incomplete. Missing: " + ", ".join(missing)

    try:
        port = int(smtp_cfg.get("port", 587))
    except (TypeError, ValueError):
        return None, "SMTP port must be a number, normally 587 for Gmail STARTTLS."

    if not 1 <= port <= 65535:
        return None, "SMTP port must be between 1 and 65535."

    host = str(smtp_cfg["host"]).strip()
    username = str(smtp_cfg["username"]).strip()
    password = str(smtp_cfg["password"]).strip()
    sender_name = str(smtp_cfg.get("sender_name", "AquaTrack Support")).strip() or "AquaTrack Support"
    raw_tls = smtp_cfg.get("use_tls", True)
    if isinstance(raw_tls, str):
        use_tls = raw_tls.strip().lower() not in {"false", "0", "no", "off"}
    else:
        use_tls = bool(raw_tls)

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "sender_name": sender_name,
        "use_tls": use_tls,
    }, "SMTP configuration loaded"


def smtp_status():
    """Return safe SMTP status information without exposing the password."""
    cfg, message = get_smtp_config()
    if not cfg:
        return {
            "configured": False,
            "message": message,
            "host": "-",
            "port": "-",
            "username": "-",
            "sender_name": "AquaTrack Support",
        }
    return {
        "configured": True,
        "message": "SMTP configuration is ready",
        "host": cfg["host"],
        "port": cfg["port"],
        "username": cfg["username"],
        "sender_name": cfg["sender_name"],
    }


def send_email_alert(to_email, subject, body):
    """Send a real email through the configured SMTP server.

    Returns (True, message) on successful SMTP delivery and (False, reason)
    when configuration or delivery fails. The password is never included in
    the returned error text shown by AquaTrack.
    """
    to_email = str(to_email or "").strip()
    if not to_email:
        return False, "No email address"

    if "@" not in to_email or "." not in to_email.rsplit("@", 1)[-1]:
        return False, "Invalid recipient email address"

    cfg, config_message = get_smtp_config()
    if not cfg:
        return False, config_message

    try:
        msg = MIMEText(str(body), "plain", "utf-8")
        msg["Subject"] = str(subject).strip() or "AquaTrack Notification"
        msg["From"] = formataddr((cfg["sender_name"], cfg["username"]))
        msg["To"] = to_email
        msg["Date"] = formatdate(localtime=True)
        msg["Reply-To"] = cfg["username"]

        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as server:
            server.ehlo()
            if cfg["use_tls"]:
                server.starttls()
                server.ehlo()
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg)

        return True, "Email sent successfully"
    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check the Gmail address and Google App Password."
    except smtplib.SMTPConnectError:
        return False, "Could not connect to the SMTP server. Check host, port and internet access."
    except smtplib.SMTPServerDisconnected:
        return False, "SMTP server disconnected before the email was delivered."
    except smtplib.SMTPException as error:
        return False, f"SMTP error: {error}"
    except (TimeoutError, OSError) as error:
        return False, f"Network error while sending email: {error}"
    except Exception as error:  # noqa: BLE001
        return False, f"Email delivery failed: {error}"


def send_test_email(to_email):
    """Send a clearly labelled test message to verify the SMTP setup."""
    cfg, _ = get_smtp_config()
    sender_name = cfg["sender_name"] if cfg else "AquaTrack Support"
    sender_email = cfg["username"] if cfg else ""
    body = (
        f"{sender_name}\n\n"
        "This is a test email from the AquaTrack alert system.\n\n"
        "If you received this message, real email delivery is configured and "
        "working through the AquaTrack SMTP sender.\n\n"
        "AquaTrack\nPredictive Water Intelligence\n"
        f"{sender_email}\n"
    )
    return send_email_alert(
        to_email,
        "AquaTrack — SMTP Test Email",
        body,
    )


def queue_zone_alert(zone_id, zone_name, message, report_id=None):
    """Create in-app alerts and attempt real email delivery for every resident."""
    residents = get_zone_residents(zone_id)
    notified_count = 0
    email_count = 0

    for _, resident in residents.iterrows():
        resident_id = int(resident["resident_id"])
        resident_name = str(resident["name"]).strip() if pd.notna(resident["name"]) else "Resident"
        email = str(resident["email"]).strip() if pd.notna(resident["email"]) else ""
        phone = str(resident["phone"]).strip() if pd.notna(resident["phone"]) else ""
        recipient = email or phone or "in-app"

        delivered = False
        delivery_reason = "No email address configured"
        if email:
            smtp_cfg, _ = get_smtp_config()
            sender_email = smtp_cfg["username"] if smtp_cfg else ""
            sender_name = smtp_cfg["sender_name"] if smtp_cfg else "AquaTrack Support"
            email_body = (
                f"Dear {resident_name or 'Resident'},\n\n"
                f"AquaTrack has generated a water-service alert for {zone_name}.\n\n"
                f"{message.strip()}\n\n"
                "Please review the AquaTrack portal for the latest information. "
                "This notification is generated by the AquaTrack project-level decision-support system.\n\n"
                "Regards,\n"
                f"{sender_name}\n"
                f"{sender_email}"
            )
            delivered, delivery_reason = send_email_alert(
                email,
                f"AquaTrack Alert — {zone_name}",
                email_body,
            )

        if delivered:
            channel = "email+in-app"
            status = "Sent (email + in-app)"
        elif email:
            channel = "in-app"
            status = f"Email failed: {delivery_reason}"
        else:
            channel = "in-app"
            status = "Queued (in-app)"

        db_execute(
            """
            INSERT INTO alerts_log
                (zone_id, report_id, resident_id, message, channel, recipient, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(zone_id),
                None if report_id is None else int(report_id),
                resident_id,
                message,
                channel,
                recipient,
                status,
            ),
        )

        notified_count += 1
        email_count += int(delivered)

    return notified_count, email_count


def get_recent_alerts(limit=100, zone_id=None):
    sql = """
        SELECT
            alerts_log.alert_id,
            alerts_log.resident_id,
            zones.zone_name,
            COALESCE(residents.name, 'Resident') AS resident_name,
            alerts_log.message,
            alerts_log.channel,
            alerts_log.recipient,
            alerts_log.status,
            alerts_log.created_at
        FROM alerts_log
        JOIN zones ON alerts_log.zone_id = zones.zone_id
        LEFT JOIN residents ON alerts_log.resident_id = residents.resident_id
    """
    params = []

    if zone_id is not None:
        sql += " WHERE alerts_log.zone_id = ?"
        params.append(int(zone_id))

    sql += " ORDER BY alerts_log.alert_id DESC LIMIT ?"
    params.append(int(limit))
    return db_query(sql, tuple(params))


def get_resident_alerts(resident_id, limit=50):
    return db_query(
        """
        SELECT
            alerts_log.alert_id,
            zones.zone_name,
            alerts_log.message,
            alerts_log.channel,
            alerts_log.status,
            alerts_log.created_at
        FROM alerts_log
        JOIN zones ON alerts_log.zone_id = zones.zone_id
        WHERE alerts_log.resident_id = ?
        ORDER BY alerts_log.alert_id DESC
        LIMIT ?
        """,
        (int(resident_id), int(limit)),
    )


def get_resident_counts():
    result = db_query(
        """
        SELECT
            (SELECT COUNT(*) FROM residents) AS residents,
            (SELECT COUNT(*) FROM alerts_log) AS alerts,
            (SELECT COUNT(*) FROM alerts_log WHERE status LIKE 'Sent%') AS delivered
        """
    )
    return result.iloc[0].to_dict()


def notify_report_zone(report_id, zone_id, issue_type, description, severity):
    if zone_id is None or pd.isna(zone_id):
        return 0, 0

    zone_id = int(zone_id)
    zones = get_zones()
    match = zones.loc[zones["zone_id"] == zone_id, "zone_name"]
    if match.empty:
        return 0, 0

    zone_name = str(match.iloc[0])
    message = (
        f"New {issue_type} report in {zone_name}. "
        f"Severity: {severity}. "
        f"{str(description).strip()[:180]}"
    )
    return queue_zone_alert(zone_id, zone_name, message, report_id=report_id)



def clear_legacy_demo_residents_and_alerts_once():
    """Remove the old demo resident/alert records once, without affecting future registrations."""
    try:
        with closing(get_db_connection()) as connection:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS aquatrack_cleanup_flags (
                    flag_name TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            already_done = cursor.execute(
                "SELECT 1 FROM aquatrack_cleanup_flags WHERE flag_name=? LIMIT 1",
                ("clear_demo_residents_alerts_v1",),
            ).fetchone()
            if already_done:
                return

            cursor.execute("DELETE FROM alerts_log")
            cursor.execute("DELETE FROM residents")
            cursor.execute(
                "INSERT INTO aquatrack_cleanup_flags (flag_name) VALUES (?)",
                ("clear_demo_residents_alerts_v1",),
            )
            connection.commit()
    except Exception as error:
        LOGGER.warning("Initial resident/alert cleanup skipped: %s", error)


def demo_mode_enabled():
    """Return whether project-generated demo data seeding is explicitly enabled."""
    value = None
    try:
        value = st.secrets.get("AQUATRACK_DEMO_MODE")
    except Exception:
        value = None
    if value is None:
        value = os.getenv("AQUATRACK_DEMO_MODE", "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def ensure_demo_data():
    """Populate project-generated demo records only when demo mode is enabled."""
    if not demo_mode_enabled():
        return
    try:
        zones = db_query("SELECT zone_id, zone_name, latitude, longitude FROM zones ORDER BY zone_id")
        if zones.empty:
            return

        # --------------------------------------------------------
        # Core quality + availability records (only if empty)
        # --------------------------------------------------------
        quality_count = int(db_query("SELECT COUNT(*) AS c FROM water_quality")['c'].iloc[0])
        availability_count = int(db_query("SELECT COUNT(*) AS c FROM availability")['c'].iloc[0])
        rng = np.random.default_rng(42)
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90, freq="D")

        if quality_count == 0:
            quality_rows = []
            for zi, z in zones.iterrows():
                for di, dt in enumerate(dates):
                    # Mostly healthy values with a deterministic small risk pattern.
                    risk_band = (zi * 17 + di) % 20
                    ph = 7.15 + 0.35 * np.sin(di / 9 + zi)
                    tds = 300 + 45 * np.sin(di / 13 + zi * 0.7)
                    turbidity = 0.45 + 0.20 * abs(np.sin(di / 11 + zi))
                    hardness = 145 + 25 * np.sin(di / 15 + zi)
                    chloride = 150 + 45 * np.sin(di / 17 + zi * 0.5)
                    fluoride = 0.55 + 0.15 * np.sin(di / 12 + zi)
                    nitrate = 22 + 10 * np.sin(di / 14 + zi * 0.4)
                    if risk_band >= 18:
                        tds += 170; turbidity += 0.75; chloride += 90; nitrate += 20
                    elif risk_band >= 15:
                        tds += 90; turbidity += 0.35
                    quality_rows.append((int(z.zone_id), dt.strftime("%Y-%m-%d"),
                                         round(ph,2), round(tds,2), round(turbidity,2),
                                         round(hardness,2), round(chloride,2), round(fluoride,2), round(nitrate,2)))
            with closing(get_db_connection()) as con:
                con.executemany("""
                    INSERT INTO water_quality
                    (zone_id,date,ph,tds,turbidity,hardness,chloride,fluoride,nitrate)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, quality_rows)
                con.commit()

        if availability_count == 0:
            availability_rows = []
            seasons = ["Winter", "Summer", "Monsoon"]
            for zi, z in zones.iterrows():
                for di, dt in enumerate(dates):
                    season = seasons[(dt.month % 12) // 4]
                    base = 10.5 + 1.7 * np.sin(di / 10 + zi * 0.8)
                    rainfall = max(0.0, 6 + 22 * np.sin(di / 8 + zi) if season == "Monsoon" else 1.5 * abs(np.sin(di / 10 + zi)))
                    supply = max(3.8, min(18.0, base + (2.0 if season == "Monsoon" else 0) - (0.8 if season == "Summer" else 0)))
                    if (zi * 11 + di) % 29 == 0:
                        supply = 4.5
                    availability_rows.append((int(z.zone_id), dt.strftime("%Y-%m-%d"), round(supply,2), round(rainfall,2), season))
            with closing(get_db_connection()) as con:
                con.executemany("""
                    INSERT INTO availability (zone_id,date,supply_hours,rainfall,season)
                    VALUES (?,?,?,?,?)
                """, availability_rows)
                con.commit()

        # --------------------------------------------------------
        # Water bodies + served areas (only if absent)
        # --------------------------------------------------------
        bodies = db_query("SELECT water_body_id FROM water_bodies LIMIT 1")
        if bodies.empty:
            body_seed = [
                ("AquaTrack Reservoir North", "Reservoir", "North Indore", 22.7605, 75.8577, "Project-defined demonstration water body", 1),
                ("AquaTrack Tank Central", "Overhead Tank", "Central Indore", 22.7196, 75.8577, "Project-defined demonstration water body", 1),
                ("AquaTrack Reservoir East", "Reservoir", "East Indore", 22.7310, 75.9030, "Project-defined demonstration water body", 1),
                ("AquaTrack Tank South", "Overhead Tank", "South Indore", 22.6740, 75.8570, "Project-defined demonstration water body", 1),
            ]
            with closing(get_db_connection()) as con:
                con.executemany("""
                    INSERT INTO water_bodies (name,water_body_type,location,latitude,longitude,description,active)
                    VALUES (?,?,?,?,?,?,?)
                """, body_seed)
                con.commit()

        body_rows = db_query("SELECT water_body_id FROM water_bodies WHERE active=1 ORDER BY water_body_id")
        supply_count = int(db_query("SELECT COUNT(*) AS c FROM water_body_supply")['c'].iloc[0])
        if supply_count == 0 and not body_rows.empty:
            supply_rows = []
            for i, b in enumerate(body_rows.itertuples(index=False)):
                for j, z in enumerate(zones.head(2).itertuples(index=False)):
                    supply_rows.append((int(b.water_body_id), int(z.zone_id), f"Demo Colony {i+1}{j+1}", f"Aqua Society {i+1}{j+1}", "Project-generated served-area relationship", 1))
            with closing(get_db_connection()) as con:
                con.executemany("""
                    INSERT INTO water_body_supply
                    (water_body_id,zone_id,colony_name,society_name,supply_description,active)
                    VALUES (?,?,?,?,?,?)
                """, supply_rows)
                con.commit()

        # --------------------------------------------------------
        # Workers + residents + service requests
        # --------------------------------------------------------
        workers = get_workers(active_only=False)
        if workers.empty:
            zone_ids = zones.zone_id.tolist()
            # Random credentials are generated locally so no reusable password is
            # embedded in source code, logs, screenshots, or the response.
            worker_seed = [
                ("worker_north", hash_worker_password(secrets.token_urlsafe(18)), "Demo Field Worker 1", "9000000001", int(zone_ids[0]), "Available"),
                ("worker_central", hash_worker_password(secrets.token_urlsafe(18)), "Demo Field Worker 2", "9000000002", int(zone_ids[min(1,len(zone_ids)-1)]), "Available"),
                ("worker_east", hash_worker_password(secrets.token_urlsafe(18)), "Demo Field Worker 3", "9000000003", int(zone_ids[min(2,len(zone_ids)-1)]), "Available"),
                ("worker_south", hash_worker_password(secrets.token_urlsafe(18)), "Demo Field Worker 4", "9000000004", int(zone_ids[min(3,len(zone_ids)-1)]), "Available"),
            ]
            with closing(get_db_connection()) as con:
                con.executemany("""
                    INSERT INTO aquatrack_workers
                    (username,password_hash,name,phone,zone_id,availability_status)
                    VALUES (?,?,?,?,?,?)
                """, worker_seed)
                con.commit()
            workers = get_workers(active_only=False)

        # Resident accounts are created only by real citizen registration.
        # No demo residents are seeded.
        residents = db_query("SELECT resident_id,zone_id,name,phone,email FROM residents ORDER BY resident_id")

        report_count = int(db_query("SELECT COUNT(*) AS c FROM water_body_reports")['c'].iloc[0])
        if report_count == 0 and not body_rows.empty and not residents.empty:
            workers = get_workers(active_only=True)
            worker_ids = workers.worker_id.tolist()
            supply = db_query("SELECT water_body_id,zone_id,colony_name,society_name FROM water_body_supply ORDER BY supply_id")
            if not supply.empty:
                now = pd.Timestamp.now()
                demo_reports=[]
                statuses=["Submitted","Assigned","In Progress","Waiting for Verification","Resolved","Resolved"]
                issues=["Low water supply","Water appears dirty","Bad smell","Floating waste","Possible contamination","Unusual colour"]
                for i, row in enumerate(supply.head(6).itertuples(index=False)):
                    status=statuses[i]
                    resident_id=int(residents.iloc[i % len(residents)].resident_id)
                    worker_id=None if status=="Submitted" else int(worker_ids[i % len(worker_ids)])
                    legacy_status="Resolved" if status=="Resolved" else ("Investigating" if status in ("Assigned","In Progress","Waiting for Verification") else "Pending")
                    created=(now-pd.Timedelta(days=6-i)).strftime("%Y-%m-%d %H:%M:%S")
                    demo_reports.append((int(row.water_body_id),int(row.zone_id),str(row.colony_name),str(row.society_name),issues[i],
                        "Project-generated demonstration service request.",None,22.72,75.86,legacy_status,["Low","Medium","High","Medium","High","Low"][i],None,status,resident_id,worker_id,created))
                with closing(get_db_connection()) as con:
                    for rec in demo_reports:
                        con.execute("""
                            INSERT INTO water_body_reports
                            (water_body_id,zone_id,colony_name,society_name,issue_type,description,photo_path,latitude,longitude,status,severity,reporter_user_id,workflow_status,resident_id,assigned_worker_id,created_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, rec)
                    con.commit()

        # Make the demo workflow internally consistent.
        reqs = db_query("SELECT report_id,workflow_status,assigned_worker_id,resident_id FROM water_body_reports WHERE workflow_status IS NOT NULL ORDER BY report_id")
        for r in reqs.itertuples(index=False):
            if pd.notna(r.assigned_worker_id) and str(r.workflow_status) in ("Assigned","In Progress","Waiting for Verification"):
                db_execute("UPDATE aquatrack_workers SET availability_status='Busy' WHERE worker_id=?", (int(r.assigned_worker_id),))
            if str(r.workflow_status)=="Resolved" and pd.notna(r.assigned_worker_id):
                db_execute("UPDATE aquatrack_workers SET availability_status='Available' WHERE worker_id=?", (int(r.assigned_worker_id),))

        # A small prediction history makes Dashboard/Admin exports useful immediately.
        pred_count = int(db_query("SELECT COUNT(*) AS c FROM predictions")['c'].iloc[0])
        if pred_count == 0:
            quality = get_water_quality()
            availability = get_availability()
            if not quality.empty and not availability.empty:
                for zname in zones.zone_name.tolist()[:6]:
                    qz = quality[quality.zone_name == zname]
                    az = availability[availability.zone_name == zname]
                    if qz.empty or az.empty:
                        continue
                    risk = predict_quality_batch(qz.tail(1))[0]
                    forecast = float(az.tail(1).supply_hours.iloc[0])
                    save_prediction(zname, risk=risk, forecast=forecast)

        # Demo resident/admin notifications only when the new inbox is empty.
        notif_count = int(db_query("SELECT COUNT(*) AS c FROM aquatrack_notifications")['c'].iloc[0])
        if notif_count == 0:
            demo_req = db_query("SELECT report_id,assigned_worker_id,resident_id FROM water_body_reports ORDER BY report_id DESC LIMIT 3")
            for r in demo_req.itertuples(index=False):
                add_notification("admin", None, int(r.report_id), f"Demo request #{int(r.report_id)} is available in the service-request workflow.")
                if pd.notna(r.assigned_worker_id):
                    add_notification("worker", int(r.assigned_worker_id), int(r.report_id), f"Demo assignment #{int(r.report_id)} is ready for field action.")
                if pd.notna(r.resident_id):
                    add_notification("resident", int(r.resident_id), int(r.report_id), f"Demo request #{int(r.report_id)} is being tracked in AquaTrack.")

    except Exception as error:
        # Demo seeding must never prevent the application from starting.
        print(f"AquaTrack demo-data seed skipped: {error}")


ensure_alert_tables()
ensure_workflow_tables()

clear_legacy_demo_residents_and_alerts_once()


# ============================================================
# MODELS (loaded once per server process)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_quality_model():
    return joblib.load(QUALITY_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_availability_model():
    return joblib.load(AVAILABILITY_MODEL_PATH)


def predict_quality_batch(df):
    model = load_quality_model()
    features = df[DB_QUALITY_COLS].copy()
    features.columns = QUALITY_FEATURES
    return model.predict(features)


def predict_water_quality(sample):
    model = load_quality_model()
    row = pd.DataFrame([{name: float(sample[name]) for name in QUALITY_FEATURES}])
    return str(model.predict(row)[0])


def quality_probabilities(sample):
    model = load_quality_model()
    row = pd.DataFrame([{name: float(sample[name]) for name in QUALITY_FEATURES}])
    probabilities = model.predict_proba(row)[0]
    series = pd.Series(probabilities, index=[str(c) for c in model.classes_])
    return series.reindex(RISK_ORDER).fillna(0.0)


def classify_availability(hours):
    if hours < 5:
        return "Scarcity Risk"
    if hours < 8:
        return "Moderate Availability"
    return "Good Availability"


def forecast_water_availability(previous_supply_hours, rainfall, season):
    if season not in SEASON_CODES:
        raise ValueError(f"Invalid season: {season}")
    model = load_availability_model()
    row = pd.DataFrame([{
        "previous_supply_hours": float(previous_supply_hours),
        "rainfall": float(rainfall),
        "season_code": SEASON_CODES[season],
    }])
    hours = float(model.predict(row)[0])
    hours = max(0.0, min(24.0, hours))
    return {"forecast_hours": round(hours, 2), "status": classify_availability(hours)}


def forecast_series(previous_supply_hours, rainfall, season, days):
    results = []
    current = float(previous_supply_hours)
    for day in range(1, days + 1):
        step = forecast_water_availability(current, rainfall, season)
        results.append({"day": day, **step})
        current = step["forecast_hours"]
    return pd.DataFrame(results)


# ============================================================
# BIS SCREENING HELPERS
# ============================================================

def rule_label(violation_count):
    if violation_count == 0:
        return "Safe"
    if violation_count <= 2:
        return "Moderate Risk"
    return "High Risk"


def violation_flags(df):
    flags = pd.DataFrame(index=df.index)
    for key, info in PARAM_INFO.items():
        bad = pd.Series(False, index=df.index)
        if "min" in info:
            bad = bad | (df[key] < info["min"])
        if "max" in info:
            bad = bad | (df[key] > info["max"])
        flags[key] = bad
    return flags


def compliance_table(values):
    rows = []
    violations = 0
    for key, info in PARAM_INFO.items():
        value = float(values[key])
        bad = ("min" in info and value < info["min"]) or ("max" in info and value > info["max"])
        violations += int(bad)
        limit = f"{info['min']} – {info['max']}" if "min" in info else f"≤ {info['max']}"
        label = info["label"] + (f" ({info['unit']})" if info["unit"] else "")
        rows.append({
            "Parameter": label,
            "Measured": round(value, 2),
            "BIS limit": limit,
            "Status": "⚠️ Exceeds limit" if bad else "✅ Within limit",
        })
    return pd.DataFrame(rows), violations


def risk_badge(risk):
    return f"{RISK_EMOJI.get(risk, '⚪')} {risk}"


def status_badge(status):
    return f"{STATUS_EMOJI.get(status, '⚪')} {status}"


# ============================================================
# PRESENTATION DATA CLEANUP
# ============================================================

def prepare_quality_demo_data(df):
    """Return a copy for display without changing measured database values.

    Demo/synthetic data is generated at seed time; presentation functions must
    never clamp or rewrite measurements because predictions, analytics and
    exports must reflect the stored source data exactly.
    """
    if df is None or df.empty:
        return df
    return df.copy()


def prepare_availability_demo_data(df):
    """Return a copy for display without altering stored supply measurements."""
    if df is None or df.empty:
        return df
    return df.copy()


# ============================================================
# CACHED DATA LAYER
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_zones():
    return db_get_zones()


@st.cache_data(ttl=60, show_spinner=False)
def get_water_quality(zone_name=None):
    data = db_get_water_quality() if zone_name is None else db_get_water_quality(zone_name)
    return prepare_quality_demo_data(data)


@st.cache_data(ttl=60, show_spinner=False)
def get_availability(zone_name=None):
    data = db_get_availability() if zone_name is None else db_get_availability(zone_name)
    return prepare_availability_demo_data(data)


@st.cache_data(ttl=60, show_spinner=False)
def get_prediction_history():
    return db_query(
        """
        SELECT
            predictions.prediction_id,
            zones.zone_name AS zone,
            predictions.prediction_date,
            predictions.risk_level,
            predictions.availability_forecast
        FROM predictions
        JOIN zones ON predictions.zone_id = zones.zone_id
        ORDER BY predictions.prediction_id DESC
        """
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_table_counts():
    counts = {}
    for name in [
        "zones", "water_quality", "availability", "predictions",
        "water_bodies", "residents", "alerts_log"
    ]:
        counts[name] = int(db_query(f"SELECT COUNT(*) AS c FROM {name}")["c"].iloc[0])
    return counts


@st.cache_data(ttl=60, show_spinner=False)
def get_water_bodies():
    return db_query(
        """
        SELECT water_body_id, name, water_body_type, location,
               latitude, longitude, description
        FROM water_bodies
        WHERE active = 1
        ORDER BY name
        """
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_water_body_supply(water_body_id):
    return db_query(
        """
        SELECT
            water_body_supply.supply_id,
            water_body_supply.water_body_id,
            water_body_supply.zone_id,
            zones.zone_name,
            water_body_supply.colony_name,
            water_body_supply.society_name,
            water_body_supply.supply_description
        FROM water_body_supply
        LEFT JOIN zones ON water_body_supply.zone_id = zones.zone_id
        WHERE water_body_supply.water_body_id = ?
          AND water_body_supply.active = 1
        ORDER BY water_body_supply.colony_name, water_body_supply.society_name
        """,
        params=(int(water_body_id),),
    )


def get_water_body_reports():
    return db_query(
        """
        SELECT
            water_body_reports.report_id,
            water_body_reports.zone_id,
            water_bodies.name AS water_body,
            water_body_reports.colony_name,
            water_body_reports.society_name,
            water_body_reports.issue_type,
            water_body_reports.description,
            water_body_reports.photo_path,
            water_body_reports.latitude,
            water_body_reports.longitude,
            water_body_reports.status,
            water_body_reports.workflow_status,
            water_body_reports.severity,
            water_body_reports.assigned_worker_id,
            water_body_reports.created_at,
            water_body_reports.updated_at,
            users.name AS reporter_name,
            aquatrack_workers.name AS worker_name
        FROM water_body_reports
        LEFT JOIN water_bodies
            ON water_body_reports.water_body_id = water_bodies.water_body_id
        LEFT JOIN users
            ON water_body_reports.reporter_user_id = users.user_id
        LEFT JOIN aquatrack_workers
            ON water_body_reports.assigned_worker_id = aquatrack_workers.worker_id
        ORDER BY water_body_reports.report_id DESC
        """
    )


def update_water_body_report_status(report_id, new_status, actor_id=None):
    """Keep legacy report status and the new worker workflow synchronized."""
    mapping = {"Pending": "Submitted", "Investigating": "In Progress", "Resolved": "Resolved"}
    legacy = str(new_status).strip()
    workflow = mapping.get(legacy)
    if workflow:
        ok, msg = change_request_status(int(report_id), workflow, "admin", actor_id)
        if not ok:
            return False, msg
    db_execute(
        "UPDATE water_body_reports SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE report_id = ?",
        (legacy, int(report_id)),
    )
    audit_event("legacy_status_changed", "admin", actor_id, report_id, legacy)
    return True, "Report status updated."


def update_water_body_report_severity(report_id, new_severity, actor_id=None):
    if str(new_severity) not in {"Low", "Medium", "High"}:
        return False, "Invalid severity."
    db_execute(
        """
        UPDATE water_body_reports
        SET severity = ?, updated_at = CURRENT_TIMESTAMP
        WHERE report_id = ?
        """,
        (new_severity, int(report_id)),
    )
    audit_event("severity_changed", "admin", actor_id, report_id, str(new_severity))
    return True, "Severity updated."


def notify_high_severity_report(report_id):
    report = db_query(
        """
        SELECT report_id, zone_id, issue_type, description, severity
        FROM water_body_reports
        WHERE report_id = ?
        """,
        (int(report_id),),
    )

    if report.empty or str(report.iloc[0]["severity"]) != "High":
        return 0, 0

    row = report.iloc[0]
    if pd.isna(row["zone_id"]):
        return 0, 0

    zones = get_zones()
    match = zones.loc[zones["zone_id"] == int(row["zone_id"]), "zone_name"]
    if match.empty:
        return 0, 0

    zone_name = str(match.iloc[0])
    message = (
        f"High-severity water issue in {zone_name}: "
        f"{row['issue_type']}. {str(row['description']).strip()[:180]}"
    )
    return queue_zone_alert(
        int(row["zone_id"]),
        zone_name,
        message,
        report_id=int(row["report_id"]),
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_zone_snapshot():
    zones = get_zones()
    quality = get_water_quality()
    availability = get_availability()

    snapshot = zones[["zone_name", "latitude", "longitude"]].copy()

    quality_columns = ["zone_name", "quality_date"] + DB_QUALITY_COLS + ["risk", "violations"]
    if not quality.empty:
        latest_quality = (
            quality.sort_values("date").groupby("zone_name").tail(1).copy()
        )
        latest_quality["risk"] = predict_quality_batch(latest_quality)
        latest_quality["violations"] = violation_flags(latest_quality).sum(axis=1)
        latest_quality = latest_quality.rename(columns={"date": "quality_date"})
        snapshot = snapshot.merge(latest_quality[quality_columns], on="zone_name", how="left")
    else:
        for column in quality_columns[1:]:
            snapshot[column] = np.nan

    availability_columns = ["zone_name", "supply_hours", "rainfall", "season"]
    if not availability.empty:
        latest_availability = (
            availability.sort_values("date").groupby("zone_name").tail(1)
        )[availability_columns]
        snapshot = snapshot.merge(latest_availability, on="zone_name", how="left")
    else:
        for column in availability_columns[1:]:
            snapshot[column] = np.nan

    snapshot["risk"] = snapshot["risk"].fillna("No Data")
    snapshot["availability_status"] = snapshot["supply_hours"].apply(
        lambda h: classify_availability(h) if pd.notna(h) else "No Data"
    )
    return snapshot


@st.cache_data(ttl=60, show_spinner=False)
def get_risk_distribution():
    quality = get_water_quality()
    if quality.empty:
        return pd.DataFrame(columns=["Risk Level", "Records"])
    counts = pd.Series(predict_quality_batch(quality)).value_counts()
    counts = counts.reindex(RISK_ORDER).fillna(0).astype(int)
    return counts.rename_axis("Risk Level").reset_index(name="Records")


@st.cache_data(ttl=60, show_spinner=False)
def get_seasonal_supply():
    availability = get_availability()
    if availability.empty:
        return pd.DataFrame(columns=["season", "supply_hours"])
    return availability.groupby("season", as_index=False)["supply_hours"].mean()


@st.cache_data(ttl=60, show_spinner=False)
def get_model_insights():
    """Evaluate the stored models on held-out records without mutating the database."""
    insights = {}

    quality = get_water_quality()
    if not quality.empty:
        X = quality[DB_QUALITY_COLS].copy()
        y = violation_flags(quality).sum(axis=1).apply(rule_label)
        stratify = y if y.value_counts().min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(y_test, predicted, labels=RISK_ORDER)
        report = classification_report(
            y_test, predicted, labels=RISK_ORDER, output_dict=True, zero_division=0
        )
        importances = pd.DataFrame({
            "Parameter": QUALITY_FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        insights["quality"] = {
            "records": len(X_test),
            "accuracy": accuracy_score(y_test, predicted),
            "matrix": pd.DataFrame(
                matrix, index=[f"Rule: {c}" for c in RISK_ORDER], columns=[f"Model: {c}" for c in RISK_ORDER]
            ),
            "report": pd.DataFrame(report).T.loc[RISK_ORDER + ["macro avg"], ["precision", "recall", "f1-score", "support"]],
            "importances": importances,
        }

    availability = get_availability()
    if not availability.empty:
        data = availability.sort_values(["zone_name", "date"]).copy()
        data["previous_supply_hours"] = data.groupby("zone_name")["supply_hours"].shift(1)
        data["season_code"] = data["season"].map(SEASON_CODES)
        data = data.dropna(subset=["previous_supply_hours", "season_code"])
        if len(data) >= 5:
            features = data[["previous_supply_hours", "rainfall", "season_code"]]
            actual = data["supply_hours"].to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(features, actual, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predicted = np.clip(model.predict(X_test), 0, 24)
            naive_mae = float(np.mean(np.abs(y_test - X_test["previous_supply_hours"].to_numpy())))
            coefficients = pd.DataFrame({
                "Feature": ["Previous supply hours", "Rainfall", "Season code"],
                "Coefficient": model.coef_,
            })
            scatter = pd.DataFrame({"Actual": y_test, "Predicted": predicted})
            insights["availability"] = {
                "records": len(X_test),
                "mae": float(mean_absolute_error(y_test, predicted)),
                "r2": float(r2_score(y_test, predicted)),
                "naive_mae": naive_mae,
                "coefficients": coefficients,
                "intercept": float(model.intercept_),
                "scatter": scatter.sample(min(600, len(scatter)), random_state=42),
            }
    return insights


@st.cache_data(ttl=30, show_spinner=False)
def get_request_history(report_id):
    return db_query(
        """SELECT history_id, old_status, new_status, changed_by_role, changed_by_id, remarks, created_at
           FROM request_status_history WHERE report_id=? ORDER BY history_id ASC""",
        (int(report_id),),
    )


def get_unread_count(role, recipient_id):
    if str(role) == "admin" and recipient_id is None:
        row = db_query(
            "SELECT COUNT(*) AS c FROM aquatrack_notifications WHERE recipient_role=? AND is_read=0",
            (str(role),),
        )
    else:
        row = db_query(
            "SELECT COUNT(*) AS c FROM aquatrack_notifications WHERE recipient_role=? AND recipient_id=? AND is_read=0",
            (str(role), int(recipient_id)),
        )
    return int(row.iloc[0]["c"]) if not row.empty else 0


def get_worker_performance():
    return db_query(
        """SELECT w.worker_id, w.name, z.zone_name,
               COUNT(r.report_id) AS total_assigned,
               SUM(CASE WHEN r.workflow_status='Resolved' THEN 1 ELSE 0 END) AS completed,
               ROUND(AVG(CASE WHEN r.resolved_at IS NOT NULL AND r.created_at IS NOT NULL
                    THEN (julianday(r.resolved_at)-julianday(r.created_at))*24 END), 2) AS avg_resolution_hours,
               ROUND(AVG(rv.rating), 2) AS avg_rating
           FROM aquatrack_workers w
           LEFT JOIN zones z ON w.zone_id=z.zone_id
           LEFT JOIN water_body_reports r ON r.assigned_worker_id=w.worker_id
           LEFT JOIN citizen_reviews rv ON rv.report_id=r.report_id
           WHERE w.active = 1
           GROUP BY w.worker_id, w.name, z.zone_name ORDER BY completed DESC, avg_rating DESC"""
    )


def get_service_metrics():
    requests_df = get_service_requests()
    workers = get_workers(active_only=True)
    reviews = get_reviews()
    if requests_df.empty:
        return {
            "total": 0, "pending": 0, "assigned": 0, "in_progress": 0,
            "verification": 0, "resolved": 0, "high_priority": 0,
            "avg_resolution_hours": None, "satisfaction": None,
            "active_workers": len(workers),
        }
    status = requests_df["workflow_status"].fillna("Submitted")
    resolved_times = pd.to_numeric(
        (pd.to_datetime(requests_df["resolved_at"], errors="coerce") -
         pd.to_datetime(requests_df["created_at"], errors="coerce")) / pd.Timedelta(hours=1),
        errors="coerce",
    )
    return {
        "total": len(requests_df),
        "pending": int(status.isin(["Submitted", "Reopened"]).sum()),
        "assigned": int((status == "Assigned").sum()),
        "in_progress": int((status == "In Progress").sum()),
        "verification": int(status.isin(["Waiting for Verification"]).sum()),
        "resolved": int((status == "Resolved").sum()),
        "high_priority": int((requests_df["severity"] == "High").sum()),
        "avg_resolution_hours": float(resolved_times.dropna().mean()) if resolved_times.notna().any() else None,
        "satisfaction": float(reviews["rating"].mean()) if not reviews.empty else None,
        "active_workers": len(workers),
    }


def create_database_backup():
    source = Path(DATABASE_NAME)
    if not source.exists():
        return None, "Database file not found."
    folder = BASE_DIR / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"aquatrack_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.db"
    try:
        with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
        audit_event("database_backup_created", "admin", None, None, target.name)
        return target, "Backup created successfully."
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Database backup failed")
        return None, "Database backup failed."


def latest_backup():
    folder = BASE_DIR / "backups"
    files = sorted(folder.glob("aquatrack_*.db"), key=lambda p: p.stat().st_mtime, reverse=True) if folder.exists() else []
    return files[0] if files else None


def get_system_health():
    """Return lightweight health information without mutating project data."""
    quality = get_water_quality()
    availability = get_availability()
    quality_date = None
    availability_date = None
    if not quality.empty and "date" in quality.columns:
        quality_date = pd.to_datetime(quality["date"], errors="coerce").max()
    if not availability.empty and "date" in availability.columns:
        availability_date = pd.to_datetime(availability["date"], errors="coerce").max()
    quality_model_ok = False
    availability_model_ok = False
    try:
        load_quality_model()
        quality_model_ok = True
    except Exception:
        LOGGER.exception("Quality model health check failed")
    try:
        load_availability_model()
        availability_model_ok = True
    except Exception:
        LOGGER.exception("Availability model health check failed")
    return {
        "models_ready": quality_model_ok and availability_model_ok,
        "quality_model_ready": quality_model_ok,
        "availability_model_ready": availability_model_ok,
        "quality_date": quality_date,
        "availability_date": availability_date,
        "database_ready": Path(DATABASE_NAME).exists(),
        "backup": latest_backup(),
    }


def clear_data_caches():
    cached_functions = (
        get_zones, get_water_quality, get_availability, get_prediction_history,
        get_table_counts, get_water_bodies, get_water_body_supply, get_zone_snapshot,
        get_risk_distribution, get_seasonal_supply, get_model_insights, get_request_history,
    )
    for cached_function in cached_functions:
        try:
            cached_function.clear()
        except Exception:
            LOGGER.warning("Could not clear cache for %s", getattr(cached_function, "__name__", "unknown"))


def save_prediction(zone_name, risk=None, forecast=None):
    zones = get_zones()
    match = zones.loc[zones["zone_name"] == zone_name, "zone_id"]
    if match.empty:
        return False, "Zone not found."
    try:
        db_execute(
            """
            INSERT INTO predictions
                (zone_id, prediction_date, risk_level, availability_forecast)
            VALUES (?, ?, ?, ?)
            """,
            (
                int(match.iloc[0]),
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                risk,
                None if forecast is None else float(forecast),
            ),
        )
        clear_data_caches()
        return True, ""
    except Exception as error:  # noqa: BLE001
        return False, str(error)


# Demo seed must run only after all data/model helpers are defined.
ensure_demo_data()


# ============================================================
# LOCATION HELPERS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def reverse_geocode(latitude, longitude):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": "AquaTrack-Student-Project/1.0"},
            timeout=8,
        )
        response.raise_for_status()
        return response.json().get("display_name", "Current location")
    except Exception:  # noqa: BLE001
        return "Current location (address lookup unavailable)"


def make_base_map(center, zoom):
    # Fast, reliable default map. Satellite remains available from the layer control.
    fmap = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(fmap)

    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    return fmap


def add_map_legend(fmap, items):
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{color};display:inline-block;"></span>'
        f"<span>{label}</span></div>"
        for label, color in items
    )
    html = (
        '<div style="position:fixed;bottom:82px;left:28px;z-index:9999;background:white;'
        "padding:10px 14px;border-radius:12px;box-shadow:0 4px 14px rgba(0,0,0,.25);"
        f'font-size:12px;font-family:"Noto Sans","Noto Sans Devanagari","Segoe UI",Arial,sans-serif;color:#173b56;">{rows}</div>'
    )
    fmap.get_root().html.add_child(folium.Element(html))


# ============================================================
# CHART HELPERS
# ============================================================

def style_fig(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color="#173b56"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def show_chart(fig, height=360):
    # Charts are ADMIN-only. Public users receive the same underlying values
    # through metrics, tables and written summaries.
    if not st.session_state.get("admin_logged_in", False):
        return
    st.plotly_chart(
        _localize_figure(style_fig(fig, height)),
        width="stretch",
        height=height,
        config={"displayModeBar": False, "responsive": True},
    )


def section(title):
    st.markdown(f'<div class="section-heading">{t(title)}</div>', unsafe_allow_html=True)


def snapshot_display(snapshot):
    display = snapshot.copy()
    display["Water quality"] = display["risk"].map(lambda r: risk_badge(t(r)))
    display["Availability"] = display["availability_status"].map(lambda s: status_badge(t(s)))
    display["violations"] = display["violations"].astype("Int64")
    display["supply_hours"] = display["supply_hours"].round(2)
    return display[
        ["zone_name", "Water quality", "violations", "supply_hours", "Availability"]
    ].rename(columns={
        "zone_name": "Zone",
        "violations": "BIS violations",
        "supply_hours": "Supply (h)",
    })


def filter_recent(df, days):
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    if days:
        data = data[data["date"] >= data["date"].max() - pd.Timedelta(days=days)]
    return data


# ============================================================
# REPORT EXPORT (built on demand, never on every rerun)
# ============================================================

def _pdf_table(rows, col_widths=None):
    table = Table(rows, colWidths=col_widths)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ])
    )
    return table


def create_pdf_report(zone_name, quality_data, availability_data, prediction_history, summary):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("AquaTrack", styles["Title"]),
        Paragraph("Water Quality Assessment and Water Availability Forecast Report", styles["Heading2"]),
        Spacer(1, 10),
        Paragraph(f"Zone: {zone_name}", styles["Normal"]),
        Paragraph(f"Generated: {pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Summary", styles["Heading2"]),
        _pdf_table(
            [
                ["Item", "Value"],
                ["Predicted water-quality risk", summary.get("risk", "No Data")],
                ["BIS parameters exceeded", str(summary.get("violations", "-"))],
                ["Latest supply (hours)", str(summary.get("supply", "-"))],
                ["Availability status", summary.get("status", "No Data")],
            ],
            [220, 180],
        ),
        Spacer(1, 12),
        Paragraph("Water Quality (latest reading)", styles["Heading2"]),
    ]

    if not quality_data.empty:
        latest = quality_data.iloc[-1].to_dict()
        table, _ = compliance_table(latest)
        rows = [["Parameter", "Value", "BIS limit", "Status"]]
        for _, row in table.iterrows():
            rows.append([
                row["Parameter"], str(row["Measured"]), row["BIS limit"],
                "Exceeds" if "Exceeds" in row["Status"] else "Within",
            ])
        elements.append(_pdf_table(rows, [150, 80, 90, 80]))
    else:
        elements.append(Paragraph("No water-quality data available.", styles["Normal"]))

    elements += [Spacer(1, 12), Paragraph("Water Availability (latest record)", styles["Heading2"])]
    if not availability_data.empty:
        latest = availability_data.iloc[-1]
        elements.append(_pdf_table(
            [
                ["Parameter", "Value"],
                ["Supply Hours", str(round(float(latest["supply_hours"]), 2))],
                ["Rainfall (mm)", str(round(float(latest["rainfall"]), 2))],
                ["Season", str(latest["season"])],
            ],
            [220, 180],
        ))
    else:
        elements.append(Paragraph("No availability data available.", styles["Normal"]))

    elements += [Spacer(1, 12), Paragraph("Prediction History (latest 10)", styles["Heading2"])]
    zone_history = prediction_history[prediction_history["zone"] == zone_name].head(10)
    if not zone_history.empty:
        rows = [["ID", "Date", "Risk", "Forecast (h)"]]
        for _, row in zone_history.iterrows():
            forecast = "-" if pd.isna(row["availability_forecast"]) else str(round(float(row["availability_forecast"]), 2))
            risk = "-" if pd.isna(row["risk_level"]) else str(row["risk_level"])
            rows.append([str(row["prediction_id"]), str(row["prediction_date"]), risk, forecast])
        elements.append(_pdf_table(rows))
    else:
        elements.append(Paragraph("No saved predictions for this zone.", styles["Normal"]))

    elements += [Spacer(1, 16), Paragraph("Disclaimer: " + DISCLAIMER, styles["Normal"])]
    document.build(elements)
    buffer.seek(0)
    return buffer


def _append_dataframe(sheet, df):
    if df.empty:
        return
    sheet.append(list(df.columns))
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    clean = df.astype(object).where(pd.notna(df), None)
    for row in clean.itertuples(index=False):
        sheet.append(list(row))


def create_excel_report(zone_name, quality_data, availability_data, prediction_history, summary):
    buffer = io.BytesIO()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    sheet["A1"] = "AquaTrack"
    sheet["A1"].font = Font(bold=True, size=14)
    sheet["A2"] = "Water Quality and Water Availability Report"
    sheet["A4"], sheet["B4"] = "Zone", zone_name
    sheet["A5"], sheet["B5"] = "Predicted risk", summary.get("risk", "No Data")
    sheet["A6"], sheet["B6"] = "BIS parameters exceeded", summary.get("violations", "-")
    sheet["A7"], sheet["B7"] = "Latest supply (hours)", summary.get("supply", "-")
    sheet["A8"], sheet["B8"] = "Availability status", summary.get("status", "No Data")
    sheet["A10"] = DISCLAIMER

    if not quality_data.empty:
        table, _ = compliance_table(quality_data.iloc[-1].to_dict())
        _append_dataframe(workbook.create_sheet("Latest BIS Check"), table)
    _append_dataframe(workbook.create_sheet("Water Quality Data"), quality_data)
    _append_dataframe(workbook.create_sheet("Availability Data"), availability_data)
    _append_dataframe(
        workbook.create_sheet("Prediction History"),
        prediction_history[prediction_history["zone"] == zone_name],
    )
    workbook.save(buffer)
    buffer.seek(0)
    return buffer


# ============================================================
# PAGE CONFIGURATION AND STYLING
# ============================================================

st.set_page_config(page_title="AquaTrack", page_icon=(str(BASE_DIR / "aquatrack_favicon.png") if (BASE_DIR / "aquatrack_favicon.png").exists() else "💧"), layout="wide")

st.markdown(
    """
    <style>
    /* ============================================================
       AQUATRACK GOVERNMENT-STYLE TYPOGRAPHY
       One consistent Noto Sans family across the entire website.
       Noto Sans Devanagari keeps Hindi visually consistent too.
       ============================================================ */

    /* ============================================================
       TABLE ALIGNMENT
       Keep all Streamlit data tables visually consistent by centering
       both table headers and cell contents.
       ============================================================ */
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"] {
        text-align: center !important;
        justify-content: center !important;
    }

    div[data-testid="stDataFrame"] [role="columnheader"] > div,
    div[data-testid="stDataFrame"] [role="gridcell"] > div {
        justify-content: center !important;
        text-align: center !important;
    }

    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700;800&display=swap');

    html, body, .stApp,
    .stApp *,
    [class*="css"],
    input, textarea, select, button,
    [role="button"], [role="option"], [role="combobox"],
    .stSelectbox, .stTextInput, .stTextArea,
    .stNumberInput, .stDateInput, .stTimeInput,
    .stRadio, .stCheckbox, .stSlider,
    .stButton, .stDownloadButton {
        font-family: "Noto Sans", "Noto Sans Devanagari", "Segoe UI", Arial, sans-serif !important;
    }

    /* Keep Streamlit/Material icons on their own icon font.
       Without this override, the global Noto Sans rule turns the
       password eye icon into the literal word "visibility". */
    /* Hide the browser's native password-reveal eye (Edge/Chromium).
       Streamlit's own password visibility control remains available. */
    input[type="password"]::-ms-reveal,
    input[type="password"]::-ms-clear {
        display: none !important;
    }

    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    .material-icons-round,
    span[data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons", sans-serif !important;
        font-style: normal !important;
        font-weight: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: "liga" !important;
    }

    .stApp { background:#f4f9fc; color:#173b56; }
    /* One shared content rail for the entire AquaTrack website.
       Every public/admin page, footer rule and fixed page element uses this rail. */
    :root { --aqua-content-width:1212px; }
    .main .block-container {
        width:min(var(--aqua-content-width), calc(100vw - 2rem)) !important;
        max-width:var(--aqua-content-width) !important;
        box-sizing:border-box !important;
        margin-left:auto !important;
        margin-right:auto !important;
        padding-top:0 !important;
        padding-bottom:0 !important;
        padding-left:1rem;
        padding-right:1rem;
    }
    div[data-testid="stMainBlockContainer"] { padding-top:0 !important; }
    .stMainBlockContainer { padding-top:0 !important; }
    section[data-testid="stMain"] > div { padding-top:0 !important; padding-bottom:0 !important; min-height:auto !important; display:flex; flex-direction:column; box-sizing:border-box; }
    header[data-testid="stHeader"] { display:none !important; height:0 !important; min-height:0 !important; }
    section[data-testid="stSidebar"] { display:none; }
    h1 { color:#173b56 !important; font-weight:800 !important; letter-spacing:-.045em; }
    h2,h3 { color:#173b56 !important; font-weight:750 !important; }
    p,label,.stMarkdown { color:#4f6d80; }
    .block-container > div { width:100%; }
    div[data-testid="stVerticalBlock"] { gap:.90rem; }
    div[data-testid="stHorizontalBlock"] { align-items:stretch; }
    div[data-testid="stHorizontalBlock"] > div { min-width:0; }
    .stCaption { color:#78909f !important; }

    /* Keep the secondary navigation visually separated from the main header.
       This is intentionally scoped to the nav spacer, not the whole website. */
    .aqua-subnav-gap {
        height:1.15rem;
        width:100%;
    }

    /* Keep all five secondary-navigation buttons vertically aligned.
       The marker is only used by the phone-only :has() selector and
       must not create extra vertical space on desktop/tablet. */
    .aqua-home-subnav-marker {
        display:none !important;
    }

    /* The hidden mobile marker's Streamlit wrapper was leaving an empty
       block beneath the desktop secondary-navigation row. Remove that
       wrapper too, while retaining the marker in the DOM for :has() rules. */
    div[data-testid="stElementContainer"]:has(.aqua-home-subnav-marker) {
        display:none !important;
    }

    /* Compact the nav-to-heading gap only on pages that show the secondary nav. */
    @media (min-width: 769px) {
        body:has(.aqua-home-subnav-marker)
        .main .block-container > div[data-testid="stVerticalBlock"]
        > div[data-testid="stElementContainer"]:has(h1) {
            margin-top:-.6rem !important;
        }

        body:has(.aqua-home-subnav-marker) h1 {
            margin-top:.15rem !important;
        }
    }

    .aqua-reference-brand { display:flex; align-items:center; gap:.62rem; min-height:42px; margin-top:0 !important; }
    .aqua-reference-logo { width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center; background:linear-gradient(145deg,#18c5d2 0%,#12a9db 100%); box-shadow:0 4px 12px rgba(14,165,233,.20); overflow:hidden; flex:0 0 auto; }
    .aqua-reference-logo svg { width:24px; height:24px; }
    .aqua-reference-name { font-size:1.24rem; font-weight:800; letter-spacing:-.045em; color:#183b56; white-space:nowrap; }

    /* Mobile navigation is hidden by default and only enabled by the
       narrow-screen media query below. */
    div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker),
    div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) {
        display:none !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button { min-height:2.35rem !important; min-width:0 !important; width:100% !important; padding:.38rem .42rem !important; background:#fff !important; color:#587286 !important; border:1px solid transparent !important; border-radius:11px !important; box-shadow:none !important; font-weight:600 !important; font-size:.78rem !important; white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button p { white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover { background:#f0f8fc !important; color:#087da4 !important; border-color:#d7ebf3 !important; transform:none !important; box-shadow:none !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] { background:#edf7ff !important; color:#0879b2 !important; border-color:transparent !important; box-shadow:none !important; }

    .hero-card { position:relative; overflow:hidden; min-height:424px; padding:3.35rem 3.2rem; border-radius:28px; background:linear-gradient(125deg,#083c69 0%,#0877bb 53%,#16b8c7 100%); box-shadow:0 20px 46px rgba(12,84,120,.18); color:white; margin-top:1.1rem; margin-bottom:1.75rem; box-sizing:border-box; }
    /* Add reliable space between the desktop header row and the homepage hero.
       Put the spacing on Streamlit's outer element wrapper so it cannot collapse. */
    @media (min-width:769px) {
        .main .block-container div[data-testid="stElementContainer"]:has(.hero-card) {
            padding-top:1.5rem !important;
        }
        .main .block-container div[data-testid="stElementContainer"] .hero-card {
            margin-top:0 !important;
        }
    }
    .hero-card:before { content:""; position:absolute; width:360px; height:360px; right:-95px; top:-145px; border-radius:50%; background:rgba(255,255,255,.075); }
    .hero-card:after { content:""; position:absolute; width:210px; height:210px; right:65px; top:65px; border-radius:50%; background:rgba(255,255,255,.035); }
    .hero-badge { display:inline-flex; align-items:center; gap:.38rem; padding:.43rem .78rem; border-radius:10px; background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.18); font-size:.72rem; font-weight:700; color:#f0fcff; }
    .hero-title { margin-top:1.35rem; max-width:590px; font-size:3.55rem; line-height:1; letter-spacing:-.065em; font-weight:800; color:white; }
    .hero-title span { color:#aeeeff; }
    .hero-text { max-width:570px; margin-top:1.15rem; color:rgba(255,255,255,.83); line-height:1.65; font-size:.93rem; }
    .hero-about { position:relative; z-index:1; max-width:680px; margin-top:1.1rem; }
    .hero-about-title { margin:0 0 .38rem; color:#fff; font-size:1.02rem; line-height:1.25; font-weight:800; }
    .hero-about-text { max-width:650px; margin:0; color:rgba(255,255,255,.86); font-size:.82rem; line-height:1.5; }
    .hero-capabilities { display:flex; flex-wrap:wrap; gap:.42rem; margin-top:.62rem; max-width:650px; }
    .hero-capabilities span { display:inline-flex; align-items:center; border:1px solid rgba(255,255,255,.24); border-radius:999px; background:rgba(255,255,255,.10); padding:.28rem .58rem; color:#f5fcff; font-size:.70rem; line-height:1.25; }
    .hero-note { max-width:650px; margin-top:.58rem; color:rgba(255,255,255,.70); font-size:.70rem; line-height:1.4; }
    .hero-mini { position:absolute; right:10%; top:50%; transform:translateY(-42%); width:112px; height:112px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,.105); border:1px solid rgba(255,255,255,.23); font-size:3.1rem; z-index:2; }

    div[data-testid="stMetric"] { background:#fff; border:1px solid #dbe8ef; border-top:3px solid #13abc5; border-radius:19px; padding:1.05rem 1.15rem; box-shadow:0 10px 25px rgba(37,82,107,.075); min-height:112px; }
    div[data-testid="stMetricLabel"] { color:#78909f !important; font-weight:600; }
    div[data-testid="stMetricValue"] { color:#2b5a7d !important; font-weight:800; }
    .feature-card {
        height:152px;
        min-height:152px;
        box-sizing:border-box;
        padding:1.15rem 1.2rem;
        border-radius:17px;
        border:1px solid #dce9ef;
        background:#fff;
        box-shadow:0 8px 23px rgba(37,82,107,.065);
        display:flex;
        flex-direction:column;
        align-items:flex-start;
    }
    .system-status-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0 1.2rem; }
    .system-status-card { min-height:108px; box-sizing:border-box; padding:1rem 1.1rem; border:1px solid #dce9ef; border-radius:17px; background:#fff; box-shadow:0 8px 23px rgba(37,82,107,.055); }
    .system-status-label { font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#78909f; }
    .system-status-value { margin-top:.35rem; font-size:1.05rem; font-weight:800; color:#173b56; }
    .system-status-detail { margin-top:.25rem; font-size:.78rem; color:#678095; line-height:1.45; }
    @media (max-width: 760px) { .system-status-grid { grid-template-columns:1fr; } }

    .feature-icon {
        height:30px;
        min-height:30px;
        display:flex;
        align-items:center;
        font-size:1.35rem;
        line-height:1;
        margin-bottom:.5rem;
    }
    .feature-title {
        min-height:24px;
        display:flex;
        align-items:center;
        font-weight:750;
        line-height:1.2;
        color:#173b56;
        margin-bottom:.28rem;
    }
    .feature-text {
        margin:0;
        font-size:.83rem;
        line-height:1.48;
        color:#718696;
        flex:1;
    }
    .home-disclaimer {
        margin-top:16px;
        padding:12px 16px;
        border-radius:10px;
        background:#e8f3ff;
        color:#52779a;
        font-size:.84rem;
        line-height:1.5;
        border:1px solid #d7e9f8;
        box-sizing:border-box;
    }
    .section-heading { display:flex; align-items:center; gap:.62rem; margin:1.35rem 0 .85rem; font-size:1.12rem; font-weight:800; color:#173b56; }
    .section-heading:before { content:""; width:5px; height:24px; border-radius:5px; background:#11b3ca; }
    div[data-baseweb="select"] > div,div[data-baseweb="input"] > div,div[data-baseweb="textarea"] > div { border-radius:11px !important; border-color:#d5e3ea !important; background:#fff !important; }
    .stButton > button,.stDownloadButton > button,.stFormSubmitButton > button { border-radius:11px; min-height:2.55rem; padding:.52rem 1rem; font-weight:700; border:1px solid #c7e2ec; background:linear-gradient(135deg,#0e9ed3,#12b5c1); color:white; box-shadow:0 7px 18px rgba(14,158,211,.14); }
    div[data-testid="stAlert"] { border-radius:13px; border-width:1px; }
    div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid #d5e4eb; }
    div[data-testid="stForm"] { border:1px solid #dbe8ef; border-radius:18px; background:#fff; padding:1.2rem 1.3rem; }
    div[data-testid="stFormSubmitButton"] { margin-top:.68rem !important; }
    section[data-testid="stFileUploaderDropzone"] { border-radius:15px; border:1.5px dashed #79bfd3; background:#f8fdff; }
    button[data-baseweb="tab"] { font-weight:700; }
    hr { border:none; border-top:1px solid #dbe8ef; margin:1.5rem 0; }

    .aqua-footer {
        width:min(var(--aqua-content-width), calc(100vw - 2rem));
        max-width:none !important;
        position:relative;
        left:50%;
        transform:translateX(-50%);
        margin:1.5rem 0 0;
        padding:.75rem 1.2rem 1rem;
        text-align:center;
        color:#8195a3;
        font-size:.8rem;
        border-top:1px solid #dbe8ef;
        box-sizing:border-box;
    }

    /* ============================================================
       FINAL MUNICIPAL UI POLISH
       Consistent hierarchy, surfaces, controls and data presentation.
       ============================================================ */
    .main .block-container > div:first-child { padding-top:.35rem !important; }
    h1 {
        font-size:clamp(1.85rem, 3vw, 2.55rem) !important;
        line-height:1.08 !important;
        margin:1.05rem 0 .35rem !important;
        position:relative;
    }
    h1::after {
        content:""; display:block; width:54px; height:4px; margin-top:.58rem;
        border-radius:10px; background:#11abc8;
    }
    h2 { font-size:1.45rem !important; margin-top:1.25rem !important; }
    h3 { font-size:1.12rem !important; }
    .stMarkdown p { line-height:1.62; }

    /* Government portal navigation */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        transition:background .16s ease, color .16s ease, border-color .16s ease, transform .16s ease !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background:linear-gradient(180deg,#eef9ff,#e6f5fc) !important;
        border:1px solid #cce9f3 !important;
        color:#0679a8 !important;
        box-shadow:0 3px 10px rgba(9,123,163,.07) !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover { transform:translateY(-1px) !important; }

    /* Inputs and forms */
    div[data-testid="stForm"] {
        border:1px solid #d9e8ee !important;
        border-radius:18px !important;
        background:rgba(255,255,255,.82) !important;
        padding:1.15rem 1.25rem !important;
        box-shadow:0 9px 25px rgba(37,82,107,.055) !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background:#fff !important;
        border:1px solid #d4e3e9 !important;
        box-shadow:0 1px 2px rgba(37,82,107,.025) !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        border-color:#79c9dc !important;
        box-shadow:0 0 0 3px rgba(17,179,202,.10) !important;
    }
    .stButton > button, .stDownloadButton > button,
    .stFormSubmitButton > button {
        border-radius:11px !important;
        font-weight:700 !important;
        min-height:2.5rem !important;
    }
    .stFormSubmitButton > button[kind="primary"], .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#087bb5,#12a9bf) !important;
        border-color:#0877a9 !important;
        color:#fff !important;
        box-shadow:0 7px 17px rgba(8,123,181,.18) !important;
    }
    div:has(.citizen-submit-report-marker) + div .stButton > button, div:has(.citizen-submit-report-marker) + div .stButton > button p, div:has(.citizen-submit-report-marker) + div .stButton > button span, div:has(.citizen-submit-report-marker) + div .stButton > button div { color:#fff !important; font-weight:700 !important; }
    div:has(.report-simple-water-problem-marker) + div .stButton > button,
    div:has(.report-simple-water-problem-marker) + div .stButton > button p,
    div:has(.report-simple-water-problem-marker) + div .stButton > button span,
    div:has(.report-simple-water-problem-marker) + div .stButton > button div {
        color:#fff !important;
        font-weight:700 !important;
    }

    .citizen-info-grid {
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:12px;
        margin:0 0 1.2rem 0;
    }
    .citizen-info-card {
        background:#fff;
        border:1px solid #dce9ef;
        border-radius:14px;
        padding:1rem 1.05rem;
        box-shadow:0 5px 16px rgba(37,82,107,.045);
    }
    .citizen-info-title {
        color:#244d66;
        font-weight:700;
        font-size:.92rem;
        margin-bottom:.35rem;
    }
    .citizen-info-text {
        color:#587286;
        font-size:.82rem;
        line-height:1.45;
    }
    @media (max-width:900px) {
        .citizen-info-grid { grid-template-columns:1fr 1fr; }
    }
    @media (max-width:600px) {
        .citizen-info-grid { grid-template-columns:1fr; }
    }

    /* Status / notification surfaces */
    div[data-testid="stAlert"] {
        border-radius:14px !important;
        border-width:1px !important;
        box-shadow:0 5px 16px rgba(37,82,107,.045) !important;
    }
    div[data-testid="stExpander"] {
        border:1px solid #dbe8ee !important;
        border-radius:15px !important;
        background:#fff !important;
        overflow:hidden !important;
    }
    div[data-testid="stExpander"] summary { font-weight:700 !important; color:#244d66 !important; }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border:1px solid #dce9ef !important;
        border-radius:14px !important;
        overflow:hidden !important;
        box-shadow:0 6px 18px rgba(37,82,107,.045) !important;
    }
    div[data-testid="stTable"] {
        border:1px solid #dce9ef !important;
        border-radius:14px !important;
        overflow:hidden !important;
    }

    /* Charts sit on a clean portal surface */
    div[data-testid="stPlotlyChart"] {
        border:1px solid #e0ebf0; border-radius:16px; background:#fff;
        padding:.15rem .15rem 0; box-shadow:0 7px 20px rgba(37,82,107,.045);
        overflow:hidden;
    }

    /* KPI cards: stronger hierarchy without looking like a gaming dashboard */
    div[data-testid="stMetric"] {
        position:relative; overflow:hidden;
        border-top:3px solid #11abc8 !important;
    }
    div[data-testid="stMetricLabel"] p { font-size:.74rem !important; letter-spacing:.035em; text-transform:uppercase; }
    div[data-testid="stMetricValue"] { font-size:1.65rem !important; }

    /* Section headings created by the existing section() helper */
    .section-heading {
        margin-top:1.55rem !important;
        margin-bottom:.9rem !important;
        padding-bottom:.55rem;
        border-bottom:1px solid #e1edf2;
    }

    /* Shared footer */
    .aqua-footer {
        border-top:1px solid #d9e8ee !important;
        background:rgba(255,255,255,.72) !important;
        color:#78909f !important;
        backdrop-filter:blur(7px);
    }

    /* Forecast page only: reduce the visible blank space below the shared footer.
       Other pages keep the original footer position unchanged. */
    body:has(.forecast-footer-space-marker) .aqua-footer {
        top:28px !important;
    }
    .feature-card, div[data-testid="stMetric"], div[data-testid="stForm"] { transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
    .feature-card:hover { transform:translateY(-2px); box-shadow:0 12px 28px rgba(37,82,107,.10); border-color:#c7e4ee; }
    @media (max-width:1100px) {
        .aqua-reference-name { font-size:1.05rem; }
        div[data-testid="stHorizontalBlock"] .stButton > button { font-size:.72rem !important; padding:.34rem .42rem !important; }
    }

    /* Tablet / small-laptop safety: keep every page inside the viewport and
       let dense Streamlit rows breathe instead of creating horizontal overflow. */
    @media (max-width:900px) {
        .main .block-container {
            width:100% !important;
            max-width:100% !important;
            padding-left:.85rem !important;
            padding-right:.85rem !important;
            box-sizing:border-box !important;
        }
        .aqua-reference-name { font-size:1.05rem; }
        .hero-card { min-height:360px; padding:2.3rem 2rem; }
        .hero-title { font-size:2.55rem; }
        .hero-mini { display:none; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            max-width:100% !important;
            overflow-x:auto !important;
        }
        .stDownloadButton, .stButton, .stFormSubmitButton {
            max-width:100% !important;
        }
    }

    /* MOBILE ONLY — isolated phone UI. Desktop/tablet CSS above is untouched. */
    @media (max-width:600px) {
        div[data-testid="stPlotlyChart"] {
            margin-top:.6rem !important;
            padding-top:.5rem !important;
            overflow:hidden !important;
            overflow-x:hidden !important;
            overflow-y:hidden !important;
            max-height:none !important;
        }
        div[data-testid="stPlotlyChart"] .js-plotly-plot {
            margin-top:4px !important;
            overflow:visible !important;
        }
        div[data-testid="stPlotlyChart"] > div,
        div[data-testid="stPlotlyChart"] iframe {
            overflow:hidden !important;
            max-height:none !important;
        }
        div[data-testid="stPlotlyChart"] .gtitle {
            transform:translateY(6px) !important;
        }
        .section-heading {
            margin-bottom:1rem !important;
        }


        /* ---------- MOBILE ALIGNMENT PASS ---------- */

        /* Header: keep kebab, logo and language button centered on one clean row. */
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) {
            grid-template-columns:48px minmax(0,1fr) 112px !important;
            gap:.6rem !important;
            align-items:center !important;
            min-height:64px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child {
            display:flex !important;
            align-items:center !important;
            justify-content:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child .stButton {
            width:48px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child .stButton > button {
            width:48px !important;
            height:48px !important;
            min-height:48px !important;
            padding:0 !important;
            display:flex !important;
            align-items:center !important;
            justify-content:center !important;
            font-size:2rem !important;
            line-height:1 !important;
            font-weight:700 !important;
            color:#173b56 !important;
            background:#fff !important;
            border:1px solid #d7e8ef !important;
            border-radius:15px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:nth-child(2) {
            display:flex !important;
            justify-content:center !important;
            min-width:0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand {
            width:100% !important;
            justify-content:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child {
            display:flex !important;
            align-items:center !important;
            justify-content:flex-end !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child .stButton {
            width:112px !important;
        }

        /* How AquaTrack Works: four complete cards in one horizontal swipe row.
           Each card gets enough width so its title/body does not collide or squeeze. */
        div[data-testid="stHorizontalBlock"]:has(.feature-card) {
            display:flex !important;
            flex-direction:row !important;
            flex-wrap:nowrap !important;
            gap:.75rem !important;
            width:100% !important;
            overflow-x:auto !important;
            overflow-y:hidden !important;
            padding:.15rem .05rem .75rem !important;
            scroll-snap-type:x mandatory !important;
            -webkit-overflow-scrolling:touch !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) > div {
            flex:0 0 78vw !important;
            width:78vw !important;
            min-width:78vw !important;
            max-width:78vw !important;
            scroll-snap-align:start !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) .feature-card {
            width:100% !important;
            min-height:300px !important;
            height:300px !important;
            padding:1.05rem !important;
            box-sizing:border-box !important;
            overflow:hidden !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) .feature-title {
            font-size:1.08rem !important;
            line-height:1.2 !important;
            min-height:auto !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) .feature-text {
            font-size:.9rem !important;
            line-height:1.48 !important;
            overflow-wrap:anywhere !important;
        }

        /* General mobile chart safety: never let two charts squeeze into one phone row. */
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) {
            grid-template-columns:minmax(0,1fr) !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) > div {
            width:100% !important;
            min-width:0 !important;
        }

        /* Give the sections consistent vertical breathing room. */
        .dashboard-insight {
            margin:.15rem 0 1rem !important;
            padding:.85rem 1rem !important;
            border:1px solid #d9e9ef !important;
            border-left:4px solid #12a9db !important;
            border-radius:10px !important;
            background:#f8fcfe !important;
            color:#35566a !important;
            font-size:.92rem !important;
            line-height:1.55 !important;
        }
        .dashboard-insight strong {
            color:#173b56 !important;
        }

        .section-heading {
            margin-top:1.35rem !important;
            margin-bottom:.75rem !important;
            font-size:1.12rem !important;
        }
        .main .block-container {
            width:100% !important;
            max-width:none !important;
            padding-left:.65rem !important;
            padding-right:.65rem !important;
            padding-bottom:1.2rem !important;
            box-sizing:border-box !important;
        }

        /* Hide the desktop navigation and the old secondary navigation only on phones. */
        div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker),
        div[data-testid="stVerticalBlock"]:has(.aqua-subnav-gap) > div[data-testid="stHorizontalBlock"] {
            display:none !important;
        }

        /* Mobile top bar: kebab | AquaTrack | Hindi. */
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) {
            display:grid !important;
            grid-template-columns:112px minmax(0,1fr) 112px !important;
            gap:.55rem !important;
            align-items:center !important;
            width:100% !important;
            margin:0 0 .7rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div {
            width:auto !important;
            min-width:0 !important;
            display:flex !important;
            align-items:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child {
            justify-content:center !important;
            justify-self:center !important;
            width:92px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:nth-child(2) {
            justify-content:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child {
            justify-content:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .stButton > button {
            min-height:3rem !important;
            width:100% !important;
            padding:.35rem .5rem !important;
            border-radius:14px !important;
            background:#fff !important;
            color:#0879b2 !important;
            border:1px solid #d7e8ef !important;
            box-shadow:0 4px 12px rgba(37,82,107,.06) !important;
            font-size:.78rem !important;
            white-space:nowrap !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child .stButton > button {
            font-size:1.55rem !important;
            font-weight:800 !important;
            line-height:1 !important;
            padding:0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child .stButton > button {
            font-size:.76rem !important;
        }
        .aqua-mobile-brand {
            justify-content:center !important;
            align-items:center !important;
            min-height:52px !important;
            padding:.15rem 0 !important;
        }
        .aqua-mobile-brand {
            min-height:52px !important;
            padding:.15rem 0 !important;
        }
        .aqua-mobile-tagline {
            font-size:.58rem !important;
            color:#8aa0ad !important;
            font-weight:600 !important;
            letter-spacing:.01em !important;
            margin-top:-2px !important;
        }
        .aqua-mobile-brand .aqua-reference-logo {
            width:42px !important;
            height:42px !important;
        }
        .aqua-mobile-brand .aqua-reference-name {
            font-size:1.15rem !important;
        }

        /* Kebab drawer: mobile only, styled like a clean account/settings panel. */
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) {
            display:flex !important;
            position:fixed !important;
            top:0 !important;
            left:0 !important;
            bottom:0 !important;
            width:min(82vw,340px) !important;
            max-width:340px !important;
            min-width:270px !important;
            height:100vh !important;
            height:100dvh !important;
            z-index:999999 !important;
            padding:3.35rem .8rem .45rem !important;
            margin:0 !important;
            box-sizing:border-box !important;
            overflow:hidden !important;
            overflow-y:hidden !important;
            background:rgba(255,255,255,.98) !important;
            border-right:1px solid #d7e8ef !important;
            box-shadow:18px 0 45px rgba(20,65,90,.18) !important;
            border-radius:0 22px 22px 0 !important;
            gap:.18rem !important;
        }

        /* When the kebab menu is open, lock the page behind it.
           The menu itself remains fixed and does not scroll. */
        html:has(.aqua-mobile-menu-marker),
        body:has(.aqua-mobile-menu-marker),
        body:has(.aqua-mobile-menu-marker) #root,
        body:has(.aqua-mobile-menu-marker) .stApp,
        body:has(.aqua-mobile-menu-marker) .stAppViewContainer,
        body:has(.aqua-mobile-menu-marker) section[data-testid="stMain"] {
            overflow:hidden !important;
            overscroll-behavior:none !important;
        }
        /* Mobile drawer header: × and AquaTrack Menu share one normal-flow row. */
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .stButton:has(button[key="mobile_menu_close"]) {
            position:relative !important;
            top:0 !important;
            left:0 !important;
            width:2.25rem !important;
            display:inline-block !important;
            margin:0 .6rem 0 0 !important;
            vertical-align:middle !important;
        }
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .stButton:has(button[key="mobile_menu_close"]) > button {
            min-height:2.25rem !important;
            width:2.25rem !important;
            padding:0 !important;
            border-radius:50% !important;
            justify-content:center !important;
            text-align:center !important;
            background:#fff !important;
            color:#183b56 !important;
            border:1px solid #dceaf0 !important;
            box-shadow:none !important;
            font-size:1.35rem !important;
            line-height:1 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .aqua-mobile-drawer-title {
            display:inline-block !important;
            margin:0 0 .6rem 0 !important;
            padding:.3rem 0 .4rem !important;
            vertical-align:middle !important;
            border-bottom:1px solid #e5eef3 !important;
            width:calc(100% - 3rem) !important;
            color:#183b56 !important;
            font-size:1rem !important;
            font-weight:800 !important;
            line-height:1.4 !important;
            white-space:nowrap !important;
            box-sizing:border-box !important;
        }
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .stButton {
            margin:0 0 .08rem !important;
        }
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .stButton > button {
            min-height:2.5rem !important;
            height:2.5rem !important;
            width:100% !important;
            text-align:left !important;
            justify-content:flex-start !important;
            padding:.35rem .7rem !important;
            border-radius:12px !important;
            background:#fff !important;
            color:#587286 !important;
            border:1px solid #e1edf2 !important;
            box-shadow:none !important;
            font-size:.84rem !important;
            white-space:normal !important;
        }
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .stButton > button:hover,
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .stButton > button[kind="primary"] {
            background:#edf8ff !important;
            color:#0879b2 !important;
            border-color:#cfe8f3 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.aqua-mobile-menu-marker) .stButton:has(button[key="mobile_menu_close"]) > button:hover {
            background:#fff !important;
            color:#183b56 !important;
            border-color:#dceaf0 !important;
        }

        /* Keep the existing phone content readable without changing desktop. */
        .hero-card {
            min-height:330px !important;
            padding:2rem 1.25rem !important;
            border-radius:22px !important;
            margin-bottom:1rem !important;
        }
        .hero-title {
            max-width:100% !important;
            font-size:2.15rem !important;
            line-height:1.05 !important;
        }
        .hero-text {
            max-width:100% !important;
            font-size:.86rem !important;
            line-height:1.55 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) {
            display:grid !important;
            grid-template-columns:repeat(2,minmax(0,1fr)) !important;
            gap:.7rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) > div {
            width:auto !important;
            min-width:0 !important;
        }
        div[data-testid="stMetric"] {
            min-height:96px !important;
            padding:.8rem .75rem !important;
            border-radius:15px !important;
        }
        div[data-testid="stMetricValue"] { font-size:1.45rem !important; }
        .feature-card { height:auto !important; min-height:140px !important; }
        .system-status-grid { grid-template-columns:1fr !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            max-width:100% !important;
            overflow-x:auto !important;
        }
        .aqua-footer {
            width:100% !important;
            font-size:.72rem !important;
            padding:.7rem .5rem .9rem !important;
        }


    /* Water-check method selector: keep the question on the left and the two choices
       in a clean, equal-width row. This is scoped only to the report/no-report selector. */
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) {
        margin-bottom:.65rem !important;
    }
    /* Keep the water-check mode selector wide, simple and readable. */
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] {
        width:100% !important;
        display:block !important;
        box-sizing:border-box !important;
        margin:0 !important;
    }
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] > label {
        display:block !important;
        width:100% !important;
        min-width:0 !important;
        margin:0 0 .35rem !important;
        padding:.25rem 0 !important;
        color:#4f6d80 !important;
        font-size:.9rem !important;
        line-height:1.35 !important;
        white-space:normal !important;
    }
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] > label p {
        margin:0 !important;
        white-space:normal !important;
    }
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] div[role="radiogroup"] {
        display:flex !important;
        width:100% !important;
        min-width:0 !important;
        gap:1.25rem !important;
        flex-wrap:nowrap !important;
        align-items:center !important;
    }
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex:0 0 auto !important;
        width:auto !important;
        min-width:0 !important;
        height:auto !important;
        box-sizing:border-box !important;
        padding:.2rem 0 !important;
        margin:0 !important;
        border:0 !important;
        border-radius:0 !important;
        background:transparent !important;
        color:#36566c !important;
        display:flex !important;
        align-items:center !important;
        justify-content:flex-start !important;
        white-space:nowrap !important;
        overflow:visible !important;
        box-shadow:none !important;
    }
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background:transparent !important;
        border:0 !important;
        color:#36566c !important;
        box-shadow:none !important;
    }

    /* Public-friendly water check: question on its own line, Yes/No directly below. */
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) {
        margin:0 0 .65rem !important;
        padding:0 !important;
        border:none !important;
        border-radius:0 !important;
        background:transparent !important;
        box-shadow:none !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] {
        display:block !important;
        width:100% !important;
        box-sizing:border-box !important;
        padding:.15rem 0 !important;
        border:none !important;
        border-radius:0 !important;
        background:transparent !important;
        box-shadow:none !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] > label {
        display:block !important;
        width:100% !important;
        min-width:0 !important;
        margin:0 0 .35rem !important;
        padding:0 !important;
        color:#173f5f !important;
        font-size:.9rem !important;
        font-weight:600 !important;
        line-height:1.35 !important;
        white-space:normal !important;
        overflow:visible !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] > label p {
        margin:0 !important;
        white-space:normal !important;
        overflow:visible !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] div[role="radiogroup"] {
        display:flex !important;
        width:auto !important;
        min-width:0 !important;
        gap:1.25rem !important;
        flex-wrap:nowrap !important;
        align-items:center !important;
        justify-content:flex-start !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex:0 0 auto !important;
        width:auto !important;
        min-width:0 !important;
        height:auto !important;
        box-sizing:border-box !important;
        padding:.15rem 0 !important;
        border:0 !important;
        border-radius:0 !important;
        background:transparent !important;
        color:#36566c !important;
        cursor:pointer !important;
        display:flex !important;
        align-items:center !important;
        justify-content:flex-start !important;
        margin:0 !important;
        white-space:nowrap !important;
        overflow:visible !important;
        box-shadow:none !important;
    }
    /* Keep one clean, native-looking white radio circle per answer. */
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        flex:0 0 auto !important;
        margin-right:.35rem !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] div[role="radiogroup"] > label:before {
        content:none !important;
        display:none !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background:transparent !important;
        border:0 !important;
        color:#36566c !important;
        box-shadow:none !important;
    }
    body:has(.without-report-marker) .simple-water-hazard {
        margin-top:.85rem !important;
        margin-bottom:1rem !important;
        padding:1rem 1.1rem !important;
        border:1px solid #f3b6b6 !important;
        border-left:5px solid #dc2626 !important;
        border-radius:13px !important;
        background:#fff5f5 !important;
        color:#7f1d1d !important;
    }
    /* QUESTIONS PAGE — phone only:
       one question per row, with Yes / No directly underneath. */
    body:has(.without-report-marker) div[data-testid="stHorizontalBlock"]:has(.simple-water-item-marker) {
        display:block !important;
        width:100% !important;
        margin:0 !important;
        padding:0 !important;
    }
    body:has(.without-report-marker) div[data-testid="stHorizontalBlock"]:has(.simple-water-item-marker) > div {
        display:block !important;
        width:100% !important;
        max-width:100% !important;
        min-width:0 !important;
        flex:0 0 100% !important;
        margin:0 !important;
        padding:0 !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) {
        width:100% !important;
        margin:0 0 .55rem !important;
        padding:0 !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] > label {
        margin:0 0 .25rem !important;
        font-size:.9rem !important;
        line-height:1.3 !important;
    }
    body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] div[role="radiogroup"] {
        display:flex !important;
        flex-direction:row !important;
        flex-wrap:nowrap !important;
        gap:1.15rem !important;
        align-items:center !important;
        justify-content:flex-start !important;
    }

    /* Report/no-report choice: stack the two choices vertically on phones. */
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] div[role="radiogroup"] {
        display:flex !important;
        flex-direction:column !important;
        align-items:flex-start !important;
        justify-content:flex-start !important;
        width:100% !important;
        gap:.15rem !important;
    }
    body:has(.quality-check-mode-marker) div[data-testid="stVerticalBlock"]:has(.quality-check-mode-marker) div[data-testid="stRadio"] div[role="radiogroup"] > label {
        width:100% !important;
        flex:0 0 auto !important;
        margin:0 !important;
        padding:.12rem 0 !important;
    }

    @media (max-width:520px) {
        body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] > label {
            font-size:.88rem !important;
        }
        body:has(.without-report-marker) div[data-testid="stVerticalBlock"]:has(.simple-water-item-marker) div[data-testid="stRadio"] div[role="radiogroup"] {
            gap:1rem !important;
        }
    }

    @media (max-width:380px) {
        div[data-testid="stHorizontalBlock"]:has(.aqua-reference-brand) > div:not(:first-child) {
            flex-basis:calc(50% - .45rem) !important;
            width:calc(50% - .45rem) !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-reference-brand) .stButton > button {
            font-size:.66rem !important;
        }
        div[data-testid="stHorizontalBlock"]:not(:has(.aqua-reference-brand)):has(.stButton) > div {
            flex:1 1 calc(50% - .55rem) !important;
        }
        .hero-title { font-size:1.9rem !important; }
    }

        /* =========================================================
           FINAL MOBILE LAYOUT — V2
           Phone only. Desktop/tablet rules remain unchanged.
           ========================================================= */

        /* Header: perfectly centered  |  kebab | AquaTrack | Hindi.
           The two outer columns are equal, so the AquaTrack brand is centered
           on the actual phone viewport rather than being shifted left. */
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) {
            display:grid !important;
            grid-template-columns:92px minmax(0,1fr) 92px !important;
            gap:.35rem !important;
            align-items:center !important;
            justify-items:center !important;
            width:100% !important;
            min-height:54px !important;
            margin:0 0 .65rem !important;
            box-sizing:border-box !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div {
            width:auto !important;
            min-width:0 !important;
            display:flex !important;
            align-items:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child {
            justify-content:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child .stButton,
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child .stButton > button {
            width:44px !important;
            min-width:44px !important;
            max-width:44px !important;
            height:44px !important;
            min-height:44px !important;
            padding:0 !important;
            margin:0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:first-child .stButton > button {
            display:flex !important;
            align-items:center !important;
            justify-content:center !important;
            font-size:1.65rem !important;
            line-height:1 !important;
            font-weight:800 !important;
            color:#173b56 !important;
            background:#fff !important;
            border:1px solid #d7e8ef !important;
            border-radius:14px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:nth-child(2) {
            justify-content:center !important;
            justify-self:center !important;
            width:100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand {
            width:max-content !important;
            max-width:100% !important;
            height:42px !important;
            min-height:42px !important;
            justify-content:center !important;
            align-items:center !important;
            padding:0 !important;
            margin:0 auto !important;
            gap:.42rem !important;
            box-sizing:border-box !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand .aqua-reference-logo {
            width:34px !important;
            height:34px !important;
            flex:0 0 34px !important;
            align-self:center !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand .aqua-reference-name {
            font-size:1rem !important;
            line-height:1 !important;
            white-space:nowrap !important;
            letter-spacing:-.02em !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-tagline {
            display:none !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child {
            justify-content:center !important;
            justify-self:center !important;
            width:92px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child .stButton,
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child .stButton > button {
            width:92px !important;
            min-width:92px !important;
            max-width:92px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div:last-child .stButton > button {
            min-height:44px !important;
            height:44px !important;
            padding:0 .25rem !important;
            font-size:.69rem !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
        }

        /* How AquaTrack Works: STACK all four cards vertically.
           Each card receives the full phone content width and its own
           complete vertical space. */
        div[data-testid="stHorizontalBlock"]:has(.feature-card) {
            display:flex !important;
            flex-direction:column !important;
            flex-wrap:nowrap !important;
            gap:1.2rem !important;
            width:100% !important;
            overflow:visible !important;
            padding:.1rem 0 .35rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) > div {
            flex:0 0 auto !important;
            width:100% !important;
            min-width:0 !important;
            max-width:none !important;
            box-sizing:border-box !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) .feature-card {
            width:100% !important;
            min-height:0 !important;
            height:auto !important;
            padding:1rem 1.05rem !important;
            box-sizing:border-box !important;
            overflow:visible !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) .feature-title {
            min-height:0 !important;
            height:auto !important;
            font-size:1.12rem !important;
            line-height:1.25 !important;
            white-space:normal !important;
            overflow-wrap:break-word !important;
            word-break:normal !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-card) .feature-text {
            font-size:.94rem !important;
            line-height:1.55 !important;
            white-space:normal !important;
            overflow-wrap:break-word !important;
            word-break:normal !important;
        }

        /* Water Availability Overview:
           chart 1 = full-width block, chart 2 = full-width block below it. */
        div[data-testid="stHorizontalBlock"]:has(.mobile-availability-marker) {
            display:block !important;
            width:100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.mobile-availability-marker) > div,
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) > div {
            display:block !important;
            width:100% !important;
            min-width:0 !important;
            max-width:none !important;
            flex:0 0 100% !important;
            box-sizing:border-box !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) {
            display:grid !important;
            grid-template-columns:minmax(0,1fr) !important;
            grid-auto-flow:row !important;
            gap:1rem !important;
            width:100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) div[data-testid="stPlotlyChart"] {
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            overflow:hidden !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) .stPlotlyChart,
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) iframe {
            max-width:100% !important;
        }

        /* Give each availability chart a comfortable phone height. */
        div[data-testid="stHorizontalBlock"]:has(.mobile-availability-marker) div[data-testid="stPlotlyChart"] {
            min-height:360px !important;
        }

        /* FINAL PHONE HEADER LOCK — one line, exact center. */
        @media (max-width:600px) {
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) {
                grid-template-columns:92px minmax(0,1fr) 92px !important;
                align-items:center !important;
                justify-items:center !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand {
                width:max-content !important;
                max-width:100% !important;
                margin:0 auto !important;
                justify-content:center !important;
                align-items:center !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-tagline {
                display:none !important;
            }
        }

        /* FINAL MOBILE HEADER ALIGNMENT ONLY — phone screens only.
           Keep the three header items on one exact horizontal center line:
           [ kebab ]   [ AquaTrack logo + name ]   [ Hindi ]
           Do not alter any other mobile content. */
        @media (max-width:600px) {
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) {
                display:grid !important;
                grid-template-columns:92px minmax(0,1fr) 92px !important;
                column-gap:0 !important;
                align-items:center !important;
                justify-items:center !important;
                width:100% !important;
                min-height:54px !important;
                margin:0 0 .65rem !important;
                padding:0 !important;
                box-sizing:border-box !important;
            }

            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"] {
                width:100% !important;
                min-width:0 !important;
                margin:0 !important;
                padding:0 !important;
                display:flex !important;
                align-items:center !important;
                justify-content:center !important;
                align-self:center !important;
                box-sizing:border-box !important;
            }

            /* Kebab — fixed size and exact vertical center. */
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:first-child .stButton,
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:first-child .stButton > button {
                width:44px !important;
                min-width:44px !important;
                max-width:44px !important;
                height:44px !important;
                min-height:44px !important;
                max-height:44px !important;
                margin:0 !important;
                padding:0 !important;
                box-sizing:border-box !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:first-child .stButton > button {
                display:flex !important;
                align-items:center !important;
                justify-content:center !important;
            }

            /* AquaTrack — centered on the actual phone viewport and vertically
               aligned to the exact center of the two side buttons. */
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:nth-child(2) {
                justify-content:center !important;
                align-items:center !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand {
                width:max-content !important;
                max-width:100% !important;
                height:44px !important;
                min-height:44px !important;
                margin:0 !important;
                padding:0 !important;
                display:flex !important;
                align-items:center !important;
                justify-content:center !important;
                gap:.42rem !important;
                box-sizing:border-box !important;
                transform:none !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand .aqua-reference-logo {
                width:34px !important;
                height:34px !important;
                flex:0 0 34px !important;
                margin:0 !important;
                align-self:center !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand > div:last-child {
                display:flex !important;
                flex-direction:column !important;
                justify-content:center !important;
                align-items:flex-start !important;
                margin:0 !important;
                padding:0 !important;
                height:44px !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-brand .aqua-reference-name {
                margin:0 !important;
                padding:0 !important;
                font-size:1rem !important;
                line-height:1 !important;
                white-space:nowrap !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .aqua-mobile-tagline {
                display:none !important;
            }

            /* Hindi — same fixed height and same vertical center as kebab. */
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:last-child {
                justify-content:center !important;
                align-items:center !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:last-child .stButton,
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:last-child .stButton > button {
                width:92px !important;
                min-width:92px !important;
                max-width:92px !important;
                height:44px !important;
                min-height:44px !important;
                max-height:44px !important;
                margin:0 !important;
                padding:0 .25rem !important;
                box-sizing:border-box !important;
            }
            div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) > div[data-testid="stColumn"]:last-child .stButton > button {
                display:flex !important;
                align-items:center !important;
                justify-content:center !important;
                white-space:nowrap !important;
                overflow:hidden !important;
                text-overflow:ellipsis !important;
                font-size:.69rem !important;
                line-height:1 !important;
            }
        }

        /* Plotly availability charts: keep the title and legend on separate
           visual space on narrow screens so they never overlap. */
        div[data-testid="stHorizontalBlock"]:has(.mobile-availability-marker) .js-plotly-plot g.gtitle text {
            font-size:18px !important;
            font-weight:700 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.mobile-availability-marker) .js-plotly-plot g.legend text {
            font-size:11px !important;
        }

        /* PHONE ONLY: hide the duplicate Home secondary-navigation row.
           Analytics / Compare / Quality / Forecast / Insights remain available
           in the working mobile sidebar. */
        div[data-testid="stHorizontalBlock"]:has(.aqua-home-subnav-marker) {
            display:none !important;
        }

    

    /* ============================================================
       CITIZEN PORTAL REDESIGN
       Compact municipal-service layout for the account experience.
       ============================================================ */
    .citizen-welcome { background:#ffffff; border:1px solid #d8e6ec; border-left:4px solid #10a9c4; border-radius:16px; padding:1.25rem 1.4rem; margin:.7rem 0 1rem; box-shadow:0 7px 20px rgba(37,82,107,.045); }
    .citizen-eyebrow { color:#0b8eaa; font-size:.7rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .citizen-title { color:#173b56; font-size:clamp(1.55rem,2.5vw,2.15rem); font-weight:800; letter-spacing:-.04em; margin-top:.25rem; }
    .citizen-subtitle { color:#6c8391; font-size:.9rem; margin-top:.25rem; line-height:1.5; }
    .citizen-section-title { color:#173b56; font-size:1.08rem; font-weight:800; margin:1.25rem 0 .55rem; }
    .citizen-report-row { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.85rem 1rem; background:#fff; border:1px solid #dce9ef; border-bottom:0; }
    .citizen-report-row:first-of-type { border-radius:12px 12px 0 0; }
    .citizen-report-row:last-of-type { border-bottom:1px solid #dce9ef; border-radius:0 0 12px 12px; }
    .citizen-report-row span { color:#6e8491; margin-left:.75rem; font-size:.86rem; }
    .citizen-status { color:#087ca0; font-size:.78rem; font-weight:800; }
    .account-summary-card { background:#fff; border:1px solid #dbe8ee; border-radius:15px; padding:1rem; margin-bottom:.8rem; }
    .account-summary-label { color:#8295a1; font-size:.67rem; font-weight:800; letter-spacing:.1em; margin-top:.5rem; }
    .account-summary-value { color:#173b56; font-weight:800; font-size:1rem; margin-top:.12rem; }
    .account-switch { text-align:center; color:#718796; font-size:.85rem; padding:.65rem 0; }
    .citizen-notification-row { display:flex; gap:.75rem; padding:.9rem 1rem; margin-bottom:.55rem; background:#fff; border:1px solid #dce9ef; border-radius:12px; }
    .citizen-notification-row.unread { border-left:3px solid #0da9c4; background:#f8fdff; }
    .citizen-notification-marker { color:#0b9fba; font-size:.8rem; padding-top:.1rem; }
    .citizen-notification-time { color:#8295a1; font-size:.72rem; margin-top:.2rem; }
    body:has(.citizen-account-page-marker) .main .block-container { padding-top:.4rem !important; }
    body:has(.citizen-account-page-marker) div[data-testid="stForm"] { max-width:760px; margin:0 auto; }
    @media (max-width:760px) {
        .citizen-report-row { align-items:flex-start; }
        .citizen-report-row span { display:block; margin:.2rem 0 0; }
        .citizen-welcome { padding:1rem; }
    }
    /* Compact header: clear hierarchy without pill-heavy controls. */
    div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker) .stButton > button {
        min-height:2.25rem !important; padding:.36rem .55rem !important; border-radius:9px !important;
        background:#fff !important; border:1px solid transparent !important; color:#587286 !important;
        box-shadow:none !important; font-size:.78rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker) .stButton > button:hover {
        background:#f3f9fc !important; color:#087da4 !important; border-color:#d7e8ef !important; transform:none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker) .stButton > button[kind="primary"] {
        background:#eaf7fb !important; color:#087da4 !important; border-color:#cde8ef !important; box-shadow:none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .stButton > button {
        border-radius:9px !important; box-shadow:none !important;
    }
    @media (max-width:760px) {
        div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker) .stButton > button { min-height:2.2rem !important; }
    }
</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BILINGUAL UI (English / colloquial Hindi — Hinglish, not shuddh Hindi)
# ============================================================
TRANSLATIONS = {"English": {
    "Alerts": "Alerts",
    "📢 Resident Water Alerts": "📢 Resident Water Alerts",
    "Register your area once. New water-problem reports for that zone will automatically appear in your in-app alerts.": "Register your area once. New water-problem reports for that zone will automatically appear in your in-app alerts.",
    "Your Zone": "Your Zone",
    "Your Name": "Your Name",
    "Phone Number": "Phone Number",
    "Email (for alerts)": "Email (for alerts)",
    "🔔 Register for Alerts": "🔔 Register for Alerts",
    "Please provide a phone number or email.": "Please provide a phone number or email.",
    "🔔 Your Recent Alerts": "🔔 Your Recent Alerts",
    "No alerts yet. New reports for your registered zone will appear here automatically.": "No alerts yet. New reports for your registered zone will appear here automatically.",
    "Registered Residents": "Registered Residents",
    "Alert Records": "Alert Records",
    "Email Deliveries": "Email Deliveries",
    "No residents have registered for alerts yet.": "No residents have registered for alerts yet.",
    "🔔 Resident Alerts": "🔔 Resident Alerts",
    "Alert Log": "Alert Log",
    "No alerts have been generated yet.": "No alerts have been generated yet.",
    "📢 Manual Broadcast": "📢 Manual Broadcast",
    "Alert message": "Alert message",
    "Write the message residents should receive...": "Write the message residents should receive...",
    "📢 Send Now": "📢 Send Now",
    "Please enter an alert message.": "Please enter an alert message.",
    "How would you like to check the water?": "How would you like to check the water?",
    "Yes": "Yes",
    "No": "No",
    "I have a water test report": "I have a water test report",
    "I do not have a water test report": "I do not have a water test report",
    "Use the values from your water test report. AquaTrack will analyze them using the water-quality model.": "Use the values from your water test report. AquaTrack will analyze them using the water-quality model.",
    "No report? You can do a simple visual check instead. This is only a preliminary check and not a laboratory test.": "No report? You can do a simple visual check instead. This is only a preliminary check and not a laboratory test.",
    "Check the water using simple questions": "Check the water using simple questions",
    "1. Is the water clear?": "1. Is the water clear?",
    "2. Can you see dirt or waste in the water?": "2. Can you see dirt or waste in the water?",
    "3. Does the water smell bad?": "3. Does the water smell bad?",
    "4. Does the water look yellow, brown or another unusual colour?": "4. Does the water look yellow, brown or another unusual colour?",
    "5. Does the water look very cloudy?": "5. Does the water look very cloudy?",
    "6. Does the water taste strange or bad?": "6. Does the water taste strange or bad?",
    "Do not intentionally taste water that may be unsafe.": "Do not intentionally taste water that may be unsafe.",
    "Check Water": "Check Water",
    "Preliminary Check Result": "Preliminary Check Result",
    "Water quality warning": "Water quality warning",
    "Several problems are visible in the water. Please do not drink it until the water has been properly tested.": "Several problems are visible in the water. Please do not drink it until the water has been properly tested.",
    "Please report this problem to the system so it can be reviewed.": "Please report this problem to the system so it can be reviewed.",
    "Report this problem to the system": "Report this problem to the system",
    "Opening the water issue report form.": "Opening the water issue report form.",
    "The water looks normal from the information provided.": "The water looks normal from the information provided.",
    "Some things about the water need attention.": "Some things about the water need attention.",
    "Several problems are visible in the water. Get the water tested before drinking it.": "Several problems are visible in the water. Get the water tested before drinking it.",
    "This is only a preliminary check based on what can be seen or noticed. It cannot measure things that are not visible.": "This is only a preliminary check based on what can be seen or noticed. It cannot measure things that are not visible.",
}, "हिंदी": {
     "About & Policies": "अबाउट और पॉलिसीज़", "About": "अबाउट",
     "Privacy Policy": "प्राइवेसी पॉलिसी", "Terms of Use": "यूज़ की शर्तें",
     "Data & Model Transparency": "डेटा और मॉडल ट्रांसपेरेंसी", "Project Scope": "प्रोजेक्ट स्कोप",
     "Contact & Support": "कॉन्टैक्ट और सपोर्ट",
    "🏠 Home": "🏠 होम", "📝 Report": "📝 रिपोर्ट", "🗺️ Map": "🗺️ मैप", "⚙️ Admin": "⚙️ एडमिन", "Logout": "लॉगआउट",
    "📊 Analytics": "📊 एनालिटिक्स", "⚖️ Compare": "⚖️ तुलना", "💧 Quality": "💧 क्वालिटी", "📈 Forecast": "📈 फोरकास्ट", "🧠 Insights": "🧠 इनसाइट्स",
    "📊 Zone-wise Analytics": "📊 ज़ोन-वाइज़ एनालिटिक्स", "⚖️ Compare Zones": "⚖️ ज़ोन कंपेयर करें", "🗺️ Indore Zone Map": "🗺️ इंदौर ज़ोन मैप", "Indore Zone Map": "इंदौर ज़ोन मैप",
    "💧 Water Quality Risk Prediction": "💧 पानी की क्वालिटी और रिस्क की जाँच", "📈 Water Availability Forecast": "📈 वाटर अवेलेबिलिटी फोरकास्ट",
    "🚨 Report Water Body Issue": "🚨 वाटर बॉडी की प्रॉब्लम रिपोर्ट करें",
    "AquaTrack Admin Panel": "AquaTrack एडमिन पैनल",
    "Restricted Access": "लिमिटेड एक्सेस",
    "Admin Login": "एडमिन लॉगिन",
    "Admin Username": "एडमिन यूज़रनेम",
    "Admin Password": "एडमिन पासवर्ड",
    "🔐 Sign In as Admin": "🔐 एडमिन की तरह लॉगिन करें",
    "← Back to Public Portal": "← पब्लिक पोर्टल पर वापस जाएँ",
    "Sign in with the AquaTrack administrator account to review reports and manage project data.": "रिपोर्ट्स देखने और प्रोजेक्ट का डेटा मैनेज करने के लिए अपने AquaTrack एडमिन अकाउंट से लॉगिन करें।",
    "This account does not have administrator privileges.": "इस अकाउंट को एडमिन एक्सेस नहीं मिला है।",
    "Invalid admin username or password.": "यूज़रनेम या पासवर्ड गलत है।",
    "Predictive Water Intelligence": "प्रिडिक्टिव वाटर इंटेलिजेंस",
    "Protect Your Water Sources": "अपने वाटर सोर्सेज़ को सेफ रखें",
    "Analyze water quality, forecast availability and understand zone-wise conditions through one integrated decision-support platform for Indore.": "इंदौर के लिए एक ही प्लेटफॉर्म पर वाटर क्वालिटी एनालाइज़ करें, अवेलेबिलिटी का फोरकास्ट देखें और हर ज़ोन की स्थिति समझें।",
     "About AquaTrack": "AquaTrack के बारे में",
     "AquaTrack brings together zone-wise water-quality screening, water-supply forecasts, citizen issue reports, and alerts in one place. It helps users identify possible risks and follow service requests across Indore.": "AquaTrack एक ही जगह पर ज़ोन-वाइज़ पानी की क्वालिटी की शुरुआती जाँच, पानी की सप्लाई का फोरकास्ट, नागरिकों की समस्या रिपोर्ट और अलर्ट देता है। इससे संभावित रिस्क पहचानने और इंदौर में सर्विस रिक्वेस्ट का स्टेटस देखने में मदद मिलती है।",
     "Water Quality Screening": "वाटर क्वालिटी स्क्रीनिंग",
     "Availability Forecasts": "अवेलेबिलिटी फोरकास्ट",
     "Citizen Reports & Alerts": "सिटिजन रिपोर्ट्स और अलर्ट्स",
     "Project prototype using project-generated data; predictions are for screening only and do not replace laboratory testing or official municipal systems.": "यह प्रोजेक्ट-लेवल प्रोटोटाइप है और प्रोजेक्ट के बनाए डेटा का उपयोग करता है। प्रिडिक्शन सिर्फ शुरुआती स्क्रीनिंग के लिए हैं; ये लैब टेस्ट या आधिकारिक म्युनिसिपल सिस्टम का विकल्प नहीं हैं。",

      "Worker Dashboard": "वर्कर डैशबोर्ड",
      "👷 Worker Dashboard": "👷 वर्कर डैशबोर्ड",
      "Signed in as": "लॉगिन यूज़र",
      "Active": "एक्टिव",
      "Assigned": "असाइन की गई",
      "In Progress": "काम जारी",
      "Verification": "वेरिफिकेशन",
      "Notifications": "नोटिफिकेशन",
      "My Assigned Requests": "मुझे असाइन की गई रिक्वेस्ट्स",
      "Total completed work": "कुल पूरा किया गया कार्य",
      "Sites personally completed": "व्यक्तिगत रूप से पूरे किए गए साइट्स",
      "Completed Work": "पूरा किया गया कार्य",
      "No active requests are currently assigned to you.": "अभी आपके नाम कोई एक्टिव रिक्वेस्ट असाइन नहीं है।",
      "No notifications.": "कोई नोटिफिकेशन नहीं है।",
      "Request Timeline": "रिक्वेस्ट टाइमलाइन",
      "No status history available.": "कोई स्टेटस हिस्ट्री उपलब्ध नहीं है।",
      "▶ Start Work": "▶ काम शुरू करें",
      "Work remarks (required)": "काम की जानकारी (ज़रूरी)",
      "Upload completion evidence (optional)": "काम पूरा होने का सबूत अपलोड करें (ज़रूरी नहीं)",
      "✅ Mark Work Completed": "✅ काम पूरा मार्क करें",
      "Mark all notifications as read": "सभी नोटिफिकेशन को पढ़ा हुआ मार्क करें",
      "Busy": "व्यस्त",
      "Offline": "ऑफलाइन",

    "PROJECT ZONES": "प्रोजेक्ट ज़ोन", "QUALITY RECORDS": "क्वालिटी रिकॉर्ड्स", "AVAILABILITY RECORDS": "अवेलेबिलिटी रिकॉर्ड्स",
    "PREDICTIONS": "प्रिडिक्शन्स",
    "Zone Status Right Now": "अभी हर ज़ोन की स्थिति",
    "SAFE ZONES": "सेफ ज़ोन", "MODERATE RISK": "मीडियम रिस्क", "HIGH RISK": "हाई रिस्क", "SCARCITY RISK": "कमी का रिस्क",
    "Active Alerts": "एक्टिव अलर्ट्स",
    "Show all": "सभी दिखाएँ", "alerts": "अलर्ट्स",
    "No active alerts. All zones look normal in the latest records.": "कोई एक्टिव अलर्ट नहीं है। लेटेस्ट रिकॉर्ड्स में सभी ज़ोन नॉर्मल लग रहे हैं।",
    "Water Quality Risk Overview": "वाटर क्वालिटी रिस्क ओवरव्यू",
    "Water Availability Overview": "वाटर अवेलेबिलिटी ओवरव्यू",
    "No water-quality data available.": "वाटर क्वालिटी का डेटा उपलब्ध नहीं है।",
    "No availability data available.": "अवेलेबिलिटी का डेटा उपलब्ध नहीं है।",
    "Recent Predictions": "रीसेंट प्रिडिक्शन्स",
    "No predictions have been recorded yet.": "अभी तक कोई प्रिडिक्शन सेव नहीं हुआ है।",
    "How AquaTrack Works": "AquaTrack कैसे काम करता है",
    "System Status": "सिस्टम स्टेटस",
    "Data & Model Health": "डेटा और मॉडल हेल्थ",
    "Refresh Dashboard Data": "डैशबोर्ड डेटा रिफ्रेश करें",
    "Dashboard data refreshed successfully.": "डैशबोर्ड डेटा सफलतापूर्वक रिफ्रेश हो गया।",
    "Models ready": "मॉडल रेडी हैं",
    "Model files are available for predictions.": "प्रिडिक्शन के लिए मॉडल फाइल्स उपलब्ध हैं।",
    "Latest quality data": "लेटेस्ट क्वालिटी डेटा",
    "Latest availability data": "लेटेस्ट अवेलेबिलिटी डेटा",
    "No date available": "डेट उपलब्ध नहीं है",
    "All core systems are available.": "सभी कोर सिस्टम उपलब्ध हैं।",
    "Model Status": "मॉडल स्टेटस",
    "Model files unavailable": "मॉडल फाइल्स उपलब्ध नहीं हैं",
    "Check the model files before running predictions.": "प्रिडिक्शन चलाने से पहले मॉडल फाइल्स चेक करें।",
    "quality records": "क्वालिटी रिकॉर्ड्स",
    "availability records": "अवेलेबिलिटी रिकॉर्ड्स",
    "Quality Assessment": "क्वालिटी असेसमेंट",
    "Seven parameters screened against BIS IS 10500:2012 and classified by a Random Forest.": "सात पैरामीटर्स को BIS IS 10500:2012 के हिसाब से चेक किया जाता है और Random Forest उन्हें क्लासिफाई करता है।",
    "Predictive Forecast": "प्रिडिक्टिव फोरकास्ट",
    "Linear Regression estimates supply hours from history, rainfall and season.": "Linear Regression पुराने डेटा, बारिश और मौसम के आधार पर सप्लाई के घंटे का अंदाज़ा लगाता है।",
    "Zone Insights": "ज़ोन इनसाइट्स",
    "Compare project-defined zones with charts, maps and trends.": "चार्ट, मैप और ट्रेंड्स के ज़रिए प्रोजेक्ट के ज़ोन कंपेयर करें।",
    "Alerts & Actions": "अलर्ट्स और एक्शन",
    "Risks are highlighted with practical recommendations.": "रिस्क को प्रैक्टिकल सुझावों के साथ हाइलाइट किया जाता है।",
    "Select Zone": "ज़ोन चुनें",
    "Period": "पीरियड",
    "Last 30 days": "पिछले 30 दिन", "Last 60 days": "पिछले 60 दिन", "Last 90 days": "पिछले 90 दिन", "All records": "सभी रिकॉर्ड्स",
    "WATER QUALITY": "वाटर क्वालिटी", "BIS PARAMETERS EXCEEDED": "BIS लिमिट से ऊपर पैरामीटर", "LATEST SUPPLY": "लेटेस्ट सप्लाई",
    "AVAILABILITY": "अवेलेबिलिटी",
    "💧 Water Quality": "💧 वाटर क्वालिटी", "🚰 Water Availability": "🚰 वाटर अवेलेबिलिटी", "📄 Reports": "📄 रिपोर्ट्स",
    "No water-quality data found for this zone.": "इस ज़ोन का वाटर क्वालिटी डेटा नहीं मिला।",
    "##### Latest reading against BIS limits": "##### BIS लिमिट के हिसाब से लेटेस्ट रीडिंग",
    "Water quality parameter": "वाटर क्वालिटी पैरामीटर",
    "Predicted risk share": "प्रिडिक्टेड रिस्क का शेयर",
    "View recent quality records": "रीसेंट क्वालिटी रिकॉर्ड्स देखें",
    "No availability data found for this zone.": "इस ज़ोन का अवेलेबिलिटी डेटा नहीं मिला।",
    "Supply hours trend": "सप्लाई ऑवर्स का ट्रेंड",
    "Supply by season": "मौसम के हिसाब से सप्लाई",
    "Rainfall vs supply hours": "बारिश बनाम सप्लाई ऑवर्स",
    "View recent availability records": "रीसेंट अवेलेबिलिटी रिकॉर्ड्स देखें",
    "Build the PDF and Excel report for this zone. Files are created only when you ask.": "इस ज़ोन की PDF और Excel रिपोर्ट बनाएँ। फाइल्स तभी बनेंगी जब आप कहेंगे।",
    "📦 Prepare reports": "📦 रिपोर्ट्स तैयार करें",
    "Building reports...": "रिपोर्ट्स बन रही हैं...",
    "📄 Download PDF Report": "📄 PDF रिपोर्ट डाउनलोड करें", "📊 Download Excel Report": "📊 Excel रिपोर्ट डाउनलोड करें",
    "⚖️ Compare Zones": "⚖️ ज़ोन कंपेयर करें",
    "Pick two to six zones and compare their latest water quality and supply side by side.": "2 से 6 ज़ोन चुनें और उनकी लेटेस्ट वाटर क्वालिटी और सप्लाई को साथ-साथ कंपेयर करें।",
    "Zones to compare": "कंपेयर करने के लिए ज़ोन",
    "Select at least two zones to start comparing.": "कंपेयर करना शुरू करने के लिए कम से कम 2 ज़ोन चुनें।",
    "Parameter Comparison": "पैरामीटर कंपेयरिज़न",
    "Parameter": "पैरामीटर",
    "Overall Profile": "ओवरऑल प्रोफाइल",
    "Each axis shows the reading as a percentage of its BIS upper limit. Beyond 100% means the limit is exceeded.": "हर एक्सिस पर रीडिंग को उसकी BIS लिमिट के % में दिखाया गया है। 100% से ज़्यादा का मतलब है कि लिमिट क्रॉस हो गई है।",
    "Supply Trend": "सप्लाई ट्रेंड",
    "No availability records for the selected zones.": "चुने गए ज़ोन का अवेलेबिलिटी डेटा नहीं मिला।",
    "Supply hours, last 60 days": "पिछले 60 दिनों के सप्लाई ऑवर्स",
    "Interactive map of the project-defined AquaTrack zones. Use the layer control for satellite view.": "प्रोजेक्ट के AquaTrack ज़ोन का इंटरैक्टिव मैप। सैटेलाइट व्यू के लिए लेयर कंट्रोल यूज़ करें।",
     "Report Location": "रिपोर्ट की लोकेशन",
     "Coordinates": "कोऑर्डिनेट्स",
     "Colony / Locality": "कॉलोनी / इलाका",
     "Society / Building": "सोसाइटी / बिल्डिंग",
     "Water body location": "वाटर बॉडी की लोकेशन",
     "Open exact location in Google Maps": "Google Maps में सही लोकेशन खोलें",
     "Open water body location in Google Maps": "Google Maps में वाटर बॉडी की लोकेशन खोलें",
     "GPS coordinates were not captured for this report. Use the zone and locality details above.": "इस रिपोर्ट के साथ GPS लोकेशन सेव नहीं हुई थी। ऊपर दिए गए ज़ोन और इलाके की जानकारी देखें।",
     "GPS location is not available for the issue; this link shows the associated water body instead.": "समस्या की GPS लोकेशन उपलब्ध नहीं है; यह लिंक संबंधित वाटर बॉडी की लोकेशन दिखाता है।",
     "Mark as Done / Solved": "काम पूरा करें / सॉल्व मार्क करें",
     "After you mark this job as done, it will be sent to the admin for final verification.": "काम पूरा मार्क करने के बाद रिक्वेस्ट अंतिम जाँच के लिए एडमिन के पास जाएगी।",
    "Show zones by water-quality risk": "वाटर क्वालिटी रिस्क के हिसाब से ज़ोन दिखाएँ",
    "Supply rings": "सप्लाई रिंग्स", "Water bodies": "वाटर बॉडीज़",
    "These are representative, project-defined locations used for the prototype and are not official municipal zone boundaries.": "ये सिर्फ प्रोटोटाइप के लिए प्रोजेक्ट-डिफाइंड लोकेशन हैं, इन्हें ऑफिशियल म्युनिसिपल ज़ोन बाउंड्री न समझें।",
    "Safe": "सेफ", "Moderate risk": "मीडियम रिस्क", "High risk": "हाई रिस्क", "No data": "डेटा नहीं", "Water body": "वाटर बॉडी",
    "Enter water-quality parameters to get the model's risk category, a BIS limit check and the model's confidence.": "पानी की क्वालिटी की जानकारी भरें और जानें कि पानी सुरक्षित है या उसमें रिस्क है।",
    "Fields start with the latest recorded reading for this zone. Change any value and press Predict.": "इन फील्ड्स में इस ज़ोन की सबसे हाल की रीडिंग दी गई है। जरूरत के अनुसार वैल्यू बदलें और Predict दबाएँ।",
    "🔍 Predict Water Quality": "🔍 पानी की क्वालिटी चेक करें",
    "Prediction Result": "जाँच का रिज़ल्ट",
    "Zone": "ज़ोन",
    "Model risk level:": "मॉडल का रिस्क लेवल:",
    "MODEL PREDICTION": "मॉडल का अनुमान", "MODEL CONFIDENCE": "मॉडल का भरोसा",
    "The direct BIS count suggests": "डायरेक्ट BIS काउंट के हिसाब से",
    "while the model predicted": "जबकि मॉडल ने प्रिडिक्ट किया",
    "This can happen for borderline readings; check the parameter table below.": "ऐसा बॉर्डरलाइन रीडिंग में हो सकता है; नीचे पैरामीटर टेबल चेक करें।",
    "##### BIS limit check": "##### BIS लिमिट की जाँच",
    "Save this prediction to history": "इस प्रिडिक्शन को हिस्ट्री में सेव करें",
    "Prediction saved to history.": "प्रिडिक्शन हिस्ट्री में सेव हो गया।",
    "Prediction could not be saved:": "प्रिडिक्शन सेव नहीं हो सका:",
    "🚨 Alerts": "🚨 अलर्ट्स", "💡 Recommendations": "💡 सुझाव",
    "Forecast water supply from the previous day's supply, rainfall and season. Choose more than one day to see a rolling forecast.": "पिछले दिन की सप्लाई, बारिश और मौसम के आधार पर वाटर सप्लाई का फोरकास्ट देखें। एक से ज़्यादा दिन चुनने पर रोलिंग फोरकास्ट दिखेगा।",
    "No historical availability data is available for the selected zone.": "चुने गए ज़ोन का पुराना अवेलेबिलिटी डेटा नहीं मिला।",
    "Starting point:": "स्टार्टिंग पॉइंट:",
    "Rainfall (mm)": "बारिश (mm)",
    "Season": "मौसम", "Forecast days": "फोरकास्ट के दिन",
    "Save the next-day forecast to history": "अगले दिन का फोरकास्ट हिस्ट्री में सेव करें",
    "📈 Forecast Availability": "📈 अवेलेबिलिटी फोरकास्ट करें",
    "Forecast Result": "फोरकास्ट रिज़ल्ट",
    "NEXT-DAY FORECAST": "अगले दिन का फोरकास्ट", "AVAILABILITY STATUS": "अवेलेबिलिटी स्टेटस",
    "LOWEST OVER": "सबसे कम", "DAY(S)": "दिन(ों) में",
    "Recorded supply and forecast": "रिकॉर्डेड सप्लाई और फोरकास्ट",
    "Later days reuse each forecast as the next day's previous supply and keep rainfall and season constant, so uncertainty grows with the horizon.": "आगे के दिनों में हर फोरकास्ट को अगले दिन की पिछली सप्लाई माना जाता है और बारिश-मौसम कॉन्स्टेंट रखे जाते हैं, इसलिए जितना आगे जाएँगे उतना अंदाज़ा कम पक्का होगा।",
    "Availability forecast saved to history.": "अवेलेबिलिटी फोरकास्ट हिस्ट्री में सेव हो गया।",
    "Forecast could not be saved:": "फोरकास्ट सेव नहीं हो सका:",
    "How the two prototype models behave on the records stored in the database.": "डेटाबेस में मौजूद रिकॉर्ड्स पर दोनों प्रोटोटाइप मॉडल कैसा परफॉर्म करते हैं।",
    "Read these numbers carefully. Risk labels are derived from the project screening rule. The displayed evaluation uses a held-out test split of the stored project data; it is not a claim of real-world municipal accuracy. Validation with authorised, laboratory-verified data is the next step.": "इन नंबर्स को ध्यान से देखें। रिस्क लेबल BIS वायलेशन काउंट से आते हैं और इन्हीं रिकॉर्ड्स से मॉडल को ट्रेन किया गया है, इसलिए ये सिर्फ स्क्रीनिंग रूल से मैच दिखाते हैं, रियल-वर्ल्ड एक्यूरेसी नहीं। अगला स्टेप है ऑथराइज़्ड, लैब-वेरिफाइड डेटा से वैलिडेशन करना।",
    "Water Quality Classifier (Random Forest)": "वाटर क्वालिटी क्लासिफायर (Random Forest)",
    "AGREEMENT WITH BIS RULE": "BIS रूल से मैच", "RECORDS CHECKED": "चेक किए गए रिकॉर्ड्स",
    "##### Per-class scores": "##### हर क्लास का स्कोर",
    "Availability Forecaster (Linear Regression)": "अवेलेबिलिटी फोरकास्टर (Linear Regression)",
    "NAIVE BASELINE MAE": "सिंपल बेसलाइन MAE",
    "The model beats the naive baseline (repeat yesterday's supply), so rainfall and season add useful information.": "मॉडल सिंपल बेसलाइन (कल की सप्लाई रिपीट करना) से बेहतर है, यानी बारिश और मौसम काम की जानकारी दे रहे हैं।",
    "The model does not beat the naive baseline (repeat yesterday's supply). Consider richer features or a time-series model.": "मॉडल सिंपल बेसलाइन (कल की सप्लाई रिपीट करना) से बेहतर नहीं है। और फीचर्स या टाइम-सीरीज़ मॉडल पर सोचा जा सकता है।",
    "Report a visible water-body or local water-supply issue for project-defined water bodies and served areas.": "प्रोजेक्ट के वाटर बॉडी और एरिया में दिख रही किसी भी वाटर बॉडी या सप्लाई प्रॉब्लम को रिपोर्ट करें।",
    "Open reports for this water body:": "इस वाटर बॉडी की ओपन रिपोर्ट्स:",
    "No project-defined water bodies are available.": "अभी कोई प्रोजेक्ट वाटर बॉडी उपलब्ध नहीं है।",
    "Select Water Body": "वाटर बॉडी चुनें",
    "No served colony/society relationship is available for this water body.": "इस वाटर बॉडी से जुड़ी कोई कॉलोनी/सोसाइटी की जानकारी नहीं है।",
    "Select Colony / Society": "कॉलोनी / सोसाइटी चुनें",
    "Issue Type": "प्रॉब्लम का टाइप",
    "Water appears dirty": "पानी गंदा लग रहा है", "Bad smell": "बदबू आ रही है", "Unusual colour": "अजीब रंग",
    "Floating waste": "कचरा तैर रहा है", "Low water supply": "कम पानी आ रहा है", "Possible contamination": "दूषित हो सकता है", "Other": "अन्य",
    "Description": "डिस्क्रिप्शन",
    "Describe what you observed...": "आपने क्या देखा, वो लिखें...",
    "Severity": "सीवियरिटी", "Low": "लो", "Medium": "मीडियम", "High": "हाई",
    "Add Photo (optional, up to 5 MB)": "फोटो जोड़ें (ऑप्शनल, 5 MB तक)",
    "Selected report photo": "सिलेक्टेड रिपोर्ट फोटो",
    "### 📍 Report Location": "### 📍 रिपोर्ट की लोकेशन",
    "🔎 Get Current Address": "🔎 करंट एड्रेस पता करें",
    "Getting location details...": "लोकेशन डिटेल्स मिल रही हैं...",
    "Select Report Location": "रिपोर्ट की लोकेशन चुनें",
    "Use Current Location": "मेरी वर्तमान लोकेशन इस्तेमाल करें",
    "Click the map to pin the exact problem location.": "समस्या की सही जगह चुनने के लिए मैप पर क्लिक करें।",
    "Selected location": "चुनी गई लोकेशन",
    "Clear selected location": "चुनी गई लोकेशन हटाएँ",
    "Current location captured with accuracy": "वर्तमान लोकेशन मिली। सटीकता:",
    "No location selected yet. Use Current Location or click the map to set one.": "अभी लोकेशन नहीं चुनी गई है। वर्तमान लोकेशन लें या सही जगह चुनने के लिए मैप पर क्लिक करें।",
    "Open selected location in Google Maps": "चुनी गई लोकेशन Google Maps में खोलें",
    "Location permission was not granted or the browser could not determine your location. You can still click the map.": "लोकेशन की अनुमति नहीं मिली या ब्राउज़र लोकेशन पता नहीं कर पाया। आप फिर भी मैप पर क्लिक करके जगह चुन सकते हैं।",
    "No GPS location selected": "GPS लोकेशन नहीं चुनी गई",
    "Location saved with report": "रिपोर्ट के साथ लोकेशन सेव हो गई",
    "Selected Water Body Location": "सिलेक्टेड वाटर बॉडी की लोकेशन",
    "My Current Location": "मेरी करंट लोकेशन",
    "Your detected current location": "आपकी करंट लोकेशन",
    "Selected water body location": "सिलेक्टेड वाटर बॉडी की लोकेशन",
    "🚨 Submit Report": "🚨 रिपोर्ट सबमिट करें",
    "Please enter a description of the issue.": "कृपया प्रॉब्लम का डिस्क्रिप्शन लिखें।",
    "The photo is larger than 5 MB. Please choose a smaller image.": "फोटो 5 MB से बड़ी है। कृपया छोटी फोटो चुनें।",
    "A photo is required to submit a report.": "रिपोर्ट सबमिट करने के लिए फोटो अपलोड करना ज़रूरी है।",
    "✅ Report submitted successfully. Status: Pending": "✅ रिपोर्ट सबमिट हो गई। स्टेटस: पेंडिंग",
    "Your report has been recorded for admin review.": "आपकी रिपोर्ट एडमिन रिव्यू के लिए दर्ज हो गई है।",
    "Unable to submit report:": "रिपोर्ट सबमिट नहीं हो सकी:",
    "This is a project-level reporting prototype. Water-body and served-area relationships shown here are project-defined demonstration data and are not claimed as official municipal relationships.": "यह सिर्फ एक प्रोजेक्ट-लेवल रिपोर्टिंग प्रोटोटाइप है। यहाँ दिखाए गए वाटर बॉडी और एरिया के कनेक्शन प्रोजेक्ट के डेमो डेटा हैं, इन्हें ऑफिशियल म्युनिसिपल कनेक्शन न समझें।",
    "Administrator": "एडमिन",
    "Admin section": "एडमिन सेक्शन",
    "Overview": "ओवरव्यू", "Water Body Reports": "वाटर बॉडी रिपोर्ट्स", "Data Import": "डेटा इम्पोर्ट",
    "Predictions & Exports": "प्रिडिक्शन्स और एक्सपोर्ट", "System": "सिस्टम",
    "Project Zones": "प्रोजेक्ट ज़ोन", "Water Quality Records": "वाटर क्वालिटी रिकॉर्ड्स",
    "Availability Records": "अवेलेबिलिटी रिकॉर्ड्स", "Water Bodies": "वाटर बॉडीज़", "Total Reports": "टोटल रिपोर्ट्स",
    "Pending": "पेंडिंग", "Resolved": "सॉल्व हो गया", "Investigating": "जाँच हो रही है",
    "📍 Zone Status": "📍 ज़ोन स्टेटस", "📍 Project Zones": "📍 प्रोजेक्ट ज़ोन",
    "🤖 Machine Learning Models": "🤖 मशीन लर्निंग मॉडल्स",
    "Water Quality Assessment": "वाटर क्वालिटी असेसमेंट", "Water Availability Forecasting": "वाटर अवेलेबिलिटी फोरकास्टिंग",
    "Algorithm: Random Forest Classifier": "एल्गोरिदम: Random Forest Classifier",
    "Algorithm: Linear Regression": "एल्गोरिदम: Linear Regression",
    "Inputs: pH, TDS, turbidity, hardness, chloride, fluoride, nitrate": "इनपुट: pH, TDS, turbidity, hardness, chloride, fluoride, nitrate",
    "Inputs: previous supply hours, rainfall, season": "इनपुट: पिछली सप्लाई के घंटे, बारिश, मौसम",
    "Output: Safe / Moderate Risk / High Risk": "आउटपुट: Safe / Moderate Risk / High Risk",
    "Output: forecast supply hours and availability category": "आउटपुट: फोरकास्ट सप्लाई ऑवर्स और अवेलेबिलिटी कैटेगरी",
    "How would you like to check the water?": "पानी की जाँच कैसे करना चाहते हैं?",
    "Yes": "हाँ",
    "No": "ना",
    "I have a water test report": "मेरे पास पानी की जाँच की रिपोर्ट है",
    "I do not have a water test report": "मेरे पास पानी की जाँच की रिपोर्ट नहीं है",
    "Use the values from your water test report. AquaTrack will analyze them using the water-quality model.": "अपनी पानी की जाँच रिपोर्ट में दी गई जानकारी भरें। AquaTrack उसी जानकारी के आधार पर पानी की क्वालिटी की जाँच करेगा।",
    "No report? You can do a simple visual check instead. This is only a preliminary check and not a laboratory test.": "रिपोर्ट नहीं है? आप पानी को देखकर कुछ आसान सवालों के जवाब दे सकते हैं। यह सिर्फ शुरुआती जाँच है, लैब टेस्ट नहीं।",
    "Check the water using simple questions": "आसान सवालों से पानी की जाँच करें",
    "1. Is the water clear?": "1. क्या पानी साफ दिखाई दे रहा है?",
    "2. Can you see dirt or waste in the water?": "2. क्या पानी में कचरा या गंदगी दिखाई दे रही है?",
    "3. Does the water smell bad?": "3. क्या पानी से बदबू आ रही है?",
    "4. Does the water look yellow, brown or another unusual colour?": "4. क्या पानी पीला, भूरा या किसी और रंग का दिखाई दे रहा है?",
    "5. Does the water look very cloudy?": "5. क्या पानी बहुत मटमैला दिखाई दे रहा है?",
    "6. Does the water taste strange or bad?": "6. क्या पानी का स्वाद अजीब या खराब लगता है?",
    "Do not intentionally taste water that may be unsafe.": "अगर पानी संदिग्ध लग रहा है तो उसे जानबूझकर चखकर जाँच न करें।",
    "Check Water": "पानी की जाँच करें",
    "Preliminary Check Result": "शुरुआती जाँच का नतीजा",
    "Water quality warning": "पानी की गुणवत्ता की चेतावनी",
    "Several problems are visible in the water. Please do not drink it until the water has been properly tested.": "पानी में कई समस्याएँ दिखाई दे रही हैं। पानी की सही जाँच होने तक इसे न पिएँ।",
    "Please report this problem to the system so it can be reviewed.": "कृपया इस समस्या की जानकारी सिस्टम में रिपोर्ट करें ताकि इसकी जाँच की जा सके।",
    "Report this problem to the system": "इस समस्या की रिपोर्ट करें",
    "Opening the water issue report form.": "पानी की समस्या की रिपोर्ट वाला फॉर्म खोला जा रहा है।",
    "The water looks normal from the information provided.": "दी गई जानकारी के अनुसार पानी सामान्य दिखाई दे रहा है।",
    "Some things about the water need attention.": "पानी में कुछ ऐसी बातें हैं जिन पर ध्यान देना चाहिए।",
    "Several problems are visible in the water. Get the water tested before drinking it.": "पानी में कई समस्याएँ दिखाई दे रही हैं। पीने से पहले पानी की जाँच करवाएँ।",
    "This is only a preliminary check based on what can be seen or noticed. It cannot measure things that are not visible.": "यह सिर्फ दिखाई देने वाली या महसूस होने वाली बातों पर आधारित शुरुआती जाँच है। इससे वे चीजें पता नहीं चल सकतीं जो दिखाई नहीं देतीं।",
    "📊 TOTAL REPORTS": "📊 कुल रिपोर्ट्स", "⏳ PENDING": "⏳ पेंडिंग",
    "🔎 INVESTIGATING": "🔎 जाँच जारी", "✅ RESOLVED": "✅ सॉल्व हो गया",
    "Project-defined water bodies available:": "उपलब्ध प्रोजेक्ट वाटर बॉडीज़:",
    "No water-body issue reports have been submitted yet.": "अभी तक कोई वाटर बॉडी प्रॉब्लम रिपोर्ट सबमिट नहीं हुई।",
    "Reports by issue type": "प्रॉब्लम टाइप के हिसाब से रिपोर्ट्स", "Reports by severity": "सीवियरिटी के हिसाब से रिपोर्ट्स",
    "Report filter": "रिपोर्ट फिल्टर",
    "⬇️ Export CSV": "⬇️ CSV एक्सपोर्ट करें",
    "All": "सभी", "PHOTO": "फोटो", "LOCATION": "लोकेशन", "DESCRIPTION": "डिस्क्रिप्शन", "DATE": "डेट", "STATUS": "स्टेटस",
    "No water body issue reports have been submitted yet.": "अभी तक कोई वाटर बॉडी प्रॉब्लम रिपोर्ट सबमिट नहीं हुई।",
    "📂 Upload Water Quality CSV": "📂 वाटर क्वालिटी CSV अपलोड करें",
    "Upload a CSV with water-quality records. The system validates the data and skips records that already exist.": "वाटर क्वालिटी रिकॉर्ड्स वाली CSV अपलोड करें। सिस्टम डेटा चेक करेगा और जो रिकॉर्ड्स पहले से हैं उन्हें स्किप कर देगा।",
    "Required columns: zone, date, pH, TDS, turbidity, hardness, chloride, fluoride, nitrate": "ज़रूरी कॉलम: zone, date, pH, TDS, turbidity, hardness, chloride, fluoride, nitrate",
    "Choose CSV file": "CSV फाइल चुनें",
    "### Uploaded Data Preview": "### अपलोड किए डेटा का प्रीव्यू",
    "Missing columns: ": "मिसिंग कॉलम: ",
    "CSV validation failed.": "CSV वेलिडेशन फेल हो गया।",
    "CSV validation successful.": "CSV वेलिडेशन सही है।",
    "### Import Summary": "### इम्पोर्ट समरी",
    "New Records": "नए रिकॉर्ड्स", "Duplicates": "डुप्लीकेट",
    "📥 Import New Records": "📥 नए रिकॉर्ड्स इम्पोर्ट करें",
    "No new records to import. All uploaded records already exist in the database.": "इम्पोर्ट करने के लिए कोई नया रिकॉर्ड नहीं है। सभी अपलोड किए रिकॉर्ड्स पहले से डेटाबेस में हैं।",
    "Unable to process CSV:": "CSV प्रोसेस नहीं हो सकी:",
    "📋 Prediction History": "📋 प्रिडिक्शन हिस्ट्री",
    "No prediction history available.": "कोई प्रिडिक्शन हिस्ट्री उपलब्ध नहीं है।",
    "Zone": "ज़ोन", "Type": "टाइप", "All zones": "सभी ज़ोन", "All types": "सभी टाइप",
    "Water Quality": "वाटर क्वालिटी", "Availability Forecast": "अवेलेबिलिटी फोरकास्ट",
    "⬇️ Download prediction history (CSV)": "⬇️ प्रिडिक्शन हिस्ट्री डाउनलोड करें (CSV)",
    "🗄️ Dataset Exports": "🗄️ डेटासेट एक्सपोर्ट",
    "⬇️ Water quality data (CSV)": "⬇️ वाटर क्वालिटी डेटा (CSV)",
    "⬇️ Availability data (CSV)": "⬇️ अवेलेबिलिटी डेटा (CSV)",
    "⚙️ System Status": "⚙️ सिस्टम स्टेटस",
    "SQLite database file": "SQLite डेटाबेस फाइल", "Water-quality model file": "वाटर क्वालिटी मॉडल फाइल",
    "Availability model file": "अवेलेबिलिटी मॉडल फाइल", "Report photo folder": "रिपोर्ट फोटो फोल्डर",
    "Available": "उपलब्ध है", "Missing": "मिसिंग है",
    "Database connection: OK (": "डेटाबेस कनेक्शन: ठीक है (",
    "quality and": "क्वालिटी और", "availability records)": "अवेलेबिलिटी रिकॉर्ड्स)",
    "Database connection failed:": "डेटाबेस कनेक्शन फेल हुआ:",
    "🔄 Refresh cached data": "🔄 कैश्ड डेटा रीफ्रेश करें",
    "Data is cached for speed. Use this after editing the database outside the app.": "स्पीड के लिए डेटा कैश किया जाता है। अगर आपने ऐप के बाहर डेटाबेस बदला है तो इसे यूज़ करें।",
    "pH": "pH — पानी कितना अम्लीय या क्षारीय है", "TDS (mg/L)": "पानी में घुले हुए पदार्थों की मात्रा (TDS)", "Turbidity (NTU)": "पानी का मटमैलापन (Turbidity)",
    "Hardness (mg/L)": "पानी का खारापन / कठोरता (Hardness)", "Chloride (mg/L)": "पानी में क्लोराइड की मात्रा (Chloride)",
    "Fluoride (mg/L)": "पानी में फ्लोराइड की मात्रा (Fluoride)", "Nitrate (mg/L)": "पानी में नाइट्रेट की मात्रा (Nitrate)",
    "Moderate Risk": "मीडियम रिस्क", "High Risk": "हाई रिस्क",
    "Good Availability": "अच्छी अवेलेबिलिटी", "Moderate Availability": "मीडियम अवेलेबिलिटी", "Scarcity Risk": "कमी का रिस्क",
    "Summer": "गर्मी", "Monsoon": "मानसून", "Winter": "सर्दी",
    "Risk": "रिस्क", "Status": "स्टेटस", "Supply": "सप्लाई", "Availability": "अवेलेबिलिटी",
    "BIS violations": "BIS वायलेशन", "Supply (h)": "सप्लाई (घंटे)", "Hours": "घंटे",
    "Water quality": "वाटर क्वालिटी", "BIS parameters exceeded": "BIS लिमिट से ऊपर पैरामीटर",
    "No Data": "डेटा नहीं", "BIS max": "BIS मैक्सिमम", "BIS min": "BIS मिनिमम",
    "Current location detected • Accuracy:": "करंट लोकेशन मिली • एक्यूरेसी:",
    "Current address:": "करंट एड्रेस:",
    "Location used for this report:": "इस रिपोर्ट के लिए यूज़ की गई लोकेशन:",
    "Current location (address lookup unavailable)": "करंट लोकेशन (एड्रेस नहीं मिल सका)",
    "Status for Report #": "रिपोर्ट # का स्टेटस", "Report #": "रिपोर्ट #",
    "Severity:": "सीवियरिटी:", "Water Body:": "वाटर बॉडी:", "Issue:": "प्रॉब्लम:",
    "Reports": "रिपोर्ट्स", "shown": "दिखाई गईं",
    "No": "नहीं",
    "existing record(s) will be skipped.": "मौजूदा रिकॉर्ड्स स्किप हो जाएँगे।",
    "Import completed successfully.": "इम्पोर्ट सफलतापूर्वक पूरा हुआ।",
    "new record(s) added.": "नए रिकॉर्ड्स जुड़ गए।",
    "Records found:": "रिकॉर्ड्स मिले:",
    "water-body reports found.": "वाटर बॉडी रिपोर्ट्स मिलीं।",
    "### 🗺️ Report Map": "### 🗺️ रिपोर्ट मैप",
    "🌐 Language": "🌐 भाषा",
    "Alerts": "अलर्ट्स",
    "📢 Resident Water Alerts": "📢 रेज़िडेंट वाटर अलर्ट्स",
    "Register your area once. New water-problem reports for that zone will automatically appear in your in-app alerts.": "अपने एरिया को एक बार रजिस्टर करें। उस ज़ोन की नई वाटर प्रॉब्लम रिपोर्ट्स के अलर्ट्स आपको अपने आप मिलेंगे।",
    "Your Zone": "आपका ज़ोन",
    "Your Name": "आपका नाम",
    "Phone Number": "फोन नंबर",
    "Email (for alerts)": "अलर्ट्स के लिए ईमेल",
    "🔔 Register for Alerts": "🔔 अलर्ट्स के लिए रजिस्टर करें",
    "Please provide a phone number or email.": "कृपया फोन नंबर या ईमेल दें।",
    "🔔 Your Recent Alerts": "🔔 आपके हाल के अलर्ट्स",
    "No alerts yet. New reports for your registered zone will appear here automatically.": "अभी कोई अलर्ट नहीं है। आपके रजिस्टर किए ज़ोन की नई रिपोर्ट्स यहाँ अपने आप दिखेंगी।",
    "Registered Residents": "रजिस्टर किए रेज़िडेंट्स",
    "Alert Records": "अलर्ट रिकॉर्ड्स",
    "Email Deliveries": "ईमेल डिलीवरी",
    "No residents have registered for alerts yet.": "अभी किसी रेज़िडेंट ने अलर्ट्स के लिए रजिस्टर नहीं किया है।",
    "🔔 Resident Alerts": "🔔 रेज़िडेंट अलर्ट्स",
    "Alert Log": "अलर्ट लॉग",
    "No alerts have been generated yet.": "अभी कोई अलर्ट जनरेट नहीं हुआ है।",
    "📢 Manual Broadcast": "📢 मैनुअल ब्रॉडकास्ट",
    "Alert message": "अलर्ट मैसेज",
    "Write the message residents should receive...": "रेज़िडेंट्स को मिलने वाला मैसेज लिखें...",
    "📢 Send Now": "📢 अभी भेजें",
    "Please enter an alert message.": "कृपया अलर्ट मैसेज लिखें।",
}}


def t(text):
    if st.session_state.get("language", "English") != "हिंदी" or text is None:
        return text

    value = str(text)
    translations = TRANSLATIONS.get("हिंदी", {})
    if value in translations:
        return translations[value]

    for english, hindi in sorted(translations.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"\b{re.escape(english)}\b"
        value = re.sub(pattern, hindi, value)

    return value


def _localize_dataframe(data):
    if st.session_state.get("language", "English") != "हिंदी" or not isinstance(data, pd.DataFrame):
        return data
    out = data.copy()
    rename = {
        "zone_name": "Zone", "risk": "Risk", "availability_status": "Availability",
        "violations": "BIS violations", "supply_hours": "Supply (h)", "date": "Date",
        "prediction_id": "Prediction ID", "prediction_date": "Prediction Date", "risk_level": "Risk",
        "availability_forecast": "Availability Forecast", "Type": "Type", "season": "Season",
        "rainfall": "Rainfall (mm)", "previous_supply_hours": "Previous supply hours",
        "forecast_hours": "Forecast (h)", "status": "Status", "ph": "pH", "tds": "TDS",
        "turbidity": "Turbidity", "hardness": "Hardness", "chloride": "Chloride",
        "fluoride": "Fluoride", "nitrate": "Nitrate", "latitude": "Latitude", "longitude": "Longitude",
    }
    out = out.rename(columns={k: t(v) for k, v in rename.items() if k in out.columns})
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: t(v) if isinstance(v, str) else v)
    return out


def _localize_figure(fig):
    if st.session_state.get("language", "English") != "हिंदी":
        return fig
    try:
        if getattr(fig.layout, "title", None) and fig.layout.title.text:
            fig.layout.title.text = t(fig.layout.title.text)
        for axis_name in ("xaxis", "yaxis"):
            axis = getattr(fig.layout, axis_name, None)
            if axis is not None and getattr(axis.title, "text", None):
                axis.title.text = t(axis.title.text)
        for ann in fig.layout.annotations:
            if getattr(ann, "text", None):
                ann.text = t(ann.text)
        for trace in fig.data:
            if getattr(trace, "name", None):
                trace.name = t(trace.name)
        return fig
    except Exception:  # noqa: BLE001
        return fig


def _display_dataframe(data, *args, **kwargs):
    return st.dataframe(_localize_dataframe(data), *args, **kwargs)


def _metric_bilingual(label, value=None, *args, **kwargs):
    return st.metric(t(label), t(value) if isinstance(value, str) else value, *args, **kwargs)

# ============================================================
# SESSION STATE AND NAVIGATION CALLBACKS
# ============================================================

st.session_state.setdefault("admin_logged_in", False)
st.session_state.setdefault("admin_user", None)
st.session_state.setdefault("worker_logged_in", False)
st.session_state.setdefault("worker_user", None)
st.session_state.setdefault("citizen_logged_in", False)
st.session_state.setdefault("citizen_user", None)
st.session_state.setdefault("citizen_login_error", None)
st.session_state.setdefault("page", "Dashboard")
st.session_state.setdefault("admin_login_error", None)
st.session_state.setdefault("worker_login_error", None)
st.session_state.setdefault("language", "English")
st.session_state.setdefault("mobile_menu_open", False)
st.session_state.setdefault("admin_notification_popup", False)
st.session_state.setdefault("worker_notification_popup", False)


def get_current_admin_id():
    """Return the authenticated admin's database id for audit/notification ownership."""
    user = st.session_state.get("admin_user")
    if user is None:
        return None
    try:
        if isinstance(user, dict):
            for key in ("user_id", "admin_id", "id"):
                if user.get(key) is not None:
                    return int(user[key])
        return int(user[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def navigate_to(page_name):
    st.session_state.page = page_name


def open_admin_notification_popup():
    st.session_state.admin_notification_popup = True


def close_admin_notification_popup():
    st.session_state.admin_notification_popup = False


def handle_admin_notification_click(notification_id, report_id=None):
    db_execute("UPDATE aquatrack_notifications SET is_read=1 WHERE notification_id=?", (int(notification_id),))
    st.session_state.admin_notification_popup = False
    st.session_state.page = "Admin Panel"
    st.session_state.admin_section = "Water Body Reports"


def clear_admin_unread_notifications():
    mark_notifications_read("admin", None)
    st.session_state.admin_notification_popup = False


@st.dialog("Notifications", width="small")
def show_admin_notification_popup():
    st.markdown("""
    <style>
    [data-testid="stDialog"] button[aria-label="Close"] {
        transform: translateY(24px);
    }
    [data-testid="stDialog"] .stCaption {
        margin-top: -10px !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stDialog"] .admin-notification-count {
        color: #8aa0b0;
        font-size: 0.82rem;
        margin-top: -8px;
        margin-bottom: 8px;
    }
    [data-testid="stDialog"] .stButton > button {
        background: #ffffff !important;
        color: #315a72 !important;
        border: 1px solid #e2eaf0 !important;
        box-shadow: 0 4px 12px rgba(37, 82, 107, 0.06) !important;
        font-weight: 500 !important;
    }
    [data-testid="stDialog"] .stButton > button:hover {
        background: #f8fbfd !important;
        color: #244d66 !important;
        border-color: #cbdce6 !important;
    }
    [data-testid="stDialog"] [class*="st-key-admin_notification_item_"] button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    [data-testid="stDialog"] [class*="st-key-admin_clear_notifications_button"],
    [data-testid="stDialog"] [class*="st-key-worker_clear_notifications_button"] {
        display: flex !important;
        width: 100% !important;
        justify-content: flex-end !important;
        margin-top: 8px !important;
    }
    [data-testid="stDialog"] [class*="st-key-admin_clear_notifications_button"] button,
    [data-testid="stDialog"] [class*="st-key-worker_clear_notifications_button"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #315a72 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        cursor: pointer !important;
        padding: 0 !important;
        min-height: 0 !important;
        width: max-content !important;
        display: block !important;
        margin-left: auto !important;
        margin-right: 0 !important;
    }
    [data-testid="stDialog"] [class*="st-key-admin_clear_notifications_button"] button:hover,
    [data-testid="stDialog"] [class*="st-key-worker_clear_notifications_button"] button:hover {
        background: transparent !important;
        color: #244d66 !important;
        text-decoration: underline !important;
    }
    </style>
    """, unsafe_allow_html=True)
    notes = get_notifications("admin", None, limit=50)
    unread_notes = notes[notes["is_read"].fillna(0).astype(int) == 0] if not notes.empty else notes

    if not unread_notes.empty:
        st.markdown('<div class="admin-notification-count">{} unread notification(s)</div>'.format(len(unread_notes)), unsafe_allow_html=True)

    if unread_notes.empty:
        st.info("No unread notifications.")
        return

    for _, note in unread_notes.iterrows():
        notification_id = int(note["notification_id"])
        report_id = note.get("report_id")
        message = str(note.get("message") or "Notification")
        created_at = str(note.get("created_at") or "")
        label = f"{message}\n\n{created_at}"
        if st.button(label, key=f"admin_notification_item_{notification_id}", width="stretch"):
            handle_admin_notification_click(notification_id, report_id)
            st.rerun()

    if st.button("Clear notifications", key="admin_clear_notifications_button", width="content"):
        clear_admin_unread_notifications()
        st.session_state.admin_notification_popup = True
        st.rerun()


def open_worker_notification_popup():
    st.session_state.worker_notification_popup = True


def close_worker_notification_popup():
    st.session_state.worker_notification_popup = False


def handle_worker_notification_click(notification_id, report_id=None):
    db_execute(
        "UPDATE aquatrack_notifications SET is_read=1 WHERE notification_id=? AND recipient_role='worker' AND recipient_id=?",
        (
            int(notification_id),
            int(st.session_state.worker_user["worker_id"]),
        ),
    )
    st.session_state.worker_notification_popup = False
    st.session_state.page = "Worker Dashboard"


def clear_worker_unread_notifications():
    worker = st.session_state.get("worker_user") or {}
    worker_id = worker.get("worker_id")
    if worker_id is not None:
        mark_notifications_read("worker", int(worker_id))
    st.session_state.worker_notification_popup = False


@st.dialog("Notifications", width="small")
def show_worker_notification_popup():
    st.markdown("""
    <style>
    [data-testid="stDialog"] button[aria-label="Close"] {
        transform: translateY(24px);
    }
    [data-testid="stDialog"] .stCaption {
        margin-top: -10px !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stDialog"] .admin-notification-count {
        color: #8aa0b0;
        font-size: 0.82rem;
        margin-top: -8px;
        margin-bottom: 8px;
    }
    [data-testid="stDialog"] .stButton > button {
        background: #ffffff !important;
        color: #315a72 !important;
        border: 1px solid #e2eaf0 !important;
        box-shadow: 0 4px 12px rgba(37, 82, 107, 0.06) !important;
        font-weight: 500 !important;
    }
    [data-testid="stDialog"] .stButton > button:hover {
        background: #f8fbfd !important;
        color: #244d66 !important;
        border-color: #cbdce6 !important;
    }
    [data-testid="stDialog"] [class*="st-key-worker_notification_item_"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
    }
    [data-testid="stDialog"] [class*="st-key-worker_notification_item_"] button p {
        text-align: left !important;
        width: 100% !important;
    }
    [data-testid="stDialog"] [class*="st-key-admin_clear_notifications_button"],
    [data-testid="stDialog"] [class*="st-key-worker_clear_notifications_button"] {
        display: flex !important;
        width: 100% !important;
        justify-content: flex-end !important;
        margin-top: 8px !important;
    }
    [data-testid="stDialog"] [class*="st-key-admin_clear_notifications_button"] button,
    [data-testid="stDialog"] [class*="st-key-worker_clear_notifications_button"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #315a72 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        cursor: pointer !important;
        padding: 0 !important;
        min-height: 0 !important;
        width: max-content !important;
        display: block !important;
        margin-left: auto !important;
        margin-right: 0 !important;
    }
    [data-testid="stDialog"] [class*="st-key-admin_clear_notifications_button"] button:hover,
    [data-testid="stDialog"] [class*="st-key-worker_clear_notifications_button"] button:hover {
        background: transparent !important;
        color: #244d66 !important;
        text-decoration: underline !important;
    }
    </style>
    """, unsafe_allow_html=True)

    worker = st.session_state.get("worker_user") or {}
    worker_id = worker.get("worker_id")
    if worker_id is None:
        st.info("Please sign in as a worker to view notifications.")
        return

    notes = get_notifications("worker", int(worker_id), limit=50)
    unread_notes = notes[notes["is_read"].fillna(0).astype(int) == 0] if not notes.empty else notes

    if not unread_notes.empty:
        st.markdown('<div class="admin-notification-count">{} unread notification(s)</div>'.format(len(unread_notes)), unsafe_allow_html=True)

    if unread_notes.empty:
        st.info("No unread notifications.")
        return

    for _, note in unread_notes.iterrows():
        notification_id = int(note["notification_id"])
        report_id = note.get("report_id")
        message = str(note.get("message") or "Notification")
        created_at = str(note.get("created_at") or "")
        label = f"{message}\n\n{created_at}"
        if st.button(label, key=f"worker_notification_item_{notification_id}", width="stretch"):
            handle_worker_notification_click(notification_id, report_id)
            st.rerun()

    if st.button("Clear notifications", key="worker_clear_notifications_button", width="content"):
        clear_worker_unread_notifications()
        st.session_state.worker_notification_popup = True
        st.rerun()


def open_admin():
    st.session_state.page = "Admin Panel" if st.session_state.admin_logged_in else "Admin Login"


def open_worker():
    st.session_state.page = "Worker Dashboard" if st.session_state.worker_logged_in else "Worker Login"


def do_logout():
    if st.session_state.admin_logged_in:
        st.session_state.admin_logged_in = False
        st.session_state.admin_user = None
    if st.session_state.worker_logged_in:
        st.session_state.worker_logged_in = False
        st.session_state.worker_user = None
    st.session_state.page = "Dashboard"


def toggle_language():
    current = st.session_state.get("language", "English")
    st.session_state.language = "हिंदी" if current == "English" else "English"

    # Language changes trigger a Streamlit rerun; close notification dialogs
    # first so they do not reopen when the language button is clicked.
    st.session_state.admin_notification_popup = False
    st.session_state.worker_notification_popup = False


def toggle_mobile_menu():
    st.session_state.mobile_menu_open = not st.session_state.get("mobile_menu_open", False)


def close_mobile_menu():
    st.session_state.mobile_menu_open = False


def mobile_navigate(page_name):
    st.session_state.page = page_name
    st.session_state.mobile_menu_open = False


def mobile_open_admin():
    open_admin()
    st.session_state.mobile_menu_open = False


def mobile_open_worker():
    open_worker()
    st.session_state.mobile_menu_open = False


def mobile_logout():
    do_logout()
    st.session_state.mobile_menu_open = False


def mobile_toggle_language():
    toggle_language()
    st.session_state.mobile_menu_open = False


def do_citizen_logout():
    st.session_state.citizen_logged_in=False; st.session_state.citizen_user=None; st.session_state.resident_id=None; st.session_state.page="Dashboard"

def attempt_citizen_login():
    resident=authenticate_resident(st.session_state.get("citizen_email",""),st.session_state.get("citizen_password",""))
    if resident:
        st.session_state.citizen_logged_in=True
        st.session_state.citizen_user=resident
        st.session_state.resident_id=int(resident["resident_id"])
        st.session_state.resident_zone_id=int(resident["zone_id"])
        st.session_state.citizen_login_error=None
        st.session_state.page=st.session_state.pop("citizen_after_login", "Dashboard")
    else:
        st.session_state.citizen_login_error="Invalid email or password."

def open_citizen_login():
    st.session_state.page="Citizen Login"

def open_citizen_register():
    st.session_state.page="Citizen Register"

def require_citizen_login(target_page):
    """Open a citizen facility when authenticated, otherwise send the visitor to Login."""
    if st.session_state.get("citizen_logged_in", False):
        st.session_state.page = target_page
    else:
        st.session_state.citizen_after_login = target_page
        st.session_state.page = "Citizen Login"

def attempt_admin_login():
    user = authenticate_user(
        st.session_state.get("admin_username", ""),
        st.session_state.get("admin_password", ""),
    )
    if user and user[3] == "admin":
        st.session_state.admin_logged_in = True
        st.session_state.worker_logged_in = False
        st.session_state.worker_user = None
        st.session_state.admin_user = user
        st.session_state.admin_login_error = None
        st.session_state.page = "Admin Panel"
    elif user:
        st.session_state.admin_login_error = "This account does not have administrator privileges."
    else:
        st.session_state.admin_login_error = "Invalid admin username or password."


def attempt_worker_login():
    worker = authenticate_worker(
        st.session_state.get("worker_username", ""),
        st.session_state.get("worker_password", ""),
    )
    if worker:
        st.session_state.worker_logged_in = True
        st.session_state.admin_logged_in = False
        st.session_state.admin_user = None
        st.session_state.worker_user = worker
        st.session_state.worker_login_error = None
        st.session_state.page = "Worker Dashboard"
    else:
        st.session_state.worker_login_error = "Invalid worker username or password."



def _oauth_config(provider):
    """Read OAuth client settings without exposing secrets in source code."""
    provider = str(provider).strip().lower()
    try:
        cfg = st.secrets.get("oauth", {}).get(provider, {})
    except Exception:
        cfg = {}
    client_id = str(cfg.get("client_id", os.getenv(f"AQUATRACK_{provider.upper()}_CLIENT_ID", ""))).strip()
    client_secret = str(cfg.get("client_secret", os.getenv(f"AQUATRACK_{provider.upper()}_CLIENT_SECRET", ""))).strip()
    redirect_uri = str(
        cfg.get("redirect_uri", os.getenv("AQUATRACK_OAUTH_REDIRECT_URI", "http://localhost:8501/"))
    ).strip()
    return client_id, client_secret, redirect_uri


def _oauth_authorize_url(provider):
    # Only the client ID is needed to start OAuth. The client secret is used
    # server-side when the provider redirects back with the authorization code.
    client_id, client_secret, redirect_uri = _oauth_config(provider)
    if not client_id or not redirect_uri:
        return None
    provider = provider.lower()
    state = secrets.token_urlsafe(32)
    # Store the provider against the state so the callback does not need a
    # provider query parameter and the registered redirect URI can remain exact.
    st.session_state[f"oauth_state_{provider}"] = state
    st.session_state["oauth_state_provider"] = provider
    if provider == "google":
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "online",
            "prompt": "select_account",
            "state": state,
        }
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return None


def _provision_social_resident(profile):
    """Create or reuse a resident account from a verified OAuth profile."""
    email = str(profile.get("email", "")).strip().lower()
    name = str(profile.get("name", "Citizen")).strip() or "Citizen"
    if not email:
        return None, "The provider did not return an email address."
    existing = db_query(
        "SELECT resident_id, zone_id, name, phone, email, username, active FROM residents WHERE lower(email)=? LIMIT 1",
        (email,),
    )
    zones = get_zones()
    if zones.empty:
        return None, "No project-defined zones are available."
    zone_id = int(zones.iloc[0]["zone_id"])
    if not existing.empty:
        resident_id = int(existing.iloc[0]["resident_id"])
        db_execute(
            "UPDATE residents SET name=?, email=?, active=1 WHERE resident_id=?",
            (name, email, resident_id),
        )
    else:
        base = re.sub(r"[^a-z0-9]", "", email.split("@")[0].lower()) or "citizen"
        username = base[:28]
        suffix = 1
        while not db_query("SELECT resident_id FROM residents WHERE lower(username)=? LIMIT 1", (username,)).empty:
            username = f"{base[:24]}{suffix}"
            suffix += 1
        resident_id = db_execute(
            "INSERT INTO residents (zone_id,name,phone,email,username,password_hash,active) VALUES (?,?,?,?,?,?,1)",
            (zone_id, name, "", email, username, "", 1),
        )
    resident = get_resident(int(resident_id))
    if resident.empty:
        return None, "Unable to load the AquaTrack account."
    return resident.iloc[0].to_dict(), None


def _finish_social_login(profile, provider):
    resident, error = _provision_social_resident(profile)
    if error:
        st.session_state.citizen_login_error = error
        return False
    st.session_state.citizen_logged_in = True
    st.session_state.citizen_user = resident
    st.session_state.resident_id = int(resident["resident_id"])
    st.session_state.resident_zone_id = int(resident["zone_id"])
    st.session_state.citizen_login_error = None
    st.session_state.oauth_provider = provider
    st.session_state.page = st.session_state.pop("citizen_after_login", "Dashboard")
    return True


def _handle_oauth_callback():
    """Complete Google OAuth callback when the provider redirects back."""
    try:
        qp = st.query_params
        code = qp.get("code")
        state = qp.get("state")
        provider = qp.get("oauth")
        error = qp.get("error")
    except Exception:
        return False
    if not (code or error):
        return False
    provider = str(provider).lower() if provider else ""
    # The provider is intentionally not part of the registered callback URL.
    # Recover it from the OAuth state generated when the user clicked the button.
    if not provider and state:
        for candidate in ("google",):
            expected = str(st.session_state.get(f"oauth_state_{candidate}", ""))
            if expected and secrets.compare_digest(expected, str(state)):
                provider = candidate
                break
    if provider not in ("google",):
        st.session_state.citizen_login_error = "Unable to identify the sign-in provider. Please try again."
        try: st.query_params.clear()
        except Exception: pass
        return True
    if error:
        st.session_state.citizen_login_error = f"{provider.title()} sign-in was cancelled or denied."
        try: st.query_params.clear()
        except Exception: pass
        return True
    expected = st.session_state.get(f"oauth_state_{provider}")
    if not expected or not state or not secrets.compare_digest(str(expected), str(state)):
        st.session_state.citizen_login_error = "OAuth security check failed. Please try again."
        try: st.query_params.clear()
        except Exception: pass
        return True
    client_id, client_secret, redirect_uri = _oauth_config(provider)
    if not client_id or not client_secret:
        st.session_state.citizen_login_error = f"{provider.title()} OAuth is not configured."
        try: st.query_params.clear()
        except Exception: pass
        return True
    try:
        if provider == "google":
            token_resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code, "client_id": client_id, "client_secret": client_secret,
                    "redirect_uri": redirect_uri, "grant_type": "authorization_code",
                }, timeout=15,
            )
            token_resp.raise_for_status()
            access_token = token_resp.json().get("access_token")
            profile_resp = requests.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}, timeout=15,
            )
            profile_resp.raise_for_status()
            profile = profile_resp.json()
            if not profile.get("email") or profile.get("email_verified") is False:
                raise ValueError("Google did not return a verified email address.")
            profile = {"email": profile.get("email"), "name": profile.get("name") or profile.get("given_name") or "Citizen"}
        else:
            return False
        ok = _finish_social_login(profile, provider)
        try: st.query_params.clear()
        except Exception: pass
        return ok
    except Exception as exc:
        LOGGER.exception("OAuth callback failed for %s", provider)
        st.session_state.citizen_login_error = f"{provider.title()} sign-in failed. Please try again."
        try: st.query_params.clear()
        except Exception: pass
        return True


if st.session_state.page == "Admin Panel" and not st.session_state.admin_logged_in:
    st.session_state.page = "Admin Login"
if st.session_state.page == "Worker Dashboard" and not st.session_state.worker_logged_in:
    st.session_state.page = "Worker Login"

if _handle_oauth_callback():
    st.rerun()

# Password reset links are handled as a dedicated authentication screen.
try:
    _reset_token = str(st.query_params.get("reset_token", "")).strip()
    if _reset_token and not st.session_state.get("citizen_logged_in", False):
        st.session_state.page = "Citizen Reset Password"
except Exception:
    _reset_token = ""

# Small URL action used by the inline authentication links.
# This keeps the visible "Sign Up" text itself clickable without rendering
# a second full-width Streamlit button underneath it.
try:
    _auth_action = str(st.query_params.get("auth", "")).lower().strip()
    if _auth_action == "signup" and not st.session_state.get("citizen_logged_in", False):
        st.session_state.page = "Citizen Register"
        st.query_params.clear()
    elif _auth_action == "login" and not st.session_state.get("citizen_logged_in", False):
        st.session_state.page = "Citizen Login"
        st.query_params.clear()
    elif _auth_action == "forgot" and not st.session_state.get("citizen_logged_in", False):
        st.session_state.page = "Citizen Forgot Password"
        st.query_params.clear()
except Exception:
    pass

page = st.session_state.page


# ============================================================
# HEADER NAVIGATION
# ============================================================

# Mobile navigation: kebab button on the left, centered brand,
# and admin-only notifications beside the language control.
mobile_header_cols = st.columns([0.9, 3.7, 1.0, 1.2], gap="small", vertical_alignment="center")

with mobile_header_cols[0]:
    st.button(
        "×" if st.session_state.mobile_menu_open else "⋮",
        key="mobile_menu_toggle",
        width="stretch",
        on_click=toggle_mobile_menu,
        type="secondary",
    )

with mobile_header_cols[1]:
    st.markdown(
        """
        <div class="aqua-mobile-header-marker" aria-hidden="true"></div>
        <div class="aqua-reference-brand aqua-mobile-brand">
            <div class="aqua-reference-logo">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 3.2C12 3.2 6.4 10.05 6.4 14.35C6.4 17.55 8.9 20.1 12 20.1C15.1 20.1 17.6 17.55 17.6 14.35C17.6 17.55 15.1 20.1 12 20.1C8.9 20.1 6.4 17.55 6.4 14.35C6.4 10.05 12 3.2 12 3.2Z" stroke="white" stroke-width="1.65"/>
                </svg>
            </div>
            <div>
                <div class="aqua-reference-name">AquaTrack</div>
                <div class="aqua-mobile-tagline">Predictive Water Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mobile_header_cols[2]:
    if st.session_state.get("admin_logged_in"):
        mobile_unread = get_unread_count("admin", None)
        mobile_notification_label = f"🔔 {mobile_unread}" if mobile_unread else "🔔"
        st.button(
            mobile_notification_label,
            key="mobile_admin_notifications_top",
            width="stretch",
            on_click=navigate_to,
            args=("Admin Panel",),
            type="primary" if mobile_unread else "secondary",
        )

with mobile_header_cols[3]:
    mobile_language_label = "हिंदी में देखें" if st.session_state.language == "English" else "View in English"
    st.button(
        mobile_language_label,
        key="mobile_language_top",
        width="stretch",
        on_click=mobile_toggle_language,
        type="secondary",
    )

if st.session_state.mobile_menu_open:
    with st.container():
        st.markdown('<div class="aqua-mobile-menu-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
        st.button("×", key="mobile_menu_close", width="content", on_click=close_mobile_menu, type="secondary")
        st.markdown('<div class="aqua-mobile-drawer-title" style="display:inline-block;">AquaTrack Menu</div>', unsafe_allow_html=True)
        if st.session_state.admin_logged_in:
            # Authenticated admins stay inside the administration workspace.
            # Public portal navigation is intentionally hidden in this state.
            mobile_items = [
                ("⚙️ " + t("Admin Panel"), "Admin Panel", "mobile_admin_panel"),
            ]
        elif st.session_state.worker_logged_in:
            mobile_items = [
                ("👷 " + t("Worker Dashboard"), "Worker Dashboard", "mobile_worker_dashboard"),
            ]
        elif st.session_state.get("citizen_logged_in", False):
            # Citizen-only actions become visible after a successful citizen login.
            mobile_items = [
                ("⌂ " + t("Home"), "Dashboard", "mobile_home_citizen"),
                ("▣ " + t("My Reports"), "My Requests", "mobile_requests_citizen"),
                ("＋ " + t("Report Issue"), "Report Water Body Issue", "mobile_report_citizen"),
                ("◷ " + t("Notifications"), "Resident Alerts", "mobile_alerts_citizen"),
                ("☆ " + t("Reviews"), "Reviews", "mobile_reviews_citizen"),
                ("☰ " + t("More"), "About & Policies", "mobile_more_citizen"),
            ]
        else:
            # Visitors do not see the citizen Report / Alerts / Reviews actions.
            mobile_items = [
                ("⌂ " + t("Home"), "Dashboard", "mobile_home"),
                ("▣ " + t("My Reports"), "My Requests", "mobile_requests"),
                ("☰ " + t("More"), "About & Policies", "mobile_more"),
            ]
        for label, target, key in mobile_items:
            if target == "__ADMIN__":
                st.button(label, key=key, width="stretch", on_click=mobile_open_admin,
                          type="primary" if page in ("Admin Panel", "Admin Login") else "secondary")
            elif target == "__WORKER__":
                st.button(label, key=key, width="stretch", on_click=mobile_open_worker,
                          type="primary" if page in ("Worker Dashboard", "Worker Login") else "secondary")
            else:
                st.button(label, key=key, width="stretch", on_click=mobile_navigate, args=(target,),
                          type="primary" if page == target else "secondary")
        if st.session_state.admin_logged_in or st.session_state.worker_logged_in:
            st.button(t("Logout"), key="mobile_logout", width="stretch", on_click=mobile_logout, type="secondary")

# Desktop header
if st.session_state.admin_logged_in:
    header_cols = st.columns([6.15, 1.15, 1.0, 1.15], gap="small", vertical_alignment="center")
elif st.session_state.worker_logged_in:
    header_cols = st.columns([6.15, 1.15, 1.25, 1.15], gap="small", vertical_alignment="center")
elif st.session_state.get("citizen_logged_in", False):
    # Authenticated citizens get the citizen portal navigation only.
    # Admin and Worker entry points are intentionally hidden after citizen login.
    header_cols = st.columns([2.35, 0.78, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90], gap="small", vertical_alignment="center")
else:
    # Keep Language, Home, Map, Worker, Admin, and Login together in one
    # continuous right-side navigation group.
    header_cols = st.columns(
        [1.75, 0.78, 0.90, 0.82, 0.90, 0.90, 0.90],
        gap="small",
        vertical_alignment="center",
    )

with header_cols[0]:
    st.markdown(
        """
        <div class="aqua-reference-brand aqua-desktop-header-marker">
            <div class="aqua-reference-logo">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 3.2C12 3.2 6.4 10.05 6.4 14.35C6.4 17.55 8.9 20.1 12 20.1C15.1 20.1 17.6 17.55 17.6 14.35C17.6 10.05 12 3.2 12 3.2Z" stroke="white" stroke-width="1.65"/>
                </svg>
            </div>
            <div>
                <div class="aqua-reference-name">AquaTrack</div>
                <div style="font-size:.61rem;color:#8aa0ad;font-weight:600;letter-spacing:.02em;margin-top:-2px;">Predictive Water Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

language_button_label = "हिंदी में देखें" if st.session_state.language == "English" else "View in English"
if st.session_state.admin_logged_in:
    header_cols[1].button(language_button_label,key="header_language_toggle",width="stretch",on_click=toggle_language,type="secondary")
    header_unread=get_unread_count("admin",None)
    header_cols[2].button(f"Notification ({header_unread})" if header_unread else "Notification",key="header_admin_notifications",width="stretch",on_click=open_admin_notification_popup,type="primary" if header_unread else "secondary")
    header_cols[3].button(t("Logout"),key="header_logout",width="stretch",on_click=do_logout,type="secondary")
elif st.session_state.worker_logged_in:
    header_cols[1].button(language_button_label,key="header_language_toggle",width="stretch",on_click=toggle_language,type="secondary")
    worker_header_id = int(st.session_state.worker_user["worker_id"])
    worker_header_unread = get_unread_count("worker", worker_header_id)
    header_cols[2].button(
        f"Notification ({worker_header_unread})" if worker_header_unread else "Notification",
        key="header_worker_notifications",
        width="stretch",
        on_click=open_worker_notification_popup,
        type="primary" if worker_header_unread else "secondary",
    )
    header_cols[3].button(t("Logout"),key="header_logout",width="stretch",on_click=do_logout,type="secondary")
elif st.session_state.get("citizen_logged_in", False):
    # Citizen-only header: no Admin, Worker, or Login buttons.
    header_cols[1].button(language_button_label,key="header_language_toggle_citizen",width="stretch",on_click=toggle_language,type="secondary")
    header_cols[2].button("Home",key="header_home_citizen",width="stretch",on_click=navigate_to,args=("Dashboard",),type="primary" if page=="Dashboard" else "secondary")
    header_cols[3].button("Report",key="header_report_citizen",width="stretch",on_click=require_citizen_login,args=("Report Water Body Issue",),type="primary" if page=="Report Water Body Issue" else "secondary")
    header_cols[4].button("Alerts",key="header_alerts_citizen",width="stretch",on_click=require_citizen_login,args=("Resident Alerts",),type="primary" if page=="Resident Alerts" else "secondary")
    header_cols[5].button("Reviews",key="header_reviews_citizen",width="stretch",on_click=require_citizen_login,args=("Reviews",),type="primary" if page=="Reviews" else "secondary")
    header_cols[6].button("Map",key="header_map_citizen",width="stretch",on_click=navigate_to,args=("Indore Zone Map",),type="primary" if page=="Indore Zone Map" else "secondary")
    header_cols[7].button("Logout",key="header_citizen_logout",width="stretch",on_click=do_citizen_logout,type="secondary")
else:
    # Public visitor header: Report, Alerts, and Reviews are shown only after citizen login.
    # Keep the Login button at the far right, with Language and Home immediately beside Map.
    header_cols[1].button(language_button_label,key="header_language_toggle",width="stretch",on_click=toggle_language,type="secondary")
    header_cols[2].button("Home",key="header_home",width="stretch",on_click=navigate_to,args=("Dashboard",),type="primary" if page=="Dashboard" else "secondary")
    header_cols[3].button("Map",key="header_map",width="stretch",on_click=navigate_to,args=("Indore Zone Map",),type="primary" if page=="Indore Zone Map" else "secondary")
    header_cols[4].button("Worker",key="header_worker",width="stretch",on_click=open_worker,type="secondary")
    header_cols[5].button("Admin",key="header_admin",width="stretch",on_click=open_admin,type="secondary")
    header_cols[6].button("Login",key="header_citizen_login",width="stretch",on_click=open_citizen_login,type="secondary")


if st.session_state.get("admin_logged_in") and st.session_state.get("admin_notification_popup"):
    show_admin_notification_popup()

if st.session_state.get("worker_logged_in") and st.session_state.get("worker_notification_popup"):
    show_worker_notification_popup()


# Analytics sub-navigation is available only to authenticated citizens.
# Visitors, workers, and administrators do not see these public-portal options.
if (
    st.session_state.get("citizen_logged_in", False)
    and not st.session_state.get("admin_logged_in", False)
    and not st.session_state.get("worker_logged_in", False)
):
    SUBNAV = [
        (t("📊 Analytics"), "Zone Analytics"),
        (t("⚖️ Compare"), "Compare Zones"),
        (t("💧 Quality"), "Water Quality Prediction"),
        (t("📈 Forecast"), "Availability Forecast"),
        (t("ℹ️ About"), "About & Policies"),
    ]
else:
    SUBNAV = []

SUBNAV_PAGES = {"Dashboard", "Indore Zone Map"} | {target for _, target in SUBNAV}

if page in SUBNAV_PAGES and page != "Indore Zone Map" and SUBNAV:
    st.markdown('<div class="aqua-subnav-gap" aria-hidden="true"></div>', unsafe_allow_html=True)
    subnav_cols = st.columns(len(SUBNAV), gap="small")
    for index, (col, (label, target)) in enumerate(zip(subnav_cols, SUBNAV)):
        # Keep the phone-only marker AFTER the button so it never adds
        # vertical space above Analytics on desktop/tablet.
        col.button(label, key=f"subnav_{target}", width="stretch", on_click=navigate_to, args=(target,), type="primary" if page == target else "secondary")
        if index == 0:
            # Phone-only marker: this row is duplicated by the mobile sidebar.
            # It remains inside the row so the phone :has() selector still works.
            col.markdown('<span class="aqua-home-subnav-marker" aria-hidden="true"></span>', unsafe_allow_html=True)


# ============================================================
# ABOUT, POLICIES & SUPPORT
# ============================================================

if page == "About & Policies":
    st.markdown('<div class="about-policies-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        /* About & Policies: compact, clean content flow without a repeated disclaimer. */
        body:has(.about-policies-marker) .main .block-container {
            padding-top: .15rem !important;
            padding-bottom: 0 !important;
        }
        body:has(.about-policies-marker) h1 {
            margin-top: .45rem !important;
            margin-bottom: .2rem !important;
        }
        body:has(.about-policies-marker) .aqua-footer {
            margin-top: 1.1rem !important;
        }
        body:has(.about-policies-marker) button[data-baseweb="tab"] {
            padding: .55rem .7rem !important;
            font-size: .84rem !important;
        }
        body:has(.about-policies-marker) [data-testid="stTabsContent"] {
            padding-top: .35rem !important;
        }
        body:has(.about-policies-marker) [data-testid="stTabsContent"] h2,
        body:has(.about-policies-marker) [data-testid="stTabsContent"] h3 {
            margin-top: .55rem !important;
            margin-bottom: .45rem !important;
        }
        body:has(.about-policies-marker) [data-testid="stTabsContent"] p {
            line-height: 1.55 !important;
            margin-bottom: .55rem !important;
        }
        body:has(.about-policies-marker) [data-testid="stTabsContent"] ul {
            margin-top: .25rem !important;
            margin-bottom: .45rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(t("ℹ️ About & Policies"))
    st.write(t("AquaTrack is a project-level decision-support prototype for water-quality assessment and water-availability forecasting in Indore."))

    about_tab, scope_tab, privacy_tab, terms_tab, transparency_tab = st.tabs([
        t("About AquaTrack"), t("Project Scope"), t("Privacy Policy"), t("Terms of Use"),
        t("Data & Model Transparency"),
    ])

    with about_tab:
        st.subheader(t("About AquaTrack"))
        st.write(t("The application combines water-quality screening, availability forecasting, zone analytics, maps, issue reporting, alerts, workflow management and report generation in one Streamlit application."))
        st.write(t("Project team: Anush Parmar, Anuj Patidar, Arpit Jangid and Bhushan Bondre. Guide: Mr. Sumit Kumar. IPS Academy, Institute of Engineering & Science, Indore."))
        st.subheader(t("What the system provides"))
        for item in [
            "Water-quality screening from pH, TDS, turbidity, hardness, chloride, fluoride and nitrate.",
            "Water-availability forecasting using previous supply hours, rainfall and season.",
            "Zone-wise analytics, comparison and map views.",
            "Resident issue reporting with request tracking and notifications.",
            "Worker field workflow and administrator verification.",
            "PDF/Excel exports, audit logging and database backup controls.",
        ]:
            st.markdown(f"- {t(item)}")

    with scope_tab:
        st.subheader(t("Project Scope"))
        st.info(t(DISCLAIMER))
        st.markdown("- Project-generated/synthetic data are used for development and demonstration.")
        st.markdown("- Project-defined zones, water bodies and served-area relationships are demonstration relationships, not official municipal records.")
        st.markdown("- Model outputs are screening indicators for academic decision support.")
        st.markdown("- Certified laboratory testing, official municipal measurements and operational decisions remain outside this prototype.")
        st.markdown("- Custom domain and HTTPS are intentionally deferred until deployment.")
        st.markdown("- Payment, checkout and commercial transaction functionality are not part of AquaTrack.")

    with privacy_tab:
        st.subheader(t("Privacy Policy"))
        st.write("AquaTrack is an academic prototype. Information entered through issue reporting or alert registration is used only to demonstrate request tracking, notifications and project workflows.")
        st.markdown("- Do not enter passwords, financial information, government IDs or other sensitive information into report descriptions.")
        st.markdown("- Contact details are stored in the local project database when required for request tracking or alerts.")
        st.markdown("- Uploaded issue photos are stored with the local prototype data directory.")
        st.markdown("- SMTP credentials, API keys and other secrets must be supplied through deployment configuration and must not be committed to source code.")
        st.markdown("- For a real deployment, retention, access control and data deletion procedures should be defined before collecting public data.")

    with terms_tab:
        st.subheader(t("Terms of Use"))
        st.write("AquaTrack is provided for academic demonstration, analysis and prototype-level planning.")
        st.markdown("- Results are not guarantees of water safety, supply continuity or future municipal conditions.")
        st.markdown("- Users should not treat model output as a replacement for certified laboratory testing or official instructions.")
        st.markdown("- The project team does not represent project-generated zone or water-body relationships as official municipal data.")
        st.markdown("- Use of the prototype should follow applicable institutional, privacy and information-security requirements.")
        st.markdown("- No payment, purchase, subscription or financial transaction is supported by this application.")

    with transparency_tab:
        st.subheader(t("Data & Model Transparency"))
        st.markdown("**Data:** Project-generated/synthetic water-quality and availability records are used for demonstration.")
        st.markdown("**Quality model:** Random Forest classifier using pH, TDS, turbidity, hardness, chloride, fluoride and nitrate.")
        st.markdown("**Availability model:** Linear Regression using previous supply hours, rainfall and season code; forecasts are bounded to 0–24 hours for presentation.")
        st.markdown("**Reference framework:** Water-quality screening uses the project implementation's BIS IS 10500:2012-based limits.")
        st.markdown("**Evaluation:** Administrator model-performance views use holdout-style evaluation where supported by the implementation; these are not claims of real-world municipal accuracy.")
        st.markdown("**Limitations:** Real-world validation with authorised laboratory and municipal data is future work.")




# ============================================================
# WORKER LOGIN
# ============================================================

# ============================================================
# WORKER LOGIN
# ============================================================

if page == "Worker Login":
    import base64
    st.markdown('<div class="restricted-auth-page-marker worker-restricted-auth-marker" aria-hidden="true"></div>', unsafe_allow_html=True)

    _auth_bg = ""
    _auth_bg_mime = "image/jpeg"
    for _candidate in (
        BASE_DIR / "assets" / "aqua_auth_background.jpg",
        BASE_DIR / "assets" / "aqua_auth_background.png",
        BASE_DIR / "aqua_auth_background.png",
        BASE_DIR / "assets" / "aqua_auth_bg.jpg",
        BASE_DIR / "aqua_auth_bg.jpg",
    ):
        try:
            if _candidate.exists():
                _auth_bg = base64.b64encode(_candidate.read_bytes()).decode("ascii")
                _auth_bg_mime = "image/jpeg" if _candidate.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                break
        except Exception:
            continue
    _bg_css = f"url(data:{_auth_bg_mime};base64,{_auth_bg})" if _auth_bg else "linear-gradient(135deg,#f2e8da 0%,#f2eef3 52%,#ece7f4 100%)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    html:has(.restricted-auth-page-marker),
    body:has(.restricted-auth-page-marker) {{
        background:{_bg_css} center/cover fixed no-repeat !important;
    }}
    body:has(.restricted-auth-page-marker) .stApp,
    body:has(.restricted-auth-page-marker) [data-testid="stAppViewContainer"],
    body:has(.restricted-auth-page-marker) section[data-testid="stMain"] {{
        background:transparent !important;
    }}
    body:has(.restricted-auth-page-marker) .main .block-container {{
        width:100% !important; max-width:none !important;
        min-height:100vh !important; padding:0 !important;
        display:flex !important; align-items:center !important; justify-content:center !important;
    }}
    body:has(.restricted-auth-page-marker) .aqua-footer,
    body:has(.restricted-auth-page-marker) .aqua-subnav-gap,
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker),
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker),
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-menu-marker),
    body:has(.restricted-auth-page-marker) [data-testid="stToolbar"] {{ display:none !important; }}

    div.st-key-worker_auth_back_public,
    div.st-key-admin_auth_back_public {{
        position:fixed !important; top:20px !important; left:24px !important; z-index:99999 !important;
        width:auto !important; margin:0 !important; padding:0 !important;
    }}
    div.st-key-worker_auth_back_public button,
    div.st-key-admin_auth_back_public button {{
        min-width:150px !important; width:150px !important; height:40px !important; min-height:40px !important;
        padding:0 14px !important; border:0 !important; border-radius:22px !important;
        background:rgba(255,255,255,.94) !important; color:#243247 !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:13px !important; font-weight:700 !important;
        box-shadow:0 8px 22px rgba(24,35,55,.14) !important;
    }}
    div.st-key-worker_auth_back_public button p,
    div.st-key-worker_auth_back_public button span,
    div.st-key-worker_auth_back_public button div,
    div.st-key-admin_auth_back_public button p,
    div.st-key-admin_auth_back_public button span,
    div.st-key-admin_auth_back_public button div {{
        color:#243247 !important; font-family:'Poppins',Arial,sans-serif !important;
        font-size:13px !important; font-weight:700 !important;
    }}

    div.st-key-worker_auth_card,
    div.st-key-admin_auth_card {{
        width:min(520px,calc(100vw - 32px)) !important;
        max-width:520px !important; margin:0 auto !important;
        padding:30px 30px 25px !important; box-sizing:border-box !important;
        border:1px solid rgba(255,255,255,.96) !important; border-radius:25px !important;
        background:rgba(255,255,255,.72) !important;
        box-shadow:0 22px 65px rgba(24,35,55,.20) !important;
        backdrop-filter:blur(18px) saturate(125%) !important;
        -webkit-backdrop-filter:blur(18px) saturate(125%) !important;
        font-family:'Poppins',Arial,sans-serif !important;
    }}
    div.st-key-worker_auth_card > div,
    div.st-key-admin_auth_card > div {{ gap:.28rem !important; }}
    div.st-key-worker_auth_card [data-testid="stForm"],
    div.st-key-admin_auth_card [data-testid="stForm"] {{
        border:0 !important; padding:0 !important; background:transparent !important; box-shadow:none !important;
    }}
    div.st-key-worker_auth_card [data-testid="stTextInput"],
    div.st-key-admin_auth_card [data-testid="stTextInput"] {{ margin:0 0 10px !important; }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] label,
    div.st-key-admin_auth_card [data-testid="stTextInput"] label {{ display:none !important; }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] input,
    div.st-key-admin_auth_card [data-testid="stTextInput"] input {{
        height:38px !important; min-height:38px !important; box-sizing:border-box !important;
        border:1px solid #dbe1ea !important; border-radius:13px !important;
        background:rgba(255,255,255,.90) !important; color:#202b3e !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:12px !important; font-weight:500 !important;
        padding:0 12px !important; box-shadow:none !important;
    }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] input::placeholder,
    div.st-key-admin_auth_card [data-testid="stTextInput"] input::placeholder {{ color:#91a0b5 !important; opacity:1 !important; }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] div[data-baseweb="input"],
    div.st-key-admin_auth_card [data-testid="stTextInput"] div[data-baseweb="input"] {{ border-radius:13px !important; background:transparent !important; box-shadow:none !important; }}
    div.st-key-worker_auth_card .restricted-auth-options,
    div.st-key-admin_auth_card .restricted-auth-options {{
        display:flex !important; align-items:center !important; justify-content:space-between !important;
        width:100% !important; margin:0 0 12px !important; min-height:24px !important;
    }}
    div.st-key-worker_auth_card .restricted-auth-options > div,
    div.st-key-admin_auth_card .restricted-auth-options > div {{
        margin:0 !important; padding:0 !important;
    }}
    div.st-key-worker_auth_card .restricted-auth-options [data-testid="stCheckbox"],
    div.st-key-admin_auth_card .restricted-auth-options [data-testid="stCheckbox"] {{
        margin:0 !important; padding:0 !important;
    }}
    div.st-key-worker_auth_card .restricted-auth-options [data-testid="stCheckbox"] label,
    div.st-key-admin_auth_card .restricted-auth-options [data-testid="stCheckbox"] label {{
        color:#536276 !important; font-family:'Poppins',Arial,sans-serif !important;
        font-size:12px !important; font-weight:500 !important;
    }}
    div.st-key-worker_auth_card .restricted-auth-forgot,
    div.st-key-admin_auth_card .restricted-auth-forgot {{
        text-align:right !important; white-space:nowrap !important; padding-top:1px !important;
    }}
    div.st-key-worker_auth_card .restricted-auth-forgot a,
    div.st-key-admin_auth_card .restricted-auth-forgot a {{
        color:#0085a8 !important; text-decoration:none !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:12px !important; font-weight:600 !important;
    }}
    div.st-key-worker_auth_card .restricted-auth-forgot a:hover,
    div.st-key-admin_auth_card .restricted-auth-forgot a:hover {{ text-decoration:underline !important; }}
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button {{
        width:100% !important; height:40px !important; min-height:40px !important; margin-top:5px !important;
        border:0 !important; border-radius:12px !important;
        background:linear-gradient(135deg,#0e9ed3,#12b5c1) !important; color:#fff !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:13px !important; font-weight:700 !important;
        box-shadow:0 12px 22px rgba(14,158,211,.22) !important;
    }}
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button p,
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button span,
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button div,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button p,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button span,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button div {{
        color:#fff !important; font-family:'Poppins',Arial,sans-serif !important;
        font-size:13px !important; font-weight:700 !important;
    }}
    .restricted-auth-brand {{ text-align:center; margin:0 0 19px; }}
    .restricted-auth-logo {{
        width:30px;height:30px;margin:0 auto 6px;border-radius:50%;display:grid;place-items:center;background:#087f9b;
        box-shadow:0 7px 18px rgba(8,127,155,.20);
    }}
    .restricted-auth-logo svg {{ width:17px;height:17px; }}
    .restricted-auth-brand-name {{ color:#10182b;font-size:18px;line-height:1.1;font-weight:800;letter-spacing:-.035em; }}
    .restricted-auth-kicker {{ margin:0 0 6px;text-align:left;color:#087f9b;font-size:11px;line-height:1.1;font-weight:800;letter-spacing:.14em;text-transform:uppercase; }}
    .restricted-auth-title {{ margin:0 0 7px;text-align:left;color:#10182b;font-size:27px;line-height:1.12;font-weight:800;letter-spacing:-.045em; }}
    .restricted-auth-copy {{ margin:0 0 18px;text-align:left;color:#536276;font-size:12px;line-height:1.5; }}
    @media(max-width:700px) {{
        body:has(.restricted-auth-page-marker) .main .block-container {{ min-height:100svh !important; }}
        div.st-key-worker_auth_card, div.st-key-admin_auth_card {{ width:min(520px,calc(100vw - 20px)) !important; padding:26px 22px 23px !important; border-radius:23px !important; }}
        div.st-key-worker_auth_back_public, div.st-key-admin_auth_back_public {{ top:14px !important; left:14px !important; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="worker_auth_card"):
        st.markdown(f"""
            <div class="restricted-auth-brand">
              <div class="restricted-auth-logo"><svg viewBox="0 0 32 32" fill="none"><path d="M16 3.8C16 3.8 8.4 13 8.4 18.2C8.4 23.05 11.82 27.1 16 27.1C20.18 27.1 23.6 23.05 23.6 18.2C23.6 13 16 3.8 16 3.8Z" stroke="white" stroke-width="2.2"/></svg></div>
              <div class="restricted-auth-brand-name">AquaTrack</div>
            </div>
            <div class="restricted-auth-kicker">{t("Restricted Access")}</div>
            <div class="restricted-auth-title">{t("Worker Login")}</div>
            <div class="restricted-auth-copy">{t("Sign in to view assigned field-service requests and update work progress.")}</div>
        """, unsafe_allow_html=True)
        with st.form("worker_login_form"):
            st.text_input(t("Worker Username"), key="worker_username", value=st.session_state.get("worker_remember_username", ""), placeholder=t("Worker Username"), icon=":material/mail:")
            st.text_input(t("Worker Password"), type="password", key="worker_password", placeholder=t("Worker Password"), icon=":material/lock:")
            option_cols = st.columns([1, 1], gap="small")
            with option_cols[0]:
                worker_remember = st.checkbox("Remember Me", value=False, key="worker_remember_me")
            with option_cols[1]:
                st.markdown('<div class="restricted-auth-forgot"><a href="#" onclick="return false;" aria-label="Forgot Password">Forgot Password?</a></div>', unsafe_allow_html=True)
            st.form_submit_button(t("Login →"), width="stretch", type="primary", on_click=attempt_worker_login)
            if worker_remember:
                st.session_state.worker_remember_username = st.session_state.get("worker_username", "")
            else:
                st.session_state.pop("worker_remember_username", None)
            if st.session_state.worker_login_error:
                st.toast(t(st.session_state.worker_login_error), icon="❌")

    with st.container(key="worker_auth_back_public"):
        st.button(t("← Back to Home"), key="worker_back_public_new", width="content", type="secondary", on_click=navigate_to, args=("Dashboard",))


# ============================================================
# ADMIN LOGIN
# ============================================================

elif page == "Admin Login":
    import base64
    st.markdown('<div class="restricted-auth-page-marker admin-restricted-auth-marker" aria-hidden="true"></div>', unsafe_allow_html=True)

    _auth_bg = ""
    _auth_bg_mime = "image/jpeg"
    for _candidate in (
        BASE_DIR / "assets" / "aqua_auth_background.jpg",
        BASE_DIR / "assets" / "aqua_auth_background.png",
        BASE_DIR / "aqua_auth_background.png",
        BASE_DIR / "assets" / "aqua_auth_bg.jpg",
        BASE_DIR / "aqua_auth_bg.jpg",
    ):
        try:
            if _candidate.exists():
                _auth_bg = base64.b64encode(_candidate.read_bytes()).decode("ascii")
                _auth_bg_mime = "image/jpeg" if _candidate.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                break
        except Exception:
            continue
    _bg_css = f"url(data:{_auth_bg_mime};base64,{_auth_bg})" if _auth_bg else "linear-gradient(135deg,#f2e8da 0%,#f2eef3 52%,#ece7f4 100%)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    html:has(.restricted-auth-page-marker), body:has(.restricted-auth-page-marker) {{ background:{_bg_css} center/cover fixed no-repeat !important; }}
    body:has(.restricted-auth-page-marker) .stApp,
    body:has(.restricted-auth-page-marker) [data-testid="stAppViewContainer"],
    body:has(.restricted-auth-page-marker) section[data-testid="stMain"] {{ background:transparent !important; }}
    body:has(.restricted-auth-page-marker) .main .block-container {{ width:100% !important; max-width:none !important; min-height:100vh !important; padding:0 !important; display:flex !important; align-items:center !important; justify-content:center !important; }}
    body:has(.restricted-auth-page-marker) .aqua-footer,
    body:has(.restricted-auth-page-marker) .aqua-subnav-gap,
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker),
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker),
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-menu-marker),
    body:has(.restricted-auth-page-marker) [data-testid="stToolbar"] {{ display:none !important; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    html:has(.restricted-auth-page-marker),
    body:has(.restricted-auth-page-marker) {{
        background:{_bg_css} center/cover fixed no-repeat !important;
    }}
    body:has(.restricted-auth-page-marker) .stApp,
    body:has(.restricted-auth-page-marker) [data-testid="stAppViewContainer"],
    body:has(.restricted-auth-page-marker) section[data-testid="stMain"] {{
        background:transparent !important;
    }}
    body:has(.restricted-auth-page-marker) .main .block-container {{
        width:100% !important; max-width:none !important;
        min-height:100vh !important; padding:0 !important;
        display:flex !important; align-items:center !important; justify-content:center !important;
    }}
    body:has(.restricted-auth-page-marker) .aqua-footer,
    body:has(.restricted-auth-page-marker) .aqua-subnav-gap,
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker),
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker),
    body:has(.restricted-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-menu-marker),
    body:has(.restricted-auth-page-marker) [data-testid="stToolbar"] {{ display:none !important; }}

    div.st-key-worker_auth_back_public,
    div.st-key-admin_auth_back_public {{
        position:fixed !important; top:20px !important; left:24px !important; z-index:99999 !important;
        width:auto !important; margin:0 !important; padding:0 !important;
    }}
    div.st-key-worker_auth_back_public button,
    div.st-key-admin_auth_back_public button {{
        min-width:150px !important; width:150px !important; height:40px !important; min-height:40px !important;
        padding:0 14px !important; border:0 !important; border-radius:22px !important;
        background:rgba(255,255,255,.94) !important; color:#243247 !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:13px !important; font-weight:700 !important;
        box-shadow:0 8px 22px rgba(24,35,55,.14) !important;
    }}
    div.st-key-worker_auth_back_public button p,
    div.st-key-worker_auth_back_public button span,
    div.st-key-worker_auth_back_public button div,
    div.st-key-admin_auth_back_public button p,
    div.st-key-admin_auth_back_public button span,
    div.st-key-admin_auth_back_public button div {{
        color:#243247 !important; font-family:'Poppins',Arial,sans-serif !important;
        font-size:13px !important; font-weight:700 !important;
    }}

    div.st-key-worker_auth_card,
    div.st-key-admin_auth_card {{
        width:min(520px,calc(100vw - 32px)) !important;
        max-width:520px !important; margin:0 auto !important;
        padding:30px 30px 25px !important; box-sizing:border-box !important;
        border:1px solid rgba(255,255,255,.96) !important; border-radius:25px !important;
        background:rgba(255,255,255,.72) !important;
        box-shadow:0 22px 65px rgba(24,35,55,.20) !important;
        backdrop-filter:blur(18px) saturate(125%) !important;
        -webkit-backdrop-filter:blur(18px) saturate(125%) !important;
        font-family:'Poppins',Arial,sans-serif !important;
    }}
    div.st-key-worker_auth_card > div,
    div.st-key-admin_auth_card > div {{ gap:.28rem !important; }}
    div.st-key-worker_auth_card [data-testid="stForm"],
    div.st-key-admin_auth_card [data-testid="stForm"] {{
        border:0 !important; padding:0 !important; background:transparent !important; box-shadow:none !important;
    }}
    div.st-key-worker_auth_card [data-testid="stTextInput"],
    div.st-key-admin_auth_card [data-testid="stTextInput"] {{ margin:0 0 10px !important; }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] label,
    div.st-key-admin_auth_card [data-testid="stTextInput"] label {{ display:none !important; }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] input,
    div.st-key-admin_auth_card [data-testid="stTextInput"] input {{
        height:38px !important; min-height:38px !important; box-sizing:border-box !important;
        border:1px solid #dbe1ea !important; border-radius:13px !important;
        background:rgba(255,255,255,.90) !important; color:#202b3e !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:12px !important; font-weight:500 !important;
        padding:0 12px !important; box-shadow:none !important;
    }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] input::placeholder,
    div.st-key-admin_auth_card [data-testid="stTextInput"] input::placeholder {{ color:#91a0b5 !important; opacity:1 !important; }}
    div.st-key-worker_auth_card [data-testid="stTextInput"] div[data-baseweb="input"],
    div.st-key-admin_auth_card [data-testid="stTextInput"] div[data-baseweb="input"] {{ border-radius:13px !important; background:transparent !important; box-shadow:none !important; }}
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button {{
        width:100% !important; height:40px !important; min-height:40px !important; margin-top:5px !important;
        border:0 !important; border-radius:12px !important;
        background:linear-gradient(135deg,#0e9ed3,#12b5c1) !important; color:#fff !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:13px !important; font-weight:700 !important;
        box-shadow:0 12px 22px rgba(14,158,211,.22) !important;
    }}
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button p,
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button span,
    div.st-key-worker_auth_card [data-testid="stFormSubmitButton"] button div,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button p,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button span,
    div.st-key-admin_auth_card [data-testid="stFormSubmitButton"] button div {{
        color:#fff !important; font-family:'Poppins',Arial,sans-serif !important;
        font-size:13px !important; font-weight:700 !important;
    }}
    .restricted-auth-brand {{ text-align:center; margin:0 0 19px; }}
    .restricted-auth-logo {{
        width:30px;height:30px;margin:0 auto 6px;border-radius:50%;display:grid;place-items:center;background:#087f9b;
        box-shadow:0 7px 18px rgba(8,127,155,.20);
    }}
    .restricted-auth-logo svg {{ width:17px;height:17px; }}
    .restricted-auth-brand-name {{ color:#10182b;font-size:18px;line-height:1.1;font-weight:800;letter-spacing:-.035em; }}
    .restricted-auth-kicker {{ margin:0 0 6px;text-align:left;color:#087f9b;font-size:11px;line-height:1.1;font-weight:800;letter-spacing:.14em;text-transform:uppercase; }}
    .restricted-auth-title {{ margin:0 0 7px;text-align:left;color:#10182b;font-size:27px;line-height:1.12;font-weight:800;letter-spacing:-.045em; }}
    .restricted-auth-copy {{ margin:0 0 18px;text-align:left;color:#536276;font-size:12px;line-height:1.5; }}
    @media(max-width:700px) {{
        body:has(.restricted-auth-page-marker) .main .block-container {{ min-height:100svh !important; }}
        div.st-key-worker_auth_card, div.st-key-admin_auth_card {{ width:min(520px,calc(100vw - 20px)) !important; padding:26px 22px 23px !important; border-radius:23px !important; }}
        div.st-key-worker_auth_back_public, div.st-key-admin_auth_back_public {{ top:14px !important; left:14px !important; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="admin_auth_card"):
        st.markdown(f"""
            <div class="restricted-auth-brand">
              <div class="restricted-auth-logo"><svg viewBox="0 0 32 32" fill="none"><path d="M16 3.8C16 3.8 8.4 13 8.4 18.2C8.4 23.05 11.82 27.1 16 27.1C20.18 27.1 23.6 23.05 23.6 18.2C23.6 13 16 3.8 16 3.8Z" stroke="white" stroke-width="2.2"/></svg></div>
              <div class="restricted-auth-brand-name">AquaTrack</div>
            </div>
            <div class="restricted-auth-kicker">{t("Restricted Access")}</div>
            <div class="restricted-auth-title">{t("Admin Login")}</div>
            <div class="restricted-auth-copy">{t("Sign in with the AquaTrack administrator account to review reports and manage project data.")}</div>
        """, unsafe_allow_html=True)
        with st.form("admin_login_form"):
            st.text_input(t("Admin Username"), key="admin_username", value=st.session_state.get("admin_remember_username", ""), placeholder=t("Admin Username"), icon=":material/mail:")
            st.text_input(t("Admin Password"), type="password", key="admin_password", placeholder=t("Admin Password"), icon=":material/lock:")
            option_cols = st.columns([1, 1], gap="small")
            with option_cols[0]:
                admin_remember = st.checkbox("Remember Me", value=False, key="admin_remember_me")
            with option_cols[1]:
                st.empty()
            st.form_submit_button(t("Login →"), width="stretch", type="primary", on_click=attempt_admin_login)
            if admin_remember:
                st.session_state.admin_remember_username = st.session_state.get("admin_username", "")
            else:
                st.session_state.pop("admin_remember_username", None)
            if st.session_state.admin_login_error:
                st.toast(t(st.session_state.admin_login_error), icon="❌")

    with st.container(key="admin_auth_back_public"):
        st.button(t("← Back to Home"), key="admin_back_public_new", width="content", type="secondary", on_click=navigate_to, args=("Dashboard",))


# ============================================================
# CITIZEN ACCOUNT + REPORTING
# ============================================================

elif page in ("Citizen Login", "Citizen Register", "Citizen Forgot Password", "Citizen Reset Password"):
    is_register = page == "Citizen Register"
    is_forgot = page == "Citizen Forgot Password"
    is_reset = page == "Citizen Reset Password"
    st.markdown('<div class="citizen-auth-page-marker" aria-hidden="true"></div>', unsafe_allow_html=True)

    # Reference-style AquaTrack authentication: one centered glass card,
    # Poppins typography, compact controls, and the same visual hierarchy
    # as the supplied login/signup references.
    import base64
    # Use the supplied AquaTrack water/lake image from the project assets.
    # Keep a few compatible fallbacks so the auth page still works if the asset
    # is temporarily placed beside app.py during local setup.
    _auth_bg = ""
    _auth_bg_mime = "image/png"
    for _candidate in (
        BASE_DIR / "assets" / "aqua_auth_background.jpg",
        BASE_DIR / "assets" / "aqua_auth_background.png",
        BASE_DIR / "aqua_auth_background.png",
        BASE_DIR / "assets" / "aqua_auth_bg.jpg",
        BASE_DIR / "aqua_auth_bg.jpg",
    ):
        try:
            if _candidate.exists():
                _auth_bg = base64.b64encode(_candidate.read_bytes()).decode("ascii")
                _auth_bg_mime = "image/jpeg" if _candidate.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                break
        except Exception:
            continue
    _bg_css = f"url(data:{_auth_bg_mime};base64,{_auth_bg})" if _auth_bg else "linear-gradient(135deg,#f2e8da 0%,#f2eef3 52%,#ece7f4 100%)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html:has(.citizen-auth-page-marker),
    body:has(.citizen-auth-page-marker) {{
        background:{_bg_css} center/cover fixed no-repeat !important;
    }}
    body:has(.citizen-auth-page-marker) .stApp,
    body:has(.citizen-auth-page-marker) [data-testid="stAppViewContainer"],
    body:has(.citizen-auth-page-marker) section[data-testid="stMain"] {{
        background:transparent !important;
    }}
    body:has(.citizen-auth-page-marker) .main .block-container {{
        width:100% !important; max-width:none !important;
        min-height:100vh !important;
        padding:0 !important;
        display:flex !important; align-items:center !important; justify-content:center !important;
    }}
    body:has(.citizen-auth-page-marker) .aqua-footer,
    body:has(.citizen-auth-page-marker) .aqua-subnav-gap,
    body:has(.citizen-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-desktop-header-marker),
    body:has(.citizen-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-header-marker),
    body:has(.citizen-auth-page-marker) div[data-testid="stHorizontalBlock"]:has(.aqua-mobile-menu-marker),
    body:has(.citizen-auth-page-marker) [data-testid="stToolbar"] {{ display:none !important; }}
    /* Compact Back to Home control matching the supplied reference.
       Navigation stays inside Streamlit via the button callback, so it never
       opens a second website/tab. */
    div.st-key-citizen_auth_back_home {{
        position:fixed !important; top:20px !important; left:24px !important; z-index:99999 !important;
        width:auto !important; margin:0 !important; padding:0 !important;
    }}
    div.st-key-citizen_auth_back_home button {{
        min-width:150px !important; width:150px !important; height:40px !important;
        min-height:40px !important; padding:0 14px !important;
        border:0 !important; border-radius:22px !important;
        background:rgba(255,255,255,.94) !important; color:#243247 !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:13px !important; font-weight:700 !important;
        box-shadow:0 8px 22px rgba(24,35,55,.14) !important;
    }}
    div.st-key-citizen_auth_back_home button p,
    div.st-key-citizen_auth_back_home button span,
    div.st-key-citizen_auth_back_home button div {{
        color:#243247 !important; font-family:'Poppins',Arial,sans-serif !important;
        font-size:13px !important; font-weight:700 !important;
    }}
    div.st-key-citizen_auth_back_home button:hover {{ background:#fff !important; color:#087f9b !important; }}
    div.st-key-citizen_auth_back_home button:hover p,
    div.st-key-citizen_auth_back_home button:hover span,
    div.st-key-citizen_auth_back_home button:hover div {{ color:#087f9b !important; }}

    div.st-key-citizen_auth_card {{
        width:min(520px,calc(100vw - 32px)) !important;
        position:relative !important;
        left:5px !important;
        max-width:520px !important;
        margin:0 auto !important;
        padding:30px 30px 25px !important;
        box-sizing:border-box !important;
        border:1px solid rgba(255,255,255,.96) !important;
        border-radius:25px !important;
        background:rgba(255,255,255,.72) !important;
        box-shadow:0 22px 65px rgba(24,35,55,.20) !important;
        backdrop-filter:blur(18px) saturate(125%) !important;
        -webkit-backdrop-filter:blur(18px) saturate(125%) !important;
        font-family:'Poppins',Arial,sans-serif !important;
    }}
    div.st-key-citizen_auth_card > div {{ gap:.28rem !important; }}
    div.st-key-citizen_auth_card [data-testid="stForm"] {{
        border:0 !important; padding:0 !important; background:transparent !important; box-shadow:none !important;
    }}
    div.st-key-citizen_auth_card [data-testid="stTextInput"] {{ margin:0 0 10px !important; }}
    div.st-key-citizen_auth_card [data-testid="stTextInput"] label {{ display:none !important; }}
    div.st-key-citizen_auth_card [data-testid="stTextInput"] input {{
        height:38px !important; min-height:38px !important; box-sizing:border-box !important;
        border:1px solid #dbe1ea !important; border-radius:13px !important;
        background:rgba(255,255,255,.90) !important; color:#202b3e !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:12px !important; font-weight:500 !important;
        padding:0 12px !important; box-shadow:none !important;
    }}
    div.st-key-citizen_auth_card [data-testid="stTextInput"] input::placeholder {{ color:#91a0b5 !important; opacity:1 !important; }}
    div.st-key-citizen_auth_card [data-testid="stTextInput"] div[data-baseweb="input"] {{ border-radius:13px !important; background:transparent !important; box-shadow:none !important; }}
    div.st-key-citizen_auth_card [data-testid="stCheckbox"] {{ margin:0 !important; padding:0 !important; }}
    div.st-key-citizen_auth_card [data-testid="stCheckbox"] label {{ font-family:'Poppins',Arial,sans-serif !important; font-size:12.5px !important; color:#34445a !important; font-weight:500 !important; }}
    div.st-key-citizen_auth_card [data-testid="stCheckbox"] input {{ accent-color:#0878ff !important; }}
    div.st-key-citizen_auth_card [data-testid="stFormSubmitButton"] button {{
        width:100% !important; height:40px !important; min-height:40px !important; margin-top:5px !important;
        border:0 !important; border-radius:12px !important;
        background:linear-gradient(135deg,#0e9ed3,#12b5c1) !important; color:#fff !important;
        font-family:'Poppins',Arial,sans-serif !important; font-size:13px !important; font-weight:700 !important;
        box-shadow:0 12px 22px rgba(14,158,211,.22) !important;
    }}
        div.st-key-citizen_auth_card [data-testid="stFormSubmitButton"] button p,
    div.st-key-citizen_auth_card [data-testid="stFormSubmitButton"] button span,
    div.st-key-citizen_auth_card [data-testid="stFormSubmitButton"] button div {{
        color:#fff !important;
        font-family:'Poppins',Arial,sans-serif !important;
        font-size:13px !important;
        font-weight:700 !important;
    }}
div.st-key-citizen_auth_card .citizen-auth-brand {{ text-align:center; margin:0 0 19px; }}
    div.st-key-citizen_auth_card .citizen-auth-logo {{
        width:30px;height:30px;margin:0 auto 6px;border-radius:50%;display:grid;place-items:center;background:#087f9b;
        box-shadow:0 7px 18px rgba(8,127,155,.20);
    }}
    div.st-key-citizen_auth_card .citizen-auth-logo svg {{ width:17px;height:17px; }}
    div.st-key-citizen_auth_card .citizen-auth-brand-name {{ color:#10182b;font-size:18px;line-height:1.1;font-weight:800;letter-spacing:-.035em; }}
    div.st-key-citizen_auth_card .citizen-auth-brand-sub {{ margin-top:4px;color:#536276;font-size:10.5px;font-weight:400; }}
    div.st-key-citizen_auth_card .citizen-auth-title {{
        margin:0 0 4px;text-align:center;color:#10182b;font-size:27px;line-height:1.12;font-weight:800;letter-spacing:-.045em;
    }}
    div.st-key-citizen_auth_card .citizen-auth-title-accent {{ color:#087f9b; }}
    div.st-key-citizen_auth_card .citizen-auth-copy {{
        margin:0 auto 18px;max-width:340px;text-align:center;color:#536276;font-size:12px;line-height:1.5;
    }}
    div.st-key-citizen_auth_card .citizen-auth-forgot {{ text-align:right;color:#087f9b;font-size:12px;font-weight:600;padding-top:3px; }}
    div.st-key-citizen_auth_card .citizen-auth-forgot a {{ color:#087f9b !important;text-decoration:none !important;font-weight:600 !important;cursor:pointer; }}
    div.st-key-citizen_auth_card .citizen-auth-forgot a:hover {{ text-decoration:underline !important; }}
    div.st-key-citizen_auth_card .citizen-auth-divider {{
        display:flex;align-items:center;gap:12px;margin:17px 0 16px;color:#66758a;font-size:13px;font-weight:700;
        text-transform:uppercase;letter-spacing:0;font-family:'Poppins',Arial,sans-serif;
    }}
    div.st-key-citizen_auth_card .citizen-auth-divider:before,
    div.st-key-citizen_auth_card .citizen-auth-divider:after {{ content:"";height:1px;background:#d8dce4;flex:1; }}
    div.st-key-citizen_auth_card .citizen-auth-social-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:0 0 14px; }}
    div.st-key-citizen_auth_card div[data-testid="column"] :has(.citizen-social) {{ margin-bottom:0 !important; }}
    div.st-key-citizen_auth_card .citizen-social button {{
        width:100% !important;height:31px !important;min-height:31px !important;padding:0 6px !important;
        border:1px solid #dce2ea !important;border-radius:10px !important;background:rgba(255,255,255,.92) !important;
        color:#263449 !important;font-family:'Poppins',Arial,sans-serif !important;font-size:8px !important;font-weight:700 !important;
        box-shadow:none !important;
    }}
    div.st-key-citizen_auth_card .citizen-social a {{
        display:flex;align-items:center;justify-content:center;width:100%;height:31px;box-sizing:border-box;
        border:1px solid #dce2ea;border-radius:10px;background:rgba(255,255,255,.92);color:#263449;
        text-decoration:none;font-family:'Poppins',Arial,sans-serif;font-size:11px;font-weight:650;
        line-height:1;white-space:nowrap;gap:6px;
    }}
    div.st-key-citizen_auth_card .citizen-social a span {{
        font-size:11px !important;line-height:1 !important;white-space:nowrap !important;
    }}
    div.st-key-citizen_auth_card .citizen-social a:hover {{ background:#fff; color:#1f2c42; }}
    div.st-key-citizen_auth_card .social-brand-icon {{ width:18px; height:18px; flex:0 0 18px; display:block; }}
    div.st-key-citizen_auth_card .citizen-social a, div.st-key-citizen_auth_card .citizen-social-disabled {{ display:flex; align-items:center; justify-content:center; gap:10px; }}
    div.st-key-citizen_auth_card .citizen-social-disabled {{ width:100%; min-height:46px; border:1px solid #dfe6ef; border-radius:14px; background:#fff; color:#1f2c42; font-weight:700; opacity:.72; box-sizing:border-box; }}
    div.st-key-citizen_auth_card .citizen-auth-switch {{ margin:30px 0 0 !important;text-align:center;color:#607086;font-size:13px;font-weight:400; }}
    div.st-key-citizen_auth_card .citizen-auth-switch strong {{ color:#087f9b;font-weight:700; }}
    div.st-key-citizen_auth_card .citizen-auth-switch a {{ color:#087f9b !important;text-decoration:none !important;cursor:pointer; }}
    div.st-key-citizen_auth_card .citizen-auth-switch a:hover {{ text-decoration:underline !important; }}
    div.st-key-citizen_auth_card .citizen-switch-button button {{
        border:0 !important;background:transparent !important;box-shadow:none !important;color:#087f9b !important;
        font-size:13px !important;font-weight:700 !important;min-height:0 !important;height:auto !important;padding:0 !important;margin-top:-5px !important;
    }}
    div.st-key-citizen_auth_card [data-testid="stAlert"] {{ font-size:12px !important; }}
    div.st-key-citizen_auth_card [data-testid="stFormSubmitButton"] button p {{ font-size:13px !important; font-weight:700 !important; }}
    div.st-key-citizen_auth_card .citizen-auth-switch {{ line-height:1.55 !important; }}
    div.st-key-citizen_auth_card .citizen-auth-switch a,
    div.st-key-citizen_auth_card .citizen-auth-switch strong {{ font-size:13px !important; }}
    div.st-key-citizen_auth_card .citizen-auth-title-accent {{ color:#087f9b !important; }}
    div.st-key-citizen_auth_card .citizen-auth-note {{ display:none !important; }}
    @media(max-width:700px) {{
        body:has(.citizen-auth-page-marker) .main .block-container {{ min-height:100svh !important; }}
        div.st-key-citizen_auth_card {{ width:min(520px,calc(100vw - 20px)) !important; padding:26px 22px 23px !important; border-radius:23px !important; }}
        div.st-key-citizen_auth_back_home {{ top:14px !important; left:14px !important; }}
        div.st-key-citizen_auth_back_home button {{ min-width:150px !important; width:150px !important; height:40px !important; min-height:40px !important; padding:0 14px !important; font-size:13px !important; }}
        div.st-key-citizen_auth_back_home button p, div.st-key-citizen_auth_back_home button span, div.st-key-citizen_auth_back_home button div {{ font-size:13px !important; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # Reference-style Back to Home control shown consistently on every citizen auth page.
    with st.container(key="citizen_auth_back_home"):
        st.button("←  Back to Home", key=f"citizen_auth_back_home_btn_{page}", width="content", type="secondary", on_click=navigate_to, args=("Dashboard",))

    if is_register:
        title_main, title_accent = "Create Your", "Account"
        subtitle = "Start your personalized water-service journey towards a better water experience."
    elif is_forgot:
        title_main, title_accent = "Reset", "Password"
        subtitle = "Enter your registered email address to receive password recovery instructions."
    elif is_reset:
        title_main, title_accent = "Create New", "Password"
        subtitle = "Choose a new password for your AquaTrack citizen account."
    else:
        title_main, title_accent = "Welcome", "Back"
        subtitle = "Continue managing your water reports and service updates."

    with st.container(key="citizen_auth_card"):
        _auth_icon = (
            '<svg viewBox="0 0 32 32" fill="none"><path d="M19.8 5.5a6.1 6.1 0 1 1-5.1 9.4L7 22.6V27h4.4v-3.1h3.1v-3.1h3.1l2.5-2.5a6.1 6.1 0 0 1-.3-12.8Zm0 3.2a2.9 2.9 0 1 0 0 5.8 2.9 2.9 0 0 0 0-5.8Z" stroke="white" stroke-width="1.8" stroke-linejoin="round"/></svg>'
            if (is_forgot or is_reset) else
            '<svg viewBox="0 0 32 32" fill="none"><path d="M16 3.8C16 3.8 8.4 13 8.4 18.2C8.4 23.05 11.82 27.1 16 27.1C20.18 27.1 23.6 23.05 23.6 18.2C23.6 13 16 3.8 16 3.8Z" stroke="white" stroke-width="2.2"/></svg>'
        )
        st.markdown(f"""
            <div class="citizen-auth-brand">
              <div class="citizen-auth-logo">{_auth_icon}</div>
              <div class="citizen-auth-brand-name">AquaTrack</div>
              <div class="citizen-auth-brand-sub">Predictive Water Intelligence</div>
            </div>
            <div class="citizen-auth-title">{title_main} <span class="citizen-auth-title-accent">{title_accent}</span></div>
            <div class="citizen-auth-copy">{subtitle}</div>
        """, unsafe_allow_html=True)

        if is_forgot:
            with st.form("citizen_forgot_form"):
                email=st.text_input("Email Address", placeholder="student@college.edu", icon=":material/mail:")
                submitted=st.form_submit_button("Send Reset Link  →", type="primary", width="stretch")
            if submitted:
                ok, message = create_citizen_password_reset(email)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
            st.markdown('<div class="citizen-auth-switch">Remembered your password? <a href="?auth=login" target="_self"><strong>Back to Sign In</strong></a></div>', unsafe_allow_html=True)

        elif is_reset:
            reset_token = str(st.query_params.get("reset_token", "")).strip()
            reset_record = get_password_reset(reset_token)
            if not reset_record:
                st.error("This password reset link is invalid or has expired.")
                st.markdown('<div class="citizen-auth-switch"><a href="?auth=login" target="_self"><strong>Back to Sign In</strong></a></div>', unsafe_allow_html=True)
            else:
                with st.form("citizen_reset_form"):
                    new_password=st.text_input("New Password", type="password", placeholder="New Password", icon=":material/lock:")
                    confirm_password=st.text_input("Confirm Password", type="password", placeholder="Confirm Password", icon=":material/lock:")
                    submitted=st.form_submit_button("Update Password  →", type="primary", width="stretch")
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        ok, message = complete_citizen_password_reset(reset_token, new_password)
                        if ok:
                            try: st.query_params.clear()
                            except Exception: pass
                            st.session_state.page="Citizen Login"
                            st.session_state.citizen_login_error=None
                            st.success(message)
                        else:
                            st.error(message)
                st.markdown('<div class="citizen-auth-switch"><a href="?auth=login" target="_self"><strong>Back to Sign In</strong></a></div>', unsafe_allow_html=True)

        elif is_register:
            zones=get_zones()
            with st.form("citizen_register_form"):
                email=st.text_input("Email Address", placeholder="Email", icon=":material/mail:")
                password=st.text_input("Password", type="password", placeholder="Password", icon=":material/lock:")
                confirm=st.text_input("Confirm Password", type="password", placeholder="Confirm Password", icon=":material/lock:")
                agreed=st.checkbox("I agree to the Terms of Service and Privacy Policy", value=False)
                submitted=st.form_submit_button("Create Account  →", type="primary", width="stretch")
            if submitted:
                if not agreed: st.error("Please accept the Terms of Service and Privacy Policy.")
                elif not email or "@" not in email: st.error("Please enter a valid email address.")
                elif password != confirm: st.error("Passwords do not match.")
                elif zones.empty: st.error("No project-defined zones are available.")
                else:
                    zone_id=int(zones.iloc[0]["zone_id"])
                    base=email.split("@")[0].strip().lower() or "citizen"
                    username=base; suffix=1
                    while not db_query("SELECT resident_id FROM residents WHERE username=? LIMIT 1",(username,)).empty:
                        username=f"{base}{suffix}"; suffix+=1
                    ok,result=register_citizen_account(username,password,email.split("@")[0].strip() or "Citizen","",email,zone_id)
                    if ok:
                        resident=get_resident(int(result)).iloc[0].to_dict()
                        st.session_state.citizen_logged_in=True
                        st.session_state.citizen_user=resident
                        st.session_state.resident_id=int(result)
                        st.session_state.resident_zone_id=zone_id
                        st.session_state.page=st.session_state.pop("citizen_after_login", "Dashboard")
                        st.rerun()
                    else: st.error(t(result))
        else:
            with st.form("citizen_login_form"):
                email=st.text_input("Email Address", key="citizen_email", value=st.session_state.get("citizen_remember_email", ""), placeholder="Email Address", icon=":material/mail:")
                password=st.text_input("Password", type="password", key="citizen_password", placeholder="Password", icon=":material/lock:")
                option_cols=st.columns([1,1], gap="small")
                with option_cols[0]:
                    remember=st.checkbox("Remember Me", value=False)
                with option_cols[1]:
                    st.markdown('<div class="citizen-auth-forgot"><a href="?auth=forgot" target="_self">Forgot Password?</a></div>', unsafe_allow_html=True)
                submitted=st.form_submit_button("Login  →", type="primary", width="stretch")
            if submitted:
                if remember:
                    st.session_state.citizen_remember_email=st.session_state.get("citizen_email", "")
                attempt_citizen_login(); st.rerun()
            if st.session_state.citizen_login_error: st.error(t(st.session_state.citizen_login_error))

        if not (is_forgot or is_reset):
            st.markdown('<div class="citizen-auth-divider"><span>OR CONTINUE WITH</span></div>', unsafe_allow_html=True)
            google_url = _oauth_authorize_url("google")
            social_cols=st.columns([1, 2, 1], gap="small")
            google_icon = """<svg class=\"social-brand-icon google-brand-icon\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"#4285F4\" d=\"M21.35 12.27c0-.71-.06-1.4-.18-2.05H12v3.88h5.24a4.48 4.48 0 0 1-1.94 2.94v2.44h3.14c1.84-1.7 2.91-4.2 2.91-7.21Z\"/><path fill=\"#34A853\" d=\"M12 21.5c2.63 0 4.84-.87 6.45-2.36l-3.14-2.44c-.87.58-1.98.92-3.31.92-2.54 0-4.69-1.72-5.46-4.03H3.29v2.52A9.74 9.74 0 0 0 12 21.5Z\"/><path fill=\"#FBBC05\" d=\"M6.54 13.59A5.85 5.85 0 0 1 6.23 12c0-.55.1-1.08.31-1.59V7.89H3.29A9.75 9.75 0 0 0 2.25 12c0 1.57.38 3.06 1.04 4.41l3.25-2.52Z\"/><path fill=\"#EA4335\" d=\"M12 6.38c1.43 0 2.72.49 3.73 1.46l2.8-2.8C16.84 3.49 14.63 2.5 12 2.5a9.74 9.74 0 0 0-8.71 5.39l3.25 2.52C7.31 8.1 9.46 6.38 12 6.38Z\"/></svg>"""
            with social_cols[1]:
                if google_url:
                    st.markdown(f'<div class="citizen-social"><a href="{google_url}" target="_self">{google_icon}<span>Continue with Google</span></a></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="citizen-social citizen-social-disabled" title="Configure Google OAuth in .streamlit/secrets.toml">{google_icon}<span>Continue with Google</span></div>', unsafe_allow_html=True)
        if is_register:
            st.markdown(
                '<div class="citizen-auth-switch">Already have an account? '
                '<a href="?auth=login" target="_self"><strong>Login</strong></a></div>',
                unsafe_allow_html=True,
            )
        elif not (is_forgot or is_reset):
            # The highlighted Sign Up text is the only control.
            # No extra full-width Sign Up button is rendered.
            st.markdown(
                '<div class="citizen-auth-switch">Don\'t have an account? '
                '<a href="?auth=signup" target="_self"><strong>Sign Up</strong></a></div>',
                unsafe_allow_html=True,
            )

elif page == "Citizen Profile":
    if not st.session_state.citizen_logged_in:
        open_citizen_login(); st.rerun()
    resident=get_resident(int(st.session_state.resident_id))
    if resident.empty: st.error(t("Account could not be loaded."))
    else:
        row=resident.iloc[0]
        st.title(t("My Account")); st.caption(t("Manage your profile and account preferences."))
        c1,c2=st.columns([2,1])
        with c1:
            with st.form("citizen_profile_form"):
                name=st.text_input(t("Full name"),value=str(row.get("name") or ""))
                phone=st.text_input(t("Phone number"),value=str(row.get("phone") or ""))
                email=st.text_input(t("Email"),value=str(row.get("email") or ""))
                zones=get_zones(); names=zones["zone_name"].tolist(); matches=zones.loc[zones["zone_id"]==int(row["zone_id"]),"zone_name"]
                current=matches.iloc[0] if not matches.empty else names[0]
                zone_name=st.selectbox(t("Your area"),names,index=names.index(current) if current in names else 0)
                if st.form_submit_button(t("Save Profile"),type="primary"):
                    zone_id=int(zones.loc[zones["zone_name"]==zone_name,"zone_id"].iloc[0]); update_citizen_profile(int(row["resident_id"]),name,phone,email,zone_id); st.session_state.citizen_user=get_resident(int(row["resident_id"])).iloc[0].to_dict(); st.toast(t("Profile updated."),icon="✅"); st.rerun()
        with c2:
            st.markdown('<div class="account-summary-card">',unsafe_allow_html=True)
            st.markdown(f"<div class='account-summary-label'>{t('USERNAME')}</div><div class='account-summary-value'>{row.get('username') or '—'}</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='account-summary-label'>{t('REPORTS')}</div><div class='account-summary-value'>{len(get_service_requests(resident_id=int(row['resident_id'])))}</div>",unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
            st.button(t("Log Out"),key="citizen_logout_profile",width="stretch",on_click=do_citizen_logout)

elif page == "Report Water Body Issue":
    if not st.session_state.citizen_logged_in:
        require_citizen_login("Report Water Body Issue")
        st.rerun()
    else:
        # Use a new widget namespace after a successful submit so Streamlit cannot
        # restore the previous form values from the old widget state.
        if "citizen_report_form_version" not in st.session_state:
            st.session_state["citizen_report_form_version"] = 0
        if st.session_state.pop("citizen_report_reset_form", False):
            st.session_state["citizen_report_form_version"] += 1
            st.session_state["citizen_report_latitude"] = None
            st.session_state["citizen_report_longitude"] = None
            st.session_state["citizen_report_location_source"] = None
            st.session_state["citizen_report_location_accuracy"] = None
            st.session_state["citizen_report_last_geo_pair"] = None
            st.session_state["citizen_report_last_map_click"] = None

        form_version = int(st.session_state["citizen_report_form_version"])
        issue_type_key = f"citizen_report_issue_type_{form_version}"
        area_key = f"citizen_report_area_{form_version}"
        description_key = f"citizen_report_description_{form_version}"
        photo_key = f"citizen_report_photo_{form_version}"
        society_key = f"citizen_report_society_{form_version}"
        location_picker_key = f"citizen_report_location_picker_{form_version}"

        completed_report_id = st.session_state.pop("citizen_report_success_id", None)
        if completed_report_id:
            st.toast(f"{t('Report submitted successfully. Your report number is')} #{int(completed_report_id)}.")
            st.markdown('<div id="citizen-report-success-message"></div>', unsafe_allow_html=True)
            st.success(f"{t('Report submitted successfully. Your report number is')} **#{int(completed_report_id)}**.")
            st.info(t("You will receive status notifications as the report moves through the workflow."))
            st.markdown("""
            <script>
            setTimeout(function() {
                var target = window.parent.document.getElementById('citizen-report-success-message');
                if (target) { target.scrollIntoView({behavior:'smooth', block:'start'}); }
            }, 100);
            </script>
            """, unsafe_allow_html=True)

        resident = get_resident(int(st.session_state.resident_id))
        row = resident.iloc[0]
        st.title(t("Report a Water Problem"))
        st.caption(t("Submit a water-related issue and track its progress from your account."))

        zones = get_zones()
        zone_names = zones["zone_name"].tolist()
        matches = zones.loc[zones["zone_id"] == int(row["zone_id"]), "zone_name"]
        current_zone = matches.iloc[0] if not matches.empty else zone_names[0]

        # Keep all form input values in session state because selecting GPS/map
        # coordinates triggers Streamlit reruns.
        st.selectbox(
            t("Issue type"),
            ["Low Water Supply", "Water Quality", "Pipeline / Leakage", "Contamination", "Other"],
            key=issue_type_key,
            format_func=t,
        )
        st.selectbox(
            t("Area"), zone_names,
            index=zone_names.index(current_zone) if current_zone in zone_names else 0,
            key=area_key,
            format_func=t,
        )
        st.text_area(
            t("Describe the problem"),
            placeholder=t("Tell us what is happening, where you noticed it and when it started."),
            height=130,
            key=description_key,
        )
        st.file_uploader(
            t("Photo (required)"),
            type=["jpg", "jpeg", "png", "webp"],
            key=photo_key,
        )
        st.text_input(t("Society / landmark (optional)"), key=society_key)

        st.markdown(f"### {t('Report Location')}")

        for key_name in (
            "citizen_report_latitude",
            "citizen_report_longitude",
            "citizen_report_location_source",
            "citizen_report_location_accuracy",
            "citizen_report_last_geo_pair",
            "citizen_report_last_map_click",
        ):
            st.session_state.setdefault(key_name, None)

        # Browser geolocation component displays its own action button and asks
        # for permission; it returns coordinates only after the user presses it.
        st.markdown(f"**{t('Use Current Location')}**")
        location_result = streamlit_geolocation()
        geo_pair = None
        if isinstance(location_result, dict):
            try:
                lat = float(location_result.get("latitude"))
                lon = float(location_result.get("longitude"))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    geo_pair = (round(lat, 6), round(lon, 6))
            except (TypeError, ValueError):
                geo_pair = None

        # The geolocation component may return its last result on later reruns.
        # Apply it only when it changes, so an old GPS result cannot override a
        # newer location that the user selected manually on the map.
        if geo_pair is not None and geo_pair != st.session_state.get("citizen_report_last_geo_pair"):
            st.session_state["citizen_report_last_geo_pair"] = geo_pair
            st.session_state["citizen_report_latitude"] = float(location_result["latitude"])
            st.session_state["citizen_report_longitude"] = float(location_result["longitude"])
            st.session_state["citizen_report_location_source"] = "current"
            try:
                accuracy_value = location_result.get("accuracy")
                st.session_state["citizen_report_location_accuracy"] = (
                    float(accuracy_value) if accuracy_value is not None else None
                )
            except (TypeError, ValueError):
                st.session_state["citizen_report_location_accuracy"] = None

        selected_lat = st.session_state.get("citizen_report_latitude")
        selected_lon = st.session_state.get("citizen_report_longitude")
        has_selected_location = selected_lat is not None and selected_lon is not None

        # Center the picker on the selected GPS point, or on the citizen's
        # selected project zone if no exact location has been chosen yet.
        map_center = [22.7196, 75.8577]
        if has_selected_location:
            map_center = [float(selected_lat), float(selected_lon)]
        elif {"latitude", "longitude"}.issubset(zones.columns):
            zone_row = zones.loc[zones["zone_name"] == st.session_state.get("citizen_report_area", current_zone)]
            if not zone_row.empty and pd.notna(zone_row.iloc[0]["latitude"]) and pd.notna(zone_row.iloc[0]["longitude"]):
                map_center = [float(zone_row.iloc[0]["latitude"]), float(zone_row.iloc[0]["longitude"])]

        location_map = make_base_map(map_center, 15 if has_selected_location else 12)
        if has_selected_location:
            folium.Marker(
                location=[float(selected_lat), float(selected_lon)],
                tooltip=t("Selected location"),
                popup=f"{t('Selected location')}: {float(selected_lat):.6f}, {float(selected_lon):.6f}",
                icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
            ).add_to(location_map)

        st.caption(t("Click the map to pin the exact problem location."))

        map_result = st_folium(
            location_map,
            width=None,
            height=360,
            key=location_picker_key,
            returned_objects=["last_clicked"],
        )

        clicked = map_result.get("last_clicked") if isinstance(map_result, dict) else None
        clicked_pair = None
        if isinstance(clicked, dict):
            try:
                clicked_lat = float(clicked.get("lat"))
                clicked_lon = float(clicked.get("lng"))
                if -90 <= clicked_lat <= 90 and -180 <= clicked_lon <= 180:
                    clicked_pair = (round(clicked_lat, 6), round(clicked_lon, 6))
            except (TypeError, ValueError):
                clicked_pair = None

        last_map_click = st.session_state.get("citizen_report_last_map_click")
        if clicked_pair is not None and clicked_pair != last_map_click:
            st.session_state["citizen_report_last_map_click"] = clicked_pair
            st.session_state["citizen_report_latitude"] = float(clicked["lat"])
            st.session_state["citizen_report_longitude"] = float(clicked["lng"])
            st.session_state["citizen_report_location_source"] = "map"
            st.session_state["citizen_report_location_accuracy"] = None
            st.rerun()

        # Refresh selected location values after processing all map/widget events.
        selected_lat = st.session_state.get("citizen_report_latitude")
        selected_lon = st.session_state.get("citizen_report_longitude")
        has_selected_location = selected_lat is not None and selected_lon is not None

        submission_locked = bool(st.session_state.get("citizen_report_submission_locked", False))
        if submission_locked:
            has_new_submission_input = bool(
                str(st.session_state.get(description_key, "") or "").strip()
                or st.session_state.get(photo_key) is not None
                or str(st.session_state.get(society_key, "") or "").strip()
                or has_selected_location
            )
            if has_new_submission_input:
                submission_locked = False
                st.session_state["citizen_report_submission_locked"] = False

        st.markdown('<span class="citizen-submit-report-marker"></span>', unsafe_allow_html=True)
        submitted = st.button(t("Submit Report"), key="citizen_submit_report", type="primary", width="stretch", disabled=submission_locked)
        if submitted:
            issue = st.session_state.get(issue_type_key, "Other")
            severity = "Medium"  # Initial triage value; Admin can review and change it.
            zone_name = st.session_state.get("citizen_report_area", current_zone)
            description = str(st.session_state.get(description_key, "") or "")
            photo = st.session_state.get(photo_key)
            society = st.session_state.get(society_key, "")

            if len(description.strip()) < 10:
                st.error(t("Please provide a little more detail about the problem."))
            elif photo is None:
                st.error(t("A photo is required to submit a report."))
            elif photo.size > 5 * 1024 * 1024:
                st.error(t("Photo must be 5 MB or smaller."))
            else:
                zone_id = int(zones.loc[zones["zone_name"] == zone_name, "zone_id"].iloc[0])
                photo_path = None
                folder = BASE_DIR / "reports" / "citizen_uploads"
                folder.mkdir(parents=True, exist_ok=True)
                try:
                    img = Image.open(photo)
                    img.verify()
                    photo.seek(0)
                except Exception:
                    st.error(t("Uploaded file is not a valid image."))
                    st.stop()
                ext = Path(photo.name).suffix.lower()
                if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                    ext = ".jpg"
                saved_name = f"report_{secrets.token_hex(16)}{ext}"
                saved = folder / saved_name
                saved.write_bytes(photo.getbuffer())
                photo_path = str(saved.relative_to(BASE_DIR))

                report_lat = st.session_state.get("citizen_report_latitude")
                report_lon = st.session_state.get("citizen_report_longitude")
                rid = create_citizen_report(
                    int(row["resident_id"]),
                    zone_id,
                    issue,
                    description,
                    severity,
                    None,
                    "",
                    society,
                    None if report_lat is None else float(report_lat),
                    None if report_lon is None else float(report_lon),
                    photo_path,
                )

                st.session_state["citizen_report_success_id"] = int(rid)
                st.session_state["citizen_report_submission_locked"] = True
                st.session_state["citizen_report_reset_form"] = True
                st.session_state["citizen_report_latitude"] = None
                st.session_state["citizen_report_longitude"] = None
                st.session_state["citizen_report_location_source"] = None
                st.session_state["citizen_report_location_accuracy"] = None
                st.rerun()

# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":
    if st.session_state.citizen_logged_in:
        citizen=st.session_state.get("citizen_user") or {}; rid=int(st.session_state.resident_id); requests=get_service_requests(resident_id=rid)
        active=int((requests["workflow_status"]!="Resolved").sum()) if not requests.empty else 0; resolved=int((requests["workflow_status"]=="Resolved").sum()) if not requests.empty else 0; unread=get_unread_count("resident",rid)
        c1,c2,c3,c4=st.columns(4); c1.metric(t("My Reports"),len(requests)); c2.metric(t("Active"),active); c3.metric(t("Resolved"),resolved); c4.metric(t("Notifications"),unread)

        st.markdown(f"""
        <div class="citizen-info-grid">
            <div class="citizen-info-card">
                <div class="citizen-info-title">{t("Water Quality Status")}</div>
                <div class="citizen-info-text">{t("Check your area water quality and risk status below.")}</div>
            </div>
            <div class="citizen-info-card">
                <div class="citizen-info-title">{t("Water Supply Information")}</div>
                <div class="citizen-info-text">{t("View your area's current supply information and availability below.")}</div>
            </div>
            <div class="citizen-info-card">
                <div class="citizen-info-title">{t("Important Notices")}</div>
                <div class="citizen-info-text">{t("Stay informed about important water-related updates in your area.")}</div>
            </div>
            <div class="citizen-info-card">
                <div class="citizen-info-title">{t("How AquaTrack Works")}</div>
                <div class="citizen-info-text">{t("Report a problem, track its review, and follow the progress until it is resolved.")}</div>
            </div>
            <div class="citizen-info-card">
                <div class="citizen-info-title">{t("Water Conservation Tips")}</div>
                <div class="citizen-info-text">{t("Use water carefully, fix leaks promptly, and avoid unnecessary water wastage.")}</div>
            </div>
        </div>
        """,unsafe_allow_html=True)

        st.markdown(f"<div class='citizen-section-title'>{t('Area Water Status')}</div>",unsafe_allow_html=True)
        snap=get_zone_snapshot(); zone_name=st.session_state.get("resident_zone_name")
        if not zone_name:
            zones=get_zones(); matches=zones.loc[zones["zone_id"]==int(citizen.get("zone_id",0)),"zone_name"]; zone_name=matches.iloc[0] if not matches.empty else None
        if zone_name and not snap[snap["zone_name"]==zone_name].empty:
            z=snap[snap["zone_name"]==zone_name].iloc[0]; a,b,c=st.columns(3); a.metric(t("Water Quality"),t(str(z["risk"]))); b.metric(t("Supply"),"-" if pd.isna(z["supply_hours"]) else f"{float(z['supply_hours']):.1f} h"); c.metric(t("Availability"),t(str(z["availability_status"])))
        else: st.info(t("Your area status will appear here after your account is set up."))

        st.markdown(
            """
            <div class="aqua-footer">
                <b>AquaTrack</b> · Predictive Water Quality &amp; Availability Analytics
                <br>
                <span>Project-level decision-support prototype · Indore · For screening and decision support</span>
                <br><span>Privacy · Terms · Data &amp; Model Transparency · Support</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-badge">● {t("Predictive Water Intelligence")}</div>
            <div class="hero-title">{t("Protect Your Water Sources")}</div>
            <div class="hero-text">
                {t("Analyze water quality, forecast availability and understand zone-wise conditions through one integrated decision-support platform for Indore.")}
            </div>
            <div class="hero-about">
                <div class="hero-about-title">{t("About AquaTrack")}</div>
                <p class="hero-about-text">{t("AquaTrack brings together zone-wise water-quality screening, water-supply forecasts, citizen issue reports, and alerts in one place. It helps users identify possible risks and follow service requests across Indore.")}</p>
                <div class="hero-capabilities">
                    <span>{t("Water Quality Screening")}</span>
                    <span>{t("Availability Forecasts")}</span>
                    <span>{t("Citizen Reports & Alerts")}</span>
                </div>
                <div class="hero-note">{t("Project prototype using project-generated data; predictions are for screening only and do not replace laboratory testing or official municipal systems.")}</div>
            </div>
            <div class="hero-mini">💧</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    counts = get_table_counts()
    snapshot = get_zone_snapshot()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("PROJECT ZONES"), counts["zones"])
    col2.metric(t("QUALITY RECORDS"), f"{counts['water_quality']:,}")
    col3.metric(t("AVAILABILITY RECORDS"), f"{counts['availability']:,}")
    col4.metric(t("PREDICTIONS"), counts["predictions"])

    section("Latest Zone Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("SAFE ZONES"), int((snapshot["risk"] == "Safe").sum()))
    col2.metric(t("MODERATE RISK"), int((snapshot["risk"] == "Moderate Risk").sum()))
    col3.metric(t("HIGH RISK"), int((snapshot["risk"] == "High Risk").sum()))
    col4.metric(t("SCARCITY RISK"), int((snapshot["availability_status"] == "Scarcity Risk").sum()))

    _display_dataframe(snapshot_display(snapshot), width="stretch", hide_index=True)

    alert_rows = []
    for _, zone_row in snapshot.iterrows():
        risk = zone_row["risk"] if zone_row["risk"] in RISK_COLORS else "Safe"
        status = zone_row["availability_status"] if zone_row["availability_status"] in STATUS_COLORS else "Good Availability"
        for alert in generate_alert(risk, status):
            alert_rows.append((zone_row["zone_name"], alert))
    alert_rows.sort(key=lambda item: 0 if item[1]["severity"] == "High" else 1)

    section("Project Alerts")
    if alert_rows:
        for zone_name, alert in alert_rows[:5]:
            message = f"**{zone_name}** — {t(alert['type'])}: {t(alert['message'])}"
            (st.error if alert["severity"] == "High" else st.warning)(message)
        if len(alert_rows) > 5:
            with st.expander(t(f"Show all {len(alert_rows)} alerts")):
                for zone_name, alert in alert_rows[5:]:
                    st.write(f"**{zone_name}** — {alert['severity']} | {t(alert['type'])}: {t(alert['message'])}")
    else:
        st.success(t("No active alerts. All zones look normal in the latest records."))

    # Detailed quality-risk and availability analytical charts are intentionally
    # kept off the public landing page. They remain available in the admin
    # dashboard, where the detailed analytical views belong.

    section("How AquaTrack Works")
    st.markdown('<div class="mobile-how-works-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for col, (icon, title, text) in zip(columns, [
        ("💧", "Quality Assessment", "Seven parameters screened against BIS IS 10500:2012 and classified by a Random Forest."),
        ("📈", "Predictive Forecast", "Linear Regression estimates supply hours from history, rainfall and season."),
        ("📍", "Zone Insights", "Compare project-defined zones using current status, records and map views."),
        ("🚨", "Alerts & Actions", "Risks are highlighted with practical recommendations."),
    ]):
        col.markdown(
            f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
            f'<div class="feature-title">{t(title)}</div><div class="feature-text">{t(text)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="home-disclaimer">{t(DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ZONE ANALYTICS
# ============================================================

elif page == "Zone Analytics":

    st.title(t("📊 Zone-wise Analytics"))

    snapshot = get_zone_snapshot()
    zone_names = snapshot["zone_name"].tolist()

    control_col1, control_col2 = st.columns([2, 1])
    selected_zone = control_col1.selectbox(t("Select Zone"), zone_names, key="analytics_zone")
    period_label = control_col2.selectbox(
        t("Period"), ["Last 30 days", "Last 60 days", "Last 90 days", "All records"], index=3,
        format_func=t, key="analytics_period",
    )
    period_days = {"Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90, "All records": None}[period_label]

    quality_data = get_water_quality(selected_zone)
    availability_data = get_availability(selected_zone)
    prediction_history = get_prediction_history()
    zone_row = snapshot[snapshot["zone_name"] == selected_zone].iloc[0]

    st.markdown(f'<div class="section-heading">📍 {selected_zone}</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("WATER QUALITY"), t(zone_row["risk"]))
    col2.metric(t("BIS PARAMETERS EXCEEDED"), "-" if pd.isna(zone_row["violations"]) else int(zone_row["violations"]))
    col3.metric(t("LATEST SUPPLY"), "-" if pd.isna(zone_row["supply_hours"]) else f"{zone_row['supply_hours']:.2f} h")
    col4.metric(t("AVAILABILITY"), t(zone_row["availability_status"]))

    tab_quality, tab_availability, tab_reports = st.tabs(
        [t("💧 Water Quality"), t("🚰 Water Availability"), t("📄 Reports")]
    )

    with tab_quality:
        if quality_data.empty:
            st.warning(t("No water-quality data found for this zone."))
        else:
            latest = quality_data.iloc[-1].to_dict()
            table, _ = compliance_table(latest)
            st.markdown(t("##### Latest reading against BIS limits"))
            _display_dataframe(table, width="stretch", hide_index=True)

            recent_quality = filter_recent(quality_data, period_days)
            recent_quality["risk"] = predict_quality_batch(recent_quality)

            with st.expander(t("View recent quality records")):
                _display_dataframe(quality_data.tail(20), width="stretch", hide_index=True)

    with tab_availability:
        if availability_data.empty:
            st.warning(t("No availability data found for this zone."))
        else:
            recent_availability = filter_recent(availability_data, period_days).copy()
            recent_availability["supply_hours"] = pd.to_numeric(
                recent_availability["supply_hours"], errors="coerce"
            )
            recent_availability["rainfall"] = pd.to_numeric(
                recent_availability["rainfall"], errors="coerce"
            )

            # Public users get the useful availability values directly instead
            # of an empty chart area. Charts remain available to administrators.
            latest_supply = float(recent_availability.iloc[-1]["supply_hours"])
            average_supply = float(recent_availability["supply_hours"].mean())
            minimum_supply = float(recent_availability["supply_hours"].min())
            maximum_supply = float(recent_availability["supply_hours"].max())

            st.markdown(t("##### Availability summary"))
            summary_cols = st.columns(4)
            summary_cols[0].metric(t("Latest supply"), f"{latest_supply:.2f} h")
            summary_cols[1].metric(t("Average supply"), f"{average_supply:.2f} h")
            summary_cols[2].metric(t("Minimum supply"), f"{minimum_supply:.2f} h")
            summary_cols[3].metric(t("Maximum supply"), f"{maximum_supply:.2f} h")

            status_counts = pd.cut(
                recent_availability["supply_hours"],
                bins=[-float("inf"), 5, 8, float("inf")],
                labels=["Scarcity Risk", "Moderate Availability", "Good Availability"],
                right=False,
            ).value_counts().reindex(
                ["Good Availability", "Moderate Availability", "Scarcity Risk"],
                fill_value=0,
            )

            st.markdown(t("##### Availability status"))
            status_table = pd.DataFrame({
                "Status": [t(x) for x in status_counts.index],
                "Records": status_counts.astype(int).values,
            })
            _display_dataframe(status_table, width="stretch", hide_index=True)

            st.markdown(t("##### Recent availability records"))
            recent_display = recent_availability.tail(20).copy()
            recent_display["date"] = pd.to_datetime(
                recent_display["date"], errors="coerce"
            ).dt.strftime("%d %b %Y")
            recent_display["supply_hours"] = recent_display["supply_hours"].round(2)
            recent_display["rainfall"] = recent_display["rainfall"].round(2)
            recent_display = recent_display[
                ["date", "supply_hours", "rainfall", "season"]
            ].rename(columns={
                "date": "Date",
                "supply_hours": "Supply (h)",
                "rainfall": "Rainfall (mm)",
                "season": "Season",
            })
            _display_dataframe(recent_display, width="stretch", hide_index=True)

    with tab_reports:
        st.write(t("Build the PDF and Excel report for this zone. Files are created only when you ask."))
        summary = {
            "risk": zone_row["risk"],
            "violations": "-" if pd.isna(zone_row["violations"]) else int(zone_row["violations"]),
            "supply": "-" if pd.isna(zone_row["supply_hours"]) else round(float(zone_row["supply_hours"]), 2),
            "status": zone_row["availability_status"],
        }
        if st.button(t("📦 Prepare reports"), key=f"prepare_reports_{selected_zone}"):
            with st.spinner(t("Building reports...")):
                st.session_state["zone_report_files"] = {
                    "zone": selected_zone,
                    "pdf": create_pdf_report(selected_zone, quality_data, availability_data, prediction_history, summary).getvalue(),
                    "xlsx": create_excel_report(selected_zone, quality_data, availability_data, prediction_history, summary).getvalue(),
                }
            st.toast(t("PDF and Excel reports are ready to download."), icon="✅")
        files = st.session_state.get("zone_report_files")
        if files and files["zone"] == selected_zone:
            safe_zone = selected_zone.replace(" ", "_")
            download_col1, download_col2 = st.columns(2)
            download_col1.download_button(
                t("📄 Download PDF Report"), data=files["pdf"],
                file_name=f"AquaTrack_{safe_zone}_Report.pdf", mime="application/pdf",
                key="download_pdf", width="stretch",
            )
            download_col2.download_button(
                t("📊 Download Excel Report"), data=files["xlsx"],
                file_name=f"AquaTrack_{safe_zone}_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_xlsx", width="stretch",
            )


# ============================================================
# COMPARE ZONES
# ============================================================

elif page == "Compare Zones":

    st.title(t("⚖️ Compare Zones"))
    st.write(t("Pick two to six zones and compare their latest water quality and supply side by side."))

    snapshot = get_zone_snapshot()
    zone_names = snapshot["zone_name"].tolist()
    selected = st.multiselect(
        t("Zones to compare"), zone_names, default=zone_names[:3], max_selections=6,
        key="compare_zones",
    )

    if len(selected) < 2:
        st.info(t("Select at least two zones to start comparing."))
    else:
        chosen = snapshot[snapshot["zone_name"].isin(selected)].copy()
        _display_dataframe(snapshot_display(chosen), width="stretch", hide_index=True)

        # Public comparison summary. Chart-only areas are intentionally omitted
        # so the page always contains useful information for residents.
        valid_supply = pd.to_numeric(chosen["supply_hours"], errors="coerce").dropna()
        valid_violations = pd.to_numeric(chosen["violations"], errors="coerce").dropna()

        risk_counts = chosen["risk"].value_counts()
        availability_counts = chosen["availability_status"].value_counts()

        highest_supply_zone = None
        lowest_supply_zone = None
        if not valid_supply.empty:
            highest_supply_zone = chosen.loc[chosen["supply_hours"].astype(float).idxmax(), "zone_name"]
            lowest_supply_zone = chosen.loc[chosen["supply_hours"].astype(float).idxmin(), "zone_name"]

        highest_risk_count = int(risk_counts.get("High Risk", 0))
        moderate_risk_count = int(risk_counts.get("Moderate Risk", 0))
        safe_count = int(risk_counts.get("Safe", 0))
        total_violations = int(valid_violations.sum()) if not valid_violations.empty else 0
        average_supply = float(valid_supply.mean()) if not valid_supply.empty else None

        section("Comparison Summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric(t("Zones compared"), len(chosen))
        summary_cols[1].metric(t("Average supply"), f"{average_supply:.2f} h" if average_supply is not None else "—")
        summary_cols[2].metric(t("Total BIS violations"), total_violations)
        summary_cols[3].metric(t("High-risk zones"), highest_risk_count)

        section("Key observations")

        observations = []
        if highest_supply_zone is not None:
            highest_value = float(chosen.loc[chosen["zone_name"] == highest_supply_zone, "supply_hours"].iloc[0])
            observations.append(
                t(f"{highest_supply_zone} has the highest latest supply at {highest_value:.2f} hours.")
            )
        if lowest_supply_zone is not None and lowest_supply_zone != highest_supply_zone:
            lowest_value = float(chosen.loc[chosen["zone_name"] == lowest_supply_zone, "supply_hours"].iloc[0])
            observations.append(
                t(f"{lowest_supply_zone} has the lowest latest supply at {lowest_value:.2f} hours.")
            )

        if highest_risk_count:
            observations.append(
                t(f"{highest_risk_count} of the selected zones are classified as High Risk.")
            )
        elif moderate_risk_count:
            observations.append(
                t(f"{moderate_risk_count} of the selected zones are classified as Moderate Risk, with no High Risk zones in this selection.")
            )
        elif safe_count == len(chosen):
            observations.append(t("All selected zones are currently classified as Safe."))

        if total_violations:
            observations.append(
                t(f"The selected zones have {total_violations} recorded BIS parameter exceedances in the latest readings.")
            )
        else:
            observations.append(t("No BIS parameter exceedances are recorded in the latest readings for the selected zones."))

        for item in observations:
            st.markdown(f"- {item}")

        section("Availability breakdown")
        availability_order = ["Good Availability", "Moderate Availability", "Scarcity Risk"]
        availability_table = pd.DataFrame({
            "Availability": [t(status) for status in availability_order],
            "Zones": [int(availability_counts.get(status, 0)) for status in availability_order],
        })
        _display_dataframe(availability_table, width="stretch", hide_index=True)


# ============================================================
# INDORE ZONE MAP
# ============================================================

elif page == "Indore Zone Map":

    # Keep the map page title aligned with the Resident Water Alerts title.
    # The secondary analytics navigation is hidden on this page, so restore
    # the same vertical breathing room before the main heading.
    st.markdown(
        """
        <div class="indore-map-heading-marker" aria-hidden="true"></div>
        <style>
        body:has(.indore-map-heading-marker) h1 {
            margin-top: 1.15rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(t("Indore Zone Map"))
    st.write(t("Interactive map of the project-defined AquaTrack zones. Use the layer control for satellite view."))

    snapshot = get_zone_snapshot()

    # Bottom-align the switches with the multiselect control, not its label.
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    risk_filter = filter_col1.multiselect(
        t("Show zones by water-quality risk"), RISK_ORDER + ["No Data"],
        default=RISK_ORDER + ["No Data"], format_func=t, key="map_risk_filter",
    )
    show_circles = filter_col2.toggle(t("Supply rings"), value=True, key="map_circles")
    show_bodies = filter_col3.toggle(t("Water bodies"), value=True, key="map_bodies")

    indore_map = make_base_map([22.7196, 75.8577], 11)

    for _, zone in snapshot[snapshot["risk"].isin(risk_filter)].iterrows():
        if pd.isna(zone["latitude"]) or pd.isna(zone["longitude"]):
            continue
        location = [float(zone["latitude"]), float(zone["longitude"])]
        supply_text = "No Data" if pd.isna(zone["supply_hours"]) else f"{zone['supply_hours']:.2f} h"
        violations_text = "-" if pd.isna(zone["violations"]) else int(zone["violations"])

        water_quality_label = t("Water quality")
        bis_label = t("BIS parameters exceeded")
        supply_label = t("Supply")
        availability_label = t("Availability")
        risk_value = t(zone["risk"])
        status_value = t(zone["availability_status"])

        popup_html = f"""
        <div style="font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px;min-width:190px;">
            <b style="font-size:14px;">{zone['zone_name']}</b><br>
            {water_quality_label}: <b>{risk_value}</b><br>
            {bis_label}: {violations_text}<br>
            {supply_label}: {supply_text}<br>
            {availability_label}: {status_value}
        </div>
        """

        if show_circles and pd.notna(zone["supply_hours"]):
            folium.Circle(
                location=location, radius=900,
                color=STATUS_COLORS.get(zone["availability_status"], "#64748b"),
                fill=True, fill_opacity=0.15, weight=1,
            ).add_to(indore_map)

        folium.Marker(
            location=location,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{zone['zone_name']} — {zone['risk']}",
            icon=folium.Icon(color=MARKER_COLORS.get(zone["risk"], "blue"), icon="tint", prefix="fa"),
        ).add_to(indore_map)

    if show_bodies:
        for _, body in get_water_bodies().iterrows():
            if pd.isna(body["latitude"]) or pd.isna(body["longitude"]):
                continue
            folium.CircleMarker(
                location=[float(body["latitude"]), float(body["longitude"])],
                radius=7, color="#075985", fill=True, fill_color="#0ea5e9", fill_opacity=0.9,
                tooltip=f"{body['name']} ({body['water_body_type']})",
                popup=folium.Popup(f"<b>{body['name']}</b><br>{body['location']}", max_width=260),
            ).add_to(indore_map)

    add_map_legend(indore_map, [
        (t("Safe"), "#16a34a"), (t("Moderate risk"), "#f59e0b"), (t("High risk"), "#dc2626"),
        (t("No data"), "#3b82f6"), (t("Water body"), "#0ea5e9"),
    ])

    st_folium(indore_map, width=None, height=600, key="zone_map", returned_objects=[])

    st.info(
        t("These are representative, project-defined locations used for the prototype and are not official municipal zone boundaries.")
    )


# ============================================================
# WATER QUALITY PREDICTION
# ============================================================

elif page == "Water Quality Prediction":

    st.title(t("💧 Water Quality Risk Prediction"))
    st.write(t("Enter water-quality parameters to get the model's risk category, a BIS limit check and the model's confidence."))

    st.markdown('<div class="quality-check-mode-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
    check_mode = st.radio(
        t("How would you like to check the water?"),
        [t("I have a water test report"), t("I do not have a water test report")],
        horizontal=True,
        key="quality_check_mode",
    )

    if check_mode == t("I do not have a water test report"):
        st.markdown('<div class="without-report-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
        st.info(t("No report? You can do a simple visual check instead. This is only a preliminary check and not a laboratory test."))
        st.subheader(t("Check the water using simple questions"))

        observation_items = [
            ("clear", "1. Is the water clear?"),
            ("dirt", "2. Can you see dirt or waste in the water?"),
            ("smell", "3. Does the water smell bad?"),
            ("colour", "4. Does the water look yellow, brown or another unusual colour?"),
            ("cloudy", "5. Does the water look very cloudy?"),
            ("taste", "6. Does the water taste strange or bad?"),
        ]
        observations = {}
        col1, col2 = st.columns(2, gap="large")
        for index, (key, question) in enumerate(observation_items):
            target = col1 if index % 2 == 0 else col2
            with target:
                st.markdown('<div class="simple-water-item-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
                selected_answer = st.radio(
                    t(question),
                    [t("Yes"), t("No")],
                    horizontal=True,
                    key=f"simple_water_{key}",
                )
                observations[key] = selected_answer == t("Yes")

        st.caption(t("Do not intentionally taste water that may be unsafe."))

        st.session_state.setdefault("simple_water_result", None)

        if st.button(t("Check Water"), type="primary", key="simple_water_check"):
            issue_keys = ["dirt", "smell", "colour", "cloudy", "taste"]
            issue_count = sum(bool(observations[key]) for key in issue_keys)
            clear_problem = not observations["clear"]

            if issue_count == 0 and not clear_problem:
                st.session_state.simple_water_result = "normal"
            elif issue_count <= 2 and not clear_problem:
                st.session_state.simple_water_result = "attention"
            else:
                st.session_state.simple_water_result = "hazard"

        simple_water_result = st.session_state.get("simple_water_result")

        if simple_water_result == "normal":
            st.success(t("The water looks normal from the information provided."))
        elif simple_water_result == "attention":
            st.warning(t("Some things about the water need attention."))
        elif simple_water_result == "hazard":
            st.markdown(
                f'''<div class="simple-water-hazard">
                    <strong>{t("Water quality warning")}</strong><br>
                    {t("Several problems are visible in the water. Please do not drink it until the water has been properly tested.")}<br><br>
                    {t("Please report this problem to the system so it can be reviewed.")}
                </div>''',
                unsafe_allow_html=True,
            )

            st.markdown(
                "<span class='report-simple-water-problem-marker'></span>",
                unsafe_allow_html=True,
            )

            if st.button(
                t("Report this problem to the system"),
                key="report_simple_water_problem",
                type="primary",
            ):
                navigate_to("Report Water Body Issue")
                st.rerun()

        st.info(t("This is only a preliminary check based on what can be seen or noticed. It cannot measure things that are not visible."))

    else:
        snapshot = get_zone_snapshot()
        zone_names = snapshot["zone_name"].tolist()
        selected_zone = st.selectbox(t("Select Zone"), zone_names, key="pred_zone")
        zone_row = snapshot[snapshot["zone_name"] == selected_zone].iloc[0]

        def default_value(key):
            value = zone_row.get(key)
            return float(value) if pd.notna(value) else INPUT_DEFAULTS[key]

        st.caption(t("Fields start with the latest recorded reading for this zone. Change any value and press Predict."))

        with st.form("quality_form"):
            col1, col2 = st.columns(2)
            values = {}
            with col1:
                values["ph"] = st.number_input(t("pH"), 0.0, 14.0, round(default_value("ph"), 2), 0.1, key=f"q_ph_{selected_zone}")
                values["tds"] = st.number_input(t("TDS (mg/L)"), 0.0, value=round(default_value("tds"), 1), step=10.0, key=f"q_tds_{selected_zone}")
                values["turbidity"] = st.number_input(t("Turbidity (NTU)"), 0.0, value=round(default_value("turbidity"), 2), step=0.1, key=f"q_turb_{selected_zone}")
                values["hardness"] = st.number_input(t("Hardness (mg/L)"), 0.0, value=round(default_value("hardness"), 1), step=10.0, key=f"q_hard_{selected_zone}")
            with col2:
                values["chloride"] = st.number_input(t("Chloride (mg/L)"), 0.0, value=round(default_value("chloride"), 1), step=10.0, key=f"q_chl_{selected_zone}")
                values["fluoride"] = st.number_input(t("Fluoride (mg/L)"), 0.0, value=round(default_value("fluoride"), 2), step=0.1, key=f"q_flu_{selected_zone}")
                values["nitrate"] = st.number_input(t("Nitrate (mg/L)"), 0.0, value=round(default_value("nitrate"), 1), step=1.0, key=f"q_nit_{selected_zone}")
                save_result = st.checkbox(t("Save this prediction to history"), value=True, key="q_save")
            submitted = st.form_submit_button(t("🔍 Predict Water Quality"), type="primary")

        if submitted:
            sample = {DB_TO_MODEL[key]: value for key, value in values.items()}
            risk = predict_water_quality(sample)
            probabilities = quality_probabilities(sample)
            table, violations = compliance_table(values)
            rule_risk = rule_label(violations)
            st.subheader(t("Prediction Result"))
            st.write(f"**{t('Zone')}:** {selected_zone}")
            (st.success if risk == "Safe" else st.warning if risk == "Moderate Risk" else st.error)(f"{t('Model risk level:')} {t(risk)}")
            col1, col2, col3 = st.columns(3)
            col1.metric(t("MODEL PREDICTION"), t(risk))
            col2.metric(t("BIS PARAMETERS EXCEEDED"), f"{violations} / 7")
            col3.metric(t("MODEL CONFIDENCE"), f"{probabilities.max() * 100:.0f}%")
            if rule_risk != risk:
                st.info(f"{t('The direct BIS count suggests')} **{t(rule_risk)}**, {t('while the model predicted')} **{t(risk)}**. {t('This can happen for borderline readings; check the parameter table below.')}")
            st.markdown(t("##### BIS limit check"))
            _display_dataframe(table, width="stretch", hide_index=True)
            if save_result:
                saved, error = save_prediction(selected_zone, risk=risk)
                if saved:
                    st.toast(t("Prediction saved to history."), icon="✅")
                else:
                    st.toast(f"{t('Prediction could not be saved:')} {error}", icon="⚠️")
            alerts = generate_alert(risk, "Good Availability")
            if alerts:
                st.subheader(t("🚨 Alerts"))
                for alert in alerts:
                    st.warning(f"{alert['severity']} | {t(alert['message'])}")
            st.subheader(t("💡 Recommendations"))
            for recommendation in generate_recommendations(risk, "Good Availability"):
                st.write("•", t(recommendation))


# ============================================================
# AVAILABILITY FORECAST
# ============================================================

elif page == "Availability Forecast":

    st.title(t("📈 Water Availability Forecast"))
    st.write(
        t("Forecast water supply from the previous day's supply, rainfall and season. Choose more than one day to see a rolling forecast.")
    )

    zones = get_zones()
    selected_zone = st.selectbox(t("Select Zone"), zones["zone_name"].tolist(), key="forecast_zone")

    zone_availability = get_availability(selected_zone)
    if zone_availability.empty:
        st.warning(t("No historical availability data is available for the selected zone."))
    else:
        zone_availability["date"] = pd.to_datetime(zone_availability["date"])
        zone_availability = zone_availability.sort_values("date")
        latest_record = zone_availability.iloc[-1]
        previous_supply_hours = float(latest_record["supply_hours"])
        latest_date = latest_record["date"]

        st.info(
            f"{t('Starting point:')} {previous_supply_hours:.2f} {t('Hours')} — {latest_date.strftime('%d %b %Y')}"
        )

        default_season = str(latest_record["season"])
        season_options = list(SEASON_CODES)

        if st.session_state.get("admin_logged_in"):
            # Administrator mode: retain the technical controls used to create
            # and store a new forecast.
            with st.form("forecast_form"):
                col1, col2, col3 = st.columns(3)
                rainfall = col1.number_input(
                    t("Rainfall (mm)"), 0.0, value=float(latest_record["rainfall"]), step=0.5,
                    key=f"f_rain_{selected_zone}",
                )
                season = col2.selectbox(
                    t("Season"), season_options, format_func=t,
                    index=season_options.index(default_season) if default_season in season_options else 0,
                    key=f"f_season_{selected_zone}",
                )
                horizon = col3.slider(t("Forecast days"), 1, 7, 3, key="f_days")
                save_result = st.checkbox(t("Save the next-day forecast to history"), value=True, key="f_save")
                submitted = st.form_submit_button(t("📈 Forecast Availability"), type="primary")

            if submitted:
                series = forecast_series(previous_supply_hours, rainfall, season, horizon)
                first = series.iloc[0]
                forecast, status = float(first["forecast_hours"]), first["status"]

                st.subheader(t("Forecast Result"))
                st.write(f"**{t('Zone')}:** {selected_zone}")
                col1, col2, col3 = st.columns(3)
                col1.metric(t("NEXT-DAY FORECAST"), f"{forecast} h")
                col2.metric(t("AVAILABILITY STATUS"), t(status))
                col3.metric(f"{t('LOWEST OVER')} {horizon} {t('DAY(S)')}", f"{series['forecast_hours'].min()} h")
                (st.success if status == "Good Availability" else st.warning if status == "Moderate Availability" else st.error)(t(status))

                series["Date"] = [latest_date + pd.Timedelta(days=int(d)) for d in series["day"]]
                history_tail = zone_availability.tail(14)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_tail["date"], y=history_tail["supply_hours"],
                    mode="lines+markers", name="Recorded supply", line=dict(color="#0e9ed3"),
                ))
                fig.add_trace(go.Scatter(
                    x=[latest_date] + series["Date"].tolist(),
                    y=[previous_supply_hours] + series["forecast_hours"].tolist(),
                    mode="lines+markers", name="Forecast", line=dict(color="#f59e0b", dash="dash"),
                ))
                fig.add_hline(y=5, line_dash="dot", line_color="#dc2626", annotation_text="Scarcity below 5 h")
                fig.add_hline(y=8, line_dash="dot", line_color="#16a34a", annotation_text="Good from 8 h")
                fig.update_layout(title=t("Recorded supply and forecast"), yaxis_title="Hours")
                show_chart(fig, 380)

                if horizon > 1:
                    display = series[["Date", "forecast_hours", "status"]].copy()
                    display["Date"] = display["Date"].dt.strftime("%d %b %Y")
                    display["status"] = display["status"].map(status_badge)
                    _display_dataframe(
                        display.rename(columns={"forecast_hours": "Forecast (h)", "status": "Status"}),
                        width="stretch", hide_index=True,
                    )
                    st.caption(t("Later days reuse each forecast as the next day's previous supply and keep rainfall and season constant, so uncertainty grows with the horizon."))

                if save_result:
                    saved, error = save_prediction(selected_zone, forecast=forecast)
                    if saved:
                        st.toast(t("Availability forecast saved to history."), icon="✅")
                    else:
                        st.toast(f"{t('Forecast could not be saved:')} {error}", icon="⚠️")

                alerts = generate_alert("Safe", status)
                if alerts:
                    st.subheader(t("🚨 Alerts"))
                    for alert in alerts:
                        st.warning(f"{alert['severity']} | {t(alert['message'])}")

                st.subheader(t("💡 Recommendations"))
                for recommendation in generate_recommendations("Safe", status):
                    st.write("•", t(recommendation))

        else:
            # Public mode: citizens select only the zone. Rainfall and season
            # come from the latest recorded availability row; no technical
            # model inputs or database-save controls are exposed.
            public_horizon = 7
            public_rainfall = float(latest_record["rainfall"])
            public_season = default_season
            series = forecast_series(
                previous_supply_hours, public_rainfall, public_season, public_horizon
            )
            first = series.iloc[0]
            forecast, status = float(first["forecast_hours"]), first["status"]

            st.subheader(t("Water Availability Forecast"))
            st.caption(
                t("The forecast uses the latest recorded rainfall and season data for the selected zone.")
            )

            result_cols = st.columns(3)
            result_cols[0].metric(t("NEXT-DAY FORECAST"), f"{forecast:.2f} h")
            result_cols[1].metric(t("AVAILABILITY STATUS"), t(status))
            result_cols[2].metric(
                t("7-DAY LOWEST SUPPLY"), f"{series['forecast_hours'].min():.2f} h"
            )
            (
                st.success
                if status == "Good Availability"
                else st.warning
                if status == "Moderate Availability"
                else st.error
            )(t(status))

            series["Date"] = [
                latest_date + pd.Timedelta(days=int(d)) for d in series["day"]
            ]
            display = series[["Date", "forecast_hours", "status"]].copy()
            display["Date"] = display["Date"].dt.strftime("%d %b %Y")
            display["forecast_hours"] = display["forecast_hours"].round(2)
            display["status"] = display["status"].map(status_badge)
            _display_dataframe(
                display.rename(
                    columns={"forecast_hours": "Forecast (h)", "status": "Status"}
                ),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                t("This is a project-level forecast for screening and decision support. It should not be treated as an official municipal supply schedule.")
            )


# ============================================================
# WORKER DASHBOARD
# ============================================================

elif page == "Worker Dashboard":
    if not st.session_state.get("worker_logged_in"):
        st.session_state.page = "Worker Login"
        st.rerun()
    worker = st.session_state.worker_user
    worker_id = int(worker["worker_id"])
    unread = get_unread_count("worker", worker_id)
    st.title(t("👷 Worker Dashboard"))
    st.caption(f"{t('Signed in as')}: {worker['name']} · {t('Status')}: {t(str(worker['availability_status']))}")

    assigned = get_service_requests(worker_id=worker_id)
    active = assigned[assigned["workflow_status"].isin(["Assigned", "In Progress", "Reopened"])] if not assigned.empty else assigned
    waiting = assigned[assigned["workflow_status"] == "Waiting for Verification"] if not assigned.empty else assigned
    completed = assigned[assigned["workflow_status"].isin(["Waiting for Verification", "Resolved"])] if not assigned.empty else assigned
    verified_completed = assigned[assigned["workflow_status"] == "Resolved"] if not assigned.empty else assigned
    # Personal completion means the worker has submitted the job for verification
    # or the admin has already verified it.
    completed_work = len(completed)
    completed_sites = 0
    if not completed.empty and "water_body_id" in completed.columns:
        completed_sites = int(completed["water_body_id"].dropna().nunique())

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric(t("Active"), len(active))
    c2.metric(t("Assigned"), int((assigned["workflow_status"] == "Assigned").sum()) if not assigned.empty else 0)
    c3.metric(t("In Progress"), int((assigned["workflow_status"] == "In Progress").sum()) if not assigned.empty else 0)
    c4.metric(t("Verification"), len(waiting))
    c5.metric(t("Notifications"), unread)
    c6.metric(t("Completed Work"), completed_work)

    st.caption(
        f"{t('Sites personally completed')}: {completed_sites} · "
        f"{t('Total completed work')}: {completed_work} · "
        f"{t('Verified / Resolved')}: {len(verified_completed)}"
    )

    st.subheader(t("My Assigned Requests"))
    if active.empty:
        st.info(t("No active requests are currently assigned to you."))
    for _, req in active.iterrows():
        report_id = int(req["report_id"])
        with st.container(border=True):
            st.markdown(f"### #{report_id} — {req['issue_type']}")
            st.write(f"**{t('Zone')}:** {req['zone_name']}  ·  **{t('Severity')}:** {req['severity']}")
            st.write(f"**{t('Description')}:** {req['description']}")
            if pd.notna(req.get("photo_path")) and str(req.get("photo_path") or "").strip() and (BASE_DIR / str(req["photo_path"])).exists():
                st.image(str(BASE_DIR / str(req["photo_path"])), width=260)

            # Show the citizen-reported location first, with the water body's
            # stored location as a clearly-labelled fallback when GPS is absent.
            st.markdown(f"**{t('Report Location')}**")
            location_details = [
                (t("Zone"), req.get("zone_name")),
                (t("Water body"), req.get("water_body")),
                (t("Colony / Locality"), req.get("colony_name")),
                (t("Society / Building"), req.get("society_name")),
                (t("Water body location"), req.get("water_body_location")),
            ]
            visible_location_details = []
            for location_label, location_value in location_details:
                if pd.notna(location_value):
                    clean_location_value = str(location_value).strip()
                    if clean_location_value and clean_location_value.lower() not in {"nan", "none", "nat"}:
                        visible_location_details.append(f"**{location_label}:** {clean_location_value}")
            if visible_location_details:
                st.markdown("  \n".join(visible_location_details))

            issue_lat, issue_lon = req.get("latitude"), req.get("longitude")
            water_lat, water_lon = req.get("water_body_latitude"), req.get("water_body_longitude")
            issue_has_gps = pd.notna(issue_lat) and pd.notna(issue_lon)
            water_has_gps = pd.notna(water_lat) and pd.notna(water_lon)

            if issue_has_gps:
                issue_lat, issue_lon = float(issue_lat), float(issue_lon)
                st.caption(f"{t('Coordinates')}: {issue_lat:.6f}, {issue_lon:.6f}")
                st.link_button(
                    t("Open exact location in Google Maps"),
                    f"https://www.google.com/maps/search/?api=1&query={issue_lat:.6f}%2C{issue_lon:.6f}",
                    key=f"worker_issue_location_{report_id}",
                    width="stretch",
                )
            elif water_has_gps:
                water_lat, water_lon = float(water_lat), float(water_lon)
                st.caption(t("GPS location is not available for the issue; this link shows the associated water body instead."))
                st.link_button(
                    t("Open water body location in Google Maps"),
                    f"https://www.google.com/maps/search/?api=1&query={water_lat:.6f}%2C{water_lon:.6f}",
                    key=f"worker_water_body_location_{report_id}",
                    width="stretch",
                )
            else:
                st.caption(t("GPS coordinates were not captured for this report. Use the zone and locality details above."))

            if req["workflow_status"] in ["Assigned", "Reopened"]:
                with st.container(key=f"worker_start_work_{report_id}"):
                    if st.button(t("▶ Start Work"), key=f"start_{report_id}", type="primary"):
                        ok, msg = change_request_status(report_id, "In Progress", "worker", worker_id)
                        (st.success if ok else st.error)(t(msg))
                        if ok: st.rerun()
            elif req["workflow_status"] == "In Progress":
                st.caption(t("After you mark this job as done, it will be sent to the admin for final verification."))
                remarks = st.text_area(t("Work remarks (required)"), key=f"remarks_{report_id}")
                evidence = st.file_uploader(t("Upload completion evidence (optional)"), type=["jpg","jpeg","png","webp"], key=f"evidence_{report_id}")
                if st.button(t("Mark as Done / Solved"), key=f"complete_{report_id}", type="primary"):
                    evidence_path = None
                    if evidence is not None:
                        if evidence.size > 5 * 1024 * 1024:
                            st.error(t("Evidence image must be 5 MB or smaller."))
                            st.stop()
                        try:
                            img = Image.open(evidence)
                            img.verify()
                            evidence.seek(0)
                        except Exception:
                            st.error(t("Uploaded file is not a valid image."))
                            st.stop()
                        ext = Path(evidence.name).suffix.lower()
                        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                            ext = ".jpg"
                        saved_name = f"evidence_{secrets.token_hex(16)}{ext}"
                        file_path = folder / saved_name
                        file_path.write_bytes(evidence.getbuffer())
                        evidence_path = str(file_path.relative_to(BASE_DIR))
                    ok, msg = complete_worker_request(report_id, worker_id, remarks, evidence_path)
                    (st.success if ok else st.error)(t(msg))
                    if ok: st.rerun()



# ============================================================
# MY REQUESTS
# ============================================================

elif page == "My Requests":
    if not st.session_state.citizen_logged_in:
        require_citizen_login("My Requests"); st.rerun()
    st.title(t("📋 My Requests"))
    resident_id = st.session_state.get("resident_id")
    if resident_id:
        requests_df = get_service_requests(resident_id=int(resident_id))
    else:
        st.info(t("Submit a problem report with your contact details first. Your request history will appear here."))
        requests_df = pd.DataFrame()

    if not requests_df.empty:
        unread = get_unread_count("resident", int(resident_id))
        c1, c2, c3 = st.columns(3)
        c1.metric(t("Your Requests"), len(requests_df))
        c2.metric(t("Resolved"), int((requests_df["workflow_status"] == "Resolved").sum()))
        c3.metric(t("Unread Notifications"), unread)
        display_cols = [c for c in ["report_id", "zone_name", "issue_type", "severity", "workflow_status", "created_at", "worker_name", "resolved_at"] if c in requests_df.columns]
        _display_dataframe(requests_df[display_cols], width="stretch", hide_index=True)
        for _, req in requests_df.head(10).iterrows():
            report_id = int(req["report_id"])
            with st.expander(f"Request #{report_id} — {t(str(req['workflow_status']))}"):
                st.write(f"**{t('Problem')}:** {req['issue_type']}")
                st.write(f"**{t('Zone')}:** {req['zone_name']}")
                st.write(f"**{t('Severity')}:** {req['severity']}")
                st.write(f"**{t('Description')}:** {req['description']}")
                st.write(f"**{t('Worker')}:** {req['worker_name'] if pd.notna(req['worker_name']) else t('Not assigned yet')}")
                st.markdown("### " + t("Request Timeline"))
                history = get_request_history(report_id)
                if history.empty:
                    st.info(t("No status history available."))
                else:
                    for _, event in history.iterrows():
                        st.markdown(f"**{event['created_at']}** · {t(str(event['new_status']))} · {event['changed_by_role']}  \n{event['remarks'] or ''}")
                if str(req["workflow_status"]) == "Resolved":
                    existing = db_query("SELECT review_id FROM citizen_reviews WHERE report_id=? AND resident_id=?", (report_id, int(resident_id)))
                    if existing.empty:
                        st.success(t("This request is resolved. You can now submit a review from the Reviews page."))
    else:
        st.info(t("No requests found yet."))
    if resident_id:
        st.subheader(t("🔔 Request Notifications"))
        resident_notes = get_notifications("resident", int(resident_id))
        if resident_notes.empty:
            st.info(t("No request notifications yet."))
        else:
            for _, note in resident_notes.iterrows():
                marker = "🟠" if int(note.get("is_read", 0) or 0) == 0 else "⚪"
                st.info(f"{marker} {note['created_at']} · {note['message']}")
            if st.button(t("Mark all notifications as read"), key="resident_mark_all_read"):
                mark_notifications_read("resident", int(resident_id))
                st.rerun()


# ============================================================
# REVIEWS
# ============================================================

elif page == "Reviews":
    if not st.session_state.citizen_logged_in:
        require_citizen_login("Reviews")
        st.rerun()
    else:
        resident_id = int(st.session_state.resident_id)

        st.title(t("Reviews & Feedback"))
        st.caption(
            t(
                "Any signed-in citizen can review a resolved request. "
                "Each account can review a request once."
            )
        )

        st.markdown("""
        <style>
        [data-testid="stFeedback"] button,
        [data-testid="stFeedback"] button * {
            color: #f5a623 !important;
            fill: #f5a623 !important;
        }

        [data-testid="stFeedback"] button:hover {
            transform: scale(1.15);
        }
        </style>
        """, unsafe_allow_html=True)

        resolved = get_service_requests(status="Resolved", resident_id=resident_id)
        reviewable = []

        for _, req in resolved.iterrows():
            if db_query(
                "SELECT review_id FROM citizen_reviews "
                "WHERE report_id=? AND resident_id=? LIMIT 1",
                (int(req["report_id"]), resident_id)
            ).empty:
                reviewable.append(req)

        if not reviewable:
            st.success(
                t("There are no resolved requests awaiting your review.")
            )

        for req in reviewable[:10]:
            report_id = int(req["report_id"])

            with st.container(border=True):
                st.markdown(
                    f"**#{report_id} · {req['issue_type']}**"
                )

                st.caption(
                    f"{t('Zone')}: {req['zone_name']} · "
                    f"{t('Resolved')}: {req.get('resolved_at') or '—'}"
                )

                with st.form(f"review_form_{report_id}"):
                    st.markdown(
                        "<span class='service-rating-marker'></span>",
                        unsafe_allow_html=True
                    )

                    st.markdown(t("Service rating"))

                    rating = st.feedback(
                        "stars",
                        key=f"rating_{report_id}"
                    )

                    if rating is not None:
                        stars = rating + 1
                    else:
                        stars = None

                    comment = st.text_area(
                        t("Describe your experience"),
                        placeholder=t(
                            "Describe the service, the work completed, "
                            "and anything you would like the admin team to know."
                        ),
                        key=f"review_comment_{report_id}",
                        max_chars=1000
                    )

                    if st.form_submit_button(
                        t("Submit Review"),
                        type="primary"
                    ):
                        if stars is None:
                            st.error(t("Please select a service rating."))
                        else:
                            ok, message = submit_review(
                                report_id,
                                resident_id,
                                stars,
                                comment
                            )

                            if ok:
                                st.success(t(message))
                                st.rerun()
                            else:
                                st.error(t(message))


# ============================================================
# RESIDENT ALERTS
# ============================================================

elif page == "Resident Alerts":
    if not st.session_state.citizen_logged_in:
        require_citizen_login("Resident Alerts"); st.rerun()
    else:
        resident_id=int(st.session_state.resident_id)
        notes=get_notifications("resident",resident_id)
        st.title(t("Notifications"))
        st.caption(t("Updates about your reports appear here."))
        if notes.empty:
            st.info(t("No notifications yet."))
        else:
            unread_count = int((notes["is_read"].fillna(0).astype(int) == 0).sum())

            for _,note in notes.iterrows():
                unread=int(note.get("is_read",0) or 0)==0
                marker="●" if unread else "○"
                st.markdown(f"<div class='citizen-notification-row {'unread' if unread else ''}'><div class='citizen-notification-marker'>{marker}</div><div><strong>{note['message']}</strong><div class='citizen-notification-time'>{note['created_at']}</div></div></div>",unsafe_allow_html=True)

            if unread_count > 0:
                if st.button(t("Mark all as read"),key="citizen_mark_all_read"):
                    db_execute(
                        "UPDATE aquatrack_notifications SET is_read=1 WHERE recipient_role=? AND recipient_id=? AND COALESCE(is_read,0)=0",
                        ("resident", int(resident_id)),
                    )
                    st.rerun()
            else:
                st.success(t("All notifications are already marked as read."))


# ============================================================
# ADMIN PANEL
# ============================================================

elif page == "Admin Panel":

    admin_unread = get_unread_count("admin", None)
    st.title(t("AquaTrack Admin Panel"))
    if admin_unread:
        st.info(f" {admin_unread} {t('unread admin notification(s) require attention.')}")

    admin_section = st.radio(
        t("Admin section"),
        [
            "Overview",
            "Water Body Reports",
            "Workers",
            "Citizen Reviews",
            "System Insights",
            "ML Model Performance",
            "Resident Alerts",
            "Data Import",
            "Predictions & Exports",
            "System",
        ],
        format_func=t,
        horizontal=True,
        label_visibility="collapsed",
        key="admin_section",
        on_change=close_admin_notification_popup,
    )
    st.divider()

    if admin_section == "Water Body Reports":
        water_body_reports = get_water_body_reports()
        water_bodies = get_water_bodies()

        # ---------------- REPORT STATISTICS ----------------
        st.subheader(t("Report Statistics"))
        workflow = water_body_reports["workflow_status"].fillna("Submitted") if not water_body_reports.empty else pd.Series(dtype=str)
        total_reports = len(water_body_reports)
        new_reopened = int(workflow.isin(["Submitted", "Reopened"]).sum()) if not water_body_reports.empty else 0
        assigned = int((workflow == "Assigned").sum()) if not water_body_reports.empty else 0
        in_progress = int((workflow == "In Progress").sum()) if not water_body_reports.empty else 0
        verification = int(workflow.isin(["Waiting for Verification"]).sum()) if not water_body_reports.empty else 0
        resolved = int((workflow == "Resolved").sum()) if not water_body_reports.empty else 0
        high_priority = int((water_body_reports["severity"] == "High").sum()) if not water_body_reports.empty else 0

        stats_row_1 = st.columns(4)
        stats_row_1[0].metric(t(" Total Reports"), total_reports)
        stats_row_1[1].metric(t(" New / Reopened"), new_reopened)
        stats_row_1[2].metric(t(" Assigned"), assigned)
        stats_row_1[3].metric(t(" In Progress"), in_progress)

        stats_row_2 = st.columns(3)
        stats_row_2[0].metric(t("⏳ Verification"), verification)
        stats_row_2[1].metric(t(" Resolved"), resolved)
        stats_row_2[2].metric(t(" High Priority"), high_priority)
        st.caption(f"{t('Project-defined water bodies available:')} {len(water_bodies)}")

        if water_body_reports.empty:
            st.info(t("No water-body issue reports have been submitted yet."))
        else:
            # ---------------- ISSUE / SEVERITY CHARTS ----------------
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                by_issue = water_body_reports["issue_type"].value_counts().reset_index()
                by_issue.columns = ["Issue type", "Reports"]
                fig = px.bar(by_issue, x="Reports", y="Issue type", orientation="h", title=t("Reports by issue type"))
                fig.update_traces(marker_color="#12a9db")
                show_chart(fig, 300)
            with chart_col2:
                by_severity = water_body_reports["severity"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
                fig = px.bar(
                    x=by_severity.index, y=by_severity.values, title=t("Reports by severity"),
                    color=by_severity.index,
                    color_discrete_map={"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"},
                    labels={"x": "", "y": "Reports"},
                )
                fig.update_layout(showlegend=False)
                show_chart(fig, 300)

            # ---------------- FILTER + CSV ----------------
            workflow_filters = [
                "All", "Submitted", "Assigned", "In Progress",
                "Waiting for Verification", "Resolved", "Reopened",
            ]
            filter_col1, filter_col2 = st.columns([3, 1])
            selected_filter = filter_col1.radio(
                t("Report filter"), workflow_filters,
                format_func=t, horizontal=True, label_visibility="collapsed", key="water_body_report_filter",
            )
            filter_col2.download_button(
                t("⬇️ Export CSV"),
                data=water_body_reports.to_csv(index=False).encode("utf-8"),
                file_name="aquatrack_water_body_reports.csv", mime="text/csv",
                key="export_reports_csv", width="stretch",
            )

            filtered = (
                water_body_reports.copy() if selected_filter == "All"
                else water_body_reports[water_body_reports["workflow_status"].fillna("Submitted") == selected_filter].copy()
            )
            st.markdown(f"### {t('Reports')} — {len(filtered)} {t('shown')}")

            column_weights = [0.75, 2.0, 3.1, 1.35, 1.45, 1.75, 1.8]
            table_header = st.columns(column_weights, gap="small")
            for header, label in zip(
                table_header,
                ["PHOTO", "LOCATION", "DESCRIPTION", "DATE", "SEVERITY", "WORKER", "STATUS"],
            ):
                header.markdown(f"**{t(label)}**")
            st.divider()

            severity_options = ["Low", "Medium", "High"]
            assignment_workers = get_workers(active_only=True)
            assignment_workers = assignment_workers[
                assignment_workers["availability_status"].fillna("Offline") != "Offline"
            ].copy()
            worker_options = [None] + [
                int(worker_id) for worker_id in assignment_workers["worker_id"].tolist()
            ]
            worker_labels = {
                None: t("Select worker"),
                **{
                    int(row["worker_id"]): f"{row['name']} ({row['availability_status']})"
                    for _, row in assignment_workers.iterrows()
                },
            }
            worker_label_to_id = {label: worker_id for worker_id, label in worker_labels.items()}

            def _admin_worker_assigned(report_id, widget_key):
                selected_worker = st.session_state.get(widget_key)
                if isinstance(selected_worker, str):
                    selected_worker = worker_label_to_id.get(selected_worker)
                if selected_worker is None:
                    return
                ok, msg = assign_worker(int(report_id), int(selected_worker), actor_id=get_current_admin_id())
                if ok:
                    st.toast(msg)
                    st.rerun()
                else:
                    st.error(t(msg))

            def _admin_workflow_status_changed(report_id, widget_key):
                new_status = str(st.session_state[widget_key])
                ok, msg = change_request_status(int(report_id), new_status, "admin", get_current_admin_id())
                if ok:
                    st.toast(f"Report status updated to {new_status}.")
                else:
                    st.error(t(msg))

            def _severity_changed(report_id, widget_key):
                new_severity = st.session_state[widget_key]
                ok, msg = update_water_body_report_severity(report_id, new_severity, actor_id=get_current_admin_id())
                if not ok:
                    st.error(t(msg))
                    return
                st.toast(f"Report severity updated to {new_severity}.", icon="️" if new_severity == "High" else "")
                if new_severity == "High":
                    count, email_count = notify_high_severity_report(report_id)
                    if count:
                        st.toast(
                            f" {count} resident(s) notified; {email_count} email(s) delivered.",
                            icon="",
                        )

            if filtered.empty:
                st.info(f"{t('No')} {t(selected_filter).lower()} {t('water-body reports found.')}")

            for _, report in filtered.iterrows():
                report_id = int(report["report_id"])
                current_status = str(report.get("workflow_status") or "Submitted")
                row_cols = st.columns(column_weights, gap="small")

                photo_value = report.get("photo_path")
                if pd.notna(photo_value) and str(photo_value).strip() and (BASE_DIR / str(photo_value)).exists():
                    row_cols[0].image(str(BASE_DIR / str(photo_value)), width=68)
                else:
                    row_cols[0].markdown("️")

                colony = str(report["colony_name"])
                society = str(report["society_name"])
                served_area = f"{colony} — {society}" if society.strip() and society.lower() not in ("nan", "none") else colony
                row_cols[1].markdown(f"**{report['water_body']}**  \n{served_area}")
                if pd.notna(report["latitude"]) and pd.notna(report["longitude"]):
                    row_cols[1].caption(f"{float(report['latitude']):.4f}, {float(report['longitude']):.4f}")
                if pd.notna(report.get("worker_name")) and str(report.get("worker_name") or "").strip():
                    row_cols[1].caption(f"{t('Worker')}: {report['worker_name']}")

                row_cols[2].markdown(f"**{t(report['issue_type'])}**  \n{report['description']}")
                row_cols[2].caption(f"{t('Severity:')} {t(report['severity'])}")

                try:
                    report_date = pd.to_datetime(report["created_at"]).strftime("%d/%m/%Y")
                except Exception:  # noqa: BLE001
                    report_date = str(report["created_at"])
                row_cols[3].write(report_date)

                severity_key = f"report_severity_dropdown_{report_id}"
                current_severity = str(report["severity"])
                row_cols[4].selectbox(
                    t("Severity"), severity_options,
                    index=(severity_options.index(current_severity) if current_severity in severity_options else 0),
                    format_func=t, key=severity_key, label_visibility="collapsed",
                    on_change=_severity_changed, args=(report_id, severity_key),
                )

                current_worker_id = report.get("assigned_worker_id")
                try:
                    current_worker_id = int(current_worker_id) if pd.notna(current_worker_id) else None
                except (TypeError, ValueError):
                    current_worker_id = None

                if current_worker_id is not None and current_worker_id not in worker_options:
                    current_worker_name = str(report.get("worker_name") or "Assigned worker")
                    worker_options_for_report = [current_worker_id] + worker_options
                    worker_labels_for_report = {
                        **worker_labels,
                        current_worker_id: current_worker_name,
                    }
                else:
                    worker_options_for_report = worker_options
                    worker_labels_for_report = worker_labels

                worker_key = f"report_worker_dropdown_{report_id}"
                row_cols[5].selectbox(
                    t("Worker"), worker_options_for_report,
                    index=(
                        worker_options_for_report.index(current_worker_id)
                        if current_worker_id in worker_options_for_report
                        else 0
                    ),
                    format_func=lambda value, labels=worker_labels_for_report: labels.get(value, t("Select worker")),
                    key=worker_key, label_visibility="collapsed",
                    on_change=_admin_worker_assigned, args=(report_id, worker_key),
                )

                # Only show valid next workflow states for the current request.
                allowed_next = sorted(STATUS_TRANSITIONS.get(current_status, set()))
                status_options = [current_status] + [x for x in allowed_next if x != current_status]
                widget_key = f"report_workflow_status_dropdown_{report_id}"
                row_cols[6].selectbox(
                    f"{t('Status for Report #')}{report_id}", status_options,
                    index=0, format_func=t, key=widget_key, label_visibility="collapsed",
                    on_change=_admin_workflow_status_changed, args=(report_id, widget_key),
                )
                row_cols[6].caption(f"{t('Report #')}{report_id} · {t(current_status)}")
                st.divider()

            # ---------------- REPORT MAP ----------------
            st.markdown(t("### ️ Report Map"))
            report_map = make_base_map([22.7196, 75.8577], 11)
            status_colors = {
                "Submitted": "red", "Reopened": "red", "Assigned": "blue",
                "In Progress": "orange",
                "Waiting for Verification": "purple", "Resolved": "green",
            }
            for _, report in water_body_reports.iterrows():
                if pd.isna(report["latitude"]) or pd.isna(report["longitude"]):
                    continue
                status = str(report.get("workflow_status") or "Submitted")
                folium.Marker(
                    location=[float(report["latitude"]), float(report["longitude"])],
                    popup=folium.Popup(
                        f"<b>Report #{int(report['report_id'])}</b><br>"
                        f"Water Body: {report['water_body']}<br>"
                        f"Issue: {report['issue_type']}<br>"
                        f"Status: {status}",
                        max_width=320,
                    ),
                    tooltip=f"Report #{int(report['report_id'])} — {status}",
                    icon=folium.Icon(
                        color=status_colors.get(status, "blue"),
                        icon="flag", prefix="fa",
                    ),
                ).add_to(report_map)
            add_map_legend(
                report_map,
                [(t("Submitted / Reopened"), "#dc2626"), (t("Assigned"), "#2563eb"),
                 (t("In Progress"), "#f59e0b"), (t("Verification"), "#7c3aed"),
                 (t("Resolved"), "#16a34a")],
            )
            st_folium(report_map, width=None, height=480, key="water_body_report_admin_map", returned_objects=[])


    # ---------------- WORKERS ----------------
    elif admin_section == "Workers":
        st.subheader(t(" Worker Management"))
        # Worker Management only operates on currently active workers.
        # Removed workers remain in the database for historical integrity,
        # but must not appear as available/editable/removable workers.
        workers = get_workers(active_only=True)
        c1,c2,c3 = st.columns(3)
        c1.metric(t("Available"), int((workers["availability_status"] == "Available").sum()) if not workers.empty else 0)
        c2.metric(t("Busy"), int((workers["availability_status"] == "Busy").sum()) if not workers.empty else 0)
        c3.metric(t("Offline"), int((workers["availability_status"] == "Offline").sum()) if not workers.empty else 0)
        if not workers.empty:
            _display_dataframe(workers[["worker_id","name","username","zone_name","availability_status","active_jobs","active","created_at"]], width="stretch", hide_index=True)
            st.subheader(t("Worker Availability"))
            selected_worker = st.selectbox(t("Worker"), workers["worker_id"].astype(int).tolist(), format_func=lambda x: str(workers.loc[workers["worker_id"] == x, "name"].iloc[0]), key="availability_worker_select")
            current_worker_status = str(workers.loc[workers["worker_id"] == selected_worker, "availability_status"].iloc[0])
            new_worker_status = st.selectbox(t("Availability"), WORKER_STATUSES, index=WORKER_STATUSES.index(current_worker_status) if current_worker_status in WORKER_STATUSES else 0, key="availability_status_select")
            if st.button(t("Update Worker Availability"), key="update_worker_availability"):
                ok, msg = set_worker_availability(selected_worker, new_worker_status, actor_id=get_current_admin_id())
                (st.success if ok else st.error)(t(msg))
                if ok: st.rerun()
        zones = get_zones()
        zone_options=[None]+[int(x) for x in zones["zone_id"].tolist()]
        zone_label=lambda x: t("Any zone") if x is None else str(zones.loc[zones["zone_id"]==x,"zone_name"].iloc[0])

        with st.expander(t("️ Edit Existing Worker")):
            editable_workers = workers[workers["active"] == 1].copy() if not workers.empty else workers
            if editable_workers.empty:
                st.info(t("No active workers available to edit."))
            else:
                edit_id = st.selectbox(
                    t("Worker to edit"), editable_workers["worker_id"].astype(int).tolist(),
                    format_func=lambda x: str(editable_workers.loc[editable_workers["worker_id"] == x, "name"].iloc[0]),
                    key="edit_worker_select",
                )
                edit_row = editable_workers[editable_workers["worker_id"] == edit_id].iloc[0]
                current_zone = None if pd.isna(edit_row["zone_id"]) else int(edit_row["zone_id"])
                with st.form("edit_worker_form"):
                    c1,c2=st.columns(2)
                    ename=c1.text_input(t("Worker name"), value=str(edit_row["name"] or ""))
                    eusername=c2.text_input(t("Username"), value=str(edit_row["username"] or ""))
                    c3,c4=st.columns(2)
                    ephone=c3.text_input(t("Phone"), value=str(edit_row["phone"] or ""))
                    epassword=c4.text_input(t("New password (optional)"), type="password")
                    ezone=st.selectbox(t("Primary zone"), zone_options, index=zone_options.index(current_zone) if current_zone in zone_options else 0, format_func=zone_label)
                    if st.form_submit_button(t("Save Worker Details"), type="primary"):
                        ok,msg=update_worker_account(edit_id,eusername,ename,ephone,ezone,epassword.strip() or None,actor_id=get_current_admin_id())
                        (st.success if ok else st.error)(t(msg))
                        if ok: st.rerun()

        with st.expander(t(" Create Worker Account")):
            with st.form("create_worker_form"):
                c1,c2=st.columns(2)
                wname=c1.text_input(t("Worker name"))
                wusername=c2.text_input(t("Username"))
                c3,c4=st.columns(2)
                wphone=c3.text_input(t("Phone"))
                wpassword=c4.text_input(t("Temporary password"), type="password")
                wzone=st.selectbox(t("Primary zone"), zone_options, format_func=zone_label)
                if st.form_submit_button(t("Create Worker"), type="primary"):
                    ok,msg=create_worker_account(wusername,wpassword,wname,wphone,wzone)
                    (st.success if ok else st.error)(t(msg))
                    if ok: st.rerun()

        with st.expander(t("️ Remove Worker")):
            removable_workers = workers[workers["active"] == 1].copy() if not workers.empty else workers
            if removable_workers.empty:
                st.info(t("No active workers available to remove."))
            else:
                remove_id = st.selectbox(
                    t("Worker to remove"), removable_workers["worker_id"].astype(int).tolist(),
                    format_func=lambda x: str(removable_workers.loc[removable_workers["worker_id"] == x, "name"].iloc[0]),
                    key="remove_worker_select",
                )
                remove_name = str(removable_workers.loc[removable_workers["worker_id"] == remove_id, "name"].iloc[0])
                st.warning(t(f"Remove {remove_name} from the active worker list? Existing completed history will be preserved."))
                if st.button(t("Remove Worker"), type="secondary", key="remove_worker_button"):
                    ok,msg=remove_worker_account(remove_id, actor_id=get_current_admin_id())
                    (st.success if ok else st.error)(t(msg))
                    if ok: st.rerun()

    # ---------------- CITIZEN REVIEWS ----------------
    elif admin_section == "Citizen Reviews":
        st.subheader(t("Citizen Reviews"))
        reviews=get_reviews()
        c1,c2=st.columns(2)
        c1.metric(t("Reviews"),len(reviews))
        c2.metric(t("Average Rating"), f"{reviews['rating'].mean():.2f}/5" if not reviews.empty else "-")
        if reviews.empty:
            st.info(t("No citizen reviews yet."))
        else:
            _display_dataframe(reviews,width="stretch",hide_index=True)
            rating_counts=reviews["rating"].value_counts().sort_index()
            fig=px.bar(x=rating_counts.index,y=rating_counts.values,title=t("Citizen Satisfaction"),labels={"x":"Rating","y":"Reviews"})
            show_chart(fig,320)

    # ---------------- SYSTEM INSIGHTS ----------------
    elif admin_section == "System Insights":
        requests_df=get_service_requests()
        snapshot=get_zone_snapshot()
        reviews=get_reviews()
        st.subheader(t(" Operational Insights"))
        c1,c2,c3,c4=st.columns(4)
        c1.metric(t("Total Requests"),len(requests_df))
        c2.metric(t("High Priority"),int((requests_df["severity"]=="High").sum()) if not requests_df.empty else 0)
        c3.metric(t("High-Risk Zones"),int((snapshot["risk"]=="High Risk").sum()) if not snapshot.empty else 0)
        c4.metric(t("Citizen Rating"),f"{reviews['rating'].mean():.2f}/5" if not reviews.empty else "-")
        if not requests_df.empty:
            issue=requests_df["issue_type"].value_counts().head(10).reset_index()
            issue.columns=["Issue","Requests"]
            fig=px.bar(issue,x="Requests",y="Issue",orientation="h",title=t("Most Reported Problems"))
            show_chart(fig,340)
            _display_dataframe(issue,width="stretch",hide_index=True)
        st.markdown("### " + t("Attention Areas"))
        attention=snapshot[snapshot["risk"].isin(["Moderate Risk","High Risk"])][["zone_name","risk","supply_hours","availability_status"]].copy()
        if attention.empty:
            st.success(t("No high-priority zone conditions detected in the latest project data."))
        else:
            _display_dataframe(attention,width="stretch",hide_index=True)

    # ---------------- ML MODEL PERFORMANCE ----------------
    elif admin_section == "ML Model Performance":
        st.subheader(t(" ML Model Performance"))
        quality=get_water_quality()
        if len(quality) >= 30:
            X=quality[DB_QUALITY_COLS].copy(); y=violation_flags(quality).sum(axis=1).apply(rule_label)
            stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=stratify)
            eval_model=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")
            eval_model.fit(X_train,y_train); pred=eval_model.predict(X_test)
            c1,c2,c3,c4=st.columns(4)
            report=classification_report(y_test,pred,labels=RISK_ORDER,output_dict=True,zero_division=0)
            c1.metric(t("Accuracy"),f"{accuracy_score(y_test,pred)*100:.2f}%")
            c2.metric(t("Precision"),f"{report['macro avg']['precision']*100:.2f}%")
            c3.metric(t("Recall"),f"{report['macro avg']['recall']*100:.2f}%")
            c4.metric(t("F1"),f"{report['macro avg']['f1-score']*100:.2f}%")
            cm=confusion_matrix(y_test,pred,labels=RISK_ORDER)
            cm_df=pd.DataFrame(cm,index=[f"Actual: {x}" for x in RISK_ORDER],columns=[f"Predicted: {x}" for x in RISK_ORDER])
            st.markdown("**"+t("Confusion Matrix")+"**")
            _display_dataframe(cm_df,width="stretch")
            fig=px.imshow(cm_df,text_auto=True,title=t("Confusion Matrix"),labels=dict(x="Predicted",y="Actual",color="Count"))
            show_chart(fig,420)
            importance=pd.DataFrame({"Feature":QUALITY_FEATURES,"Importance":eval_model.feature_importances_}).sort_values("Importance",ascending=False)
            _display_dataframe(importance,width="stretch",hide_index=True)
            fig=px.bar(importance.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title=t("Feature Importance"))
            show_chart(fig,340)
            st.caption(t("This evaluation uses a held-out stratified test split; it is separate from the deployed model file."))
        else:
            st.warning(t("Not enough records for a reliable hold-out evaluation."))

        availability=get_availability()
        if len(availability) >= 30:
            data=availability.sort_values(["zone_name","date"]).copy()
            data["previous_supply_hours"]=data.groupby("zone_name")["supply_hours"].shift(1)
            data["season_code"]=data["season"].map(SEASON_CODES)
            data=data.dropna(subset=["previous_supply_hours","season_code"])
            if len(data)>=20:
                split=max(1,int(len(data)*0.2)); train=data.iloc[:-split]; test=data.iloc[-split:]
                model=LinearRegression().fit(train[["previous_supply_hours","rainfall","season_code"]],train["supply_hours"])
                pred=np.clip(model.predict(test[["previous_supply_hours","rainfall","season_code"]]),0,24)
                c1,c2,c3=st.columns(3)
                c1.metric("MAE",f"{mean_absolute_error(test['supply_hours'],pred):.3f}")
                c2.metric("RMSE",f"{float(np.sqrt(np.mean((test['supply_hours'].to_numpy()-pred)**2))):.3f}")
                c3.metric("R²",f"{r2_score(test['supply_hours'],pred):.3f}")
                st.caption(t("Availability evaluation uses a chronological hold-out to reduce future-data leakage."))

    # ---------------- OVERVIEW ----------------
    elif admin_section == "Overview":
        counts = get_table_counts()
        reports_now = get_water_body_reports()
        metrics = get_service_metrics()
        snapshot = get_zone_snapshot()

        st.subheader(t("Operational Dashboard"))
        cols = st.columns(6)
        cols[0].metric(t("New / Reopened"), metrics["pending"])
        cols[1].metric(t("Assigned"), metrics["assigned"])
        cols[2].metric(t("In Progress"), metrics["in_progress"])
        cols[3].metric(t("Verification"), metrics["verification"])
        cols[4].metric(t("Resolved"), metrics["resolved"])
        cols[5].metric(t("High Priority"), metrics["high_priority"])

        cols = st.columns(6)
        cols[0].metric(t("Project Zones"), counts["zones"])
        cols[1].metric(t("Quality Records"), f"{counts['water_quality']:,}")
        cols[2].metric(t("Availability Records"), f"{counts['availability']:,}")
        cols[3].metric(t("Predictions"), counts["predictions"])
        cols[4].metric(t("Active Workers"), metrics["active_workers"])
        cols[5].metric(t("Satisfaction"), f"{metrics['satisfaction']:.2f}/5" if metrics["satisfaction"] is not None else "-")

        if metrics["avg_resolution_hours"] is not None:
            st.caption(f"{t('Average resolution time')}: {metrics['avg_resolution_hours']:.1f} hours")

        section(" Zone Status")
        _display_dataframe(snapshot_display(snapshot), width="stretch", hide_index=True)

        if not snapshot.empty:
            c1, c2 = st.columns(2)
            with c1:
                risk_counts = snapshot["risk"].value_counts().reindex(RISK_ORDER).fillna(0)
                fig = px.bar(x=risk_counts.index, y=risk_counts.values, title=t("Water Quality Risk Overview"), labels={"x":"Risk", "y":"Zones"})
                show_chart(fig, 300)
            with c2:
                availability_counts = snapshot["availability_status"].value_counts().reindex(STATUS_ORDER).fillna(0)
                fig = px.bar(x=availability_counts.index, y=availability_counts.values, title=t("Water Availability Overview"), labels={"x":"Status", "y":"Zones"})
                show_chart(fig, 300)

        section(" Worker Performance")
        worker_perf = get_worker_performance()
        if worker_perf.empty:
            st.info(t("No worker performance data available yet."))
        else:
            _display_dataframe(worker_perf, width="stretch", hide_index=True)

        section(" Machine Learning Models")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{t('Water Quality Assessment')}**")
            st.write(t("Algorithm: Random Forest Classifier"))
            st.write(t("Inputs: pH, TDS, turbidity, hardness, chloride, fluoride, nitrate"))
            st.write(t("Output: Safe / Moderate Risk / High Risk"))
        with col2:
            st.markdown(f"**{t('Water Availability Forecasting')}**")
            st.write(t("Algorithm: Linear Regression"))
            st.write(t("Inputs: previous supply hours, rainfall, season"))
            st.write(t("Output: forecast supply hours and availability category"))

        if admin_unread and st.button(t("Mark admin notifications as read"), key="admin_mark_all_read"):
            mark_notifications_read("admin", None)
            st.rerun()

    # ---------------- RESIDENT ALERTS ----------------
    elif admin_section == "Resident Alerts":
        st.subheader(t(" Resident Alerts"))

        counts = get_resident_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric(t("Registered Residents"), int(counts["residents"]))
        col2.metric(t("Alert Records"), int(counts["alerts"]))
        col3.metric(t("Email Deliveries"), int(counts["delivered"]))

        st.subheader(t("Registered Residents"))
        residents = db_query(
            """
            SELECT
                residents.resident_id,
                zones.zone_name,
                residents.name,
                residents.phone,
                residents.email,
                residents.created_at
            FROM residents
            JOIN zones ON residents.zone_id = zones.zone_id
            ORDER BY residents.resident_id DESC
            """
        )
        if residents.empty:
            st.info(t("No residents have registered for alerts yet."))
        else:
            _display_dataframe(residents, width="stretch", hide_index=True)

        st.subheader(t(" Alert Log"))
        zone_names = sorted(get_zones()["zone_name"].tolist())
        selected_alert_zone = st.selectbox(
            t("Zone"),
            ["All zones"] + zone_names,
            format_func=t,
            key="admin_alert_zone_filter",
        )

        if selected_alert_zone == "All zones":
            alert_log = get_recent_alerts(200)
        else:
            zone_id = int(
                get_zones()
                .loc[get_zones()["zone_name"] == selected_alert_zone, "zone_id"]
                .iloc[0]
            )
            alert_log = get_recent_alerts(200, zone_id=zone_id)

        if alert_log.empty:
            st.info(t("No alerts have been generated yet."))
        else:
            _display_dataframe(alert_log, width="stretch", hide_index=True)

        st.subheader(t(" Manual Broadcast"))
        zones = get_zones()
        with st.form("manual_broadcast"):
            zone_name = st.selectbox(
                t("Zone"),
                zones["zone_name"].tolist(),
                key="broadcast_zone",
            )
            message = st.text_area(
                t("Alert message"),
                key="broadcast_message",
                placeholder=t("Write the message residents should receive..."),
            )
            send = st.form_submit_button(
                t(" Send Now"),
                type="primary",
                width="stretch",
            )

        if send:
            if not message.strip():
                st.toast(t("Please enter an alert message."), icon="️")
            else:
                zone_id = int(
                    zones.loc[zones["zone_name"] == zone_name, "zone_id"].iloc[0]
                )
                count, email_count = queue_zone_alert(
                    zone_id=zone_id,
                    zone_name=zone_name,
                    message=message.strip(),
                )
                st.toast(
                    f" Alert queued for {count} resident(s) in {zone_name}. "
                    f"{email_count} email(s) delivered; all recipients received an in-app alert record.",
                    icon="",
                )

    # ---------------- DATA IMPORT ----------------
    elif admin_section == "Data Import":
        st.subheader(t(" Upload Water Quality CSV"))
        st.write(
            t("Upload a CSV with water-quality records. The system validates the data and skips records that already exist.")
        )

        if st.session_state.get("import_message"):
            st.toast(st.session_state.pop("import_message"))

        st.caption(t("Required columns: zone, date, pH, TDS, turbidity, hardness, chloride, fluoride, nitrate"))
        uploaded_file = st.file_uploader(t("Choose CSV file"), type=["csv"], key="quality_csv")

        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                st.write(t("### Uploaded Data Preview"))
                _display_dataframe(uploaded_data.head(10), width="stretch", hide_index=True)

                required_columns = ["zone", "date"] + QUALITY_FEATURES
                missing_columns = [c for c in required_columns if c not in uploaded_data.columns]

                if missing_columns:
                    st.toast(t("Missing columns: ") + ", ".join(missing_columns))
                else:
                    validation_errors = []

                    uploaded_data["date"] = pd.to_datetime(uploaded_data["date"], errors="coerce")
                    invalid_dates = int(uploaded_data["date"].isna().sum())
                    if invalid_dates:
                        validation_errors.append(f"{invalid_dates} invalid date value(s) found.")

                    for column in QUALITY_FEATURES:
                        uploaded_data[column] = pd.to_numeric(uploaded_data[column], errors="coerce")
                        invalid_values = int(uploaded_data[column].isna().sum())
                        if invalid_values:
                            validation_errors.append(f"{invalid_values} invalid value(s) in {column}.")

                    if ((uploaded_data["pH"] < 0) | (uploaded_data["pH"] > 14)).any():
                        validation_errors.append("pH values must be between 0 and 14.")
                    for column in QUALITY_FEATURES[1:]:
                        if (uploaded_data[column] < 0).any():
                            validation_errors.append(f"{column} cannot contain negative values.")

                    zones_table = get_zones()
                    valid_zones = set(zones_table["zone_name"].tolist())
                    unknown_zone_count = int((~uploaded_data["zone"].isin(valid_zones)).sum())
                    if unknown_zone_count:
                        validation_errors.append(f"{unknown_zone_count} record(s) contain an unknown zone.")

                    duplicate_count_csv = int(uploaded_data.duplicated(subset=["zone", "date"], keep=False).sum())
                    if duplicate_count_csv:
                        validation_errors.append(f"{duplicate_count_csv} duplicate zone-date record(s) found inside the CSV.")

                    if validation_errors:
                        st.toast(t("CSV validation failed."))
                        for message in validation_errors:
                            st.warning("️ " + message)
                    else:
                        st.toast(t("CSV validation successful."))
                        st.write(f"{t('Records found:')} {len(uploaded_data)}")

                        existing = db_query(
                            """
                            SELECT zones.zone_name AS zone, water_quality.date
                            FROM water_quality
                            JOIN zones ON water_quality.zone_id = zones.zone_id
                            """
                        )
                        existing_keys = set(zip(existing["zone"], existing["date"]))
                        zone_ids = dict(zip(zones_table["zone_name"], zones_table["zone_id"]))

                        new_records = []
                        duplicate_count = 0
                        for _, row in uploaded_data.iterrows():
                            date_value = row["date"].strftime("%Y-%m-%d")
                            if (row["zone"], date_value) in existing_keys:
                                duplicate_count += 1
                            else:
                                new_records.append((
                                    int(zone_ids[row["zone"]]), date_value,
                                    float(row["pH"]), float(row["TDS"]), float(row["turbidity"]),
                                    float(row["hardness"]), float(row["chloride"]),
                                    float(row["fluoride"]), float(row["nitrate"]),
                                ))

                        st.write(t("### Import Summary"))
                        col1, col2 = st.columns(2)
                        col1.metric(t("New Records"), len(new_records))
                        col2.metric(t("Duplicates"), duplicate_count)
                        if duplicate_count:
                            st.toast(f"{duplicate_count} {t('existing record(s) will be skipped.')}", icon="️")

                        if new_records:
                            if st.button(t(" Import New Records"), key="import_new_records"):
                                with closing(get_db_connection()) as connection:
                                    connection.executemany(
                                        """
                                        INSERT INTO water_quality
                                            (zone_id, date, ph, tds, turbidity, hardness,
                                             chloride, fluoride, nitrate)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        new_records,
                                    )
                                    connection.commit()
                                clear_data_caches()
                                st.session_state["import_message"] = (
                                    f"{t('Import completed successfully.')} {len(new_records)} {t('new record(s) added.')}"
                                )
                                st.rerun()
                        else:
                            st.info(t("No new records to import. All uploaded records already exist in the database."))
            except Exception as error:  # noqa: BLE001
                st.toast(f"{t('Unable to process CSV:')} {error}")

    # ---------------- PREDICTIONS & EXPORTS ----------------
    elif admin_section == "Predictions & Exports":
        history = get_prediction_history()
        st.subheader(t(" Prediction History"))
        if history.empty:
            st.info(t("No prediction history available."))
        else:
            admin_history = history.copy()
            admin_history["Type"] = admin_history["risk_level"].apply(
                lambda value: "Water Quality" if pd.notna(value) else "Availability Forecast"
            )
            filter_col1, filter_col2 = st.columns(2)
            zone_filter = filter_col1.selectbox(t("Zone"), ["All zones"] + sorted(admin_history["zone"].unique()), format_func=t, key="hist_zone")
            type_filter = filter_col2.selectbox(t("Type"), ["All types", "Water Quality", "Availability Forecast"], format_func=t, key="hist_type")
            if zone_filter != "All zones":
                admin_history = admin_history[admin_history["zone"] == zone_filter]
            if type_filter != "All types":
                admin_history = admin_history[admin_history["Type"] == type_filter]
            _display_dataframe(admin_history, width="stretch", hide_index=True)
            st.download_button(
                t("⬇️ Download prediction history (CSV)"),
                data=admin_history.to_csv(index=False).encode("utf-8"),
                file_name="aquatrack_prediction_history.csv", mime="text/csv", key="export_history",
            )

        st.subheader(t("️ Dataset Exports"))
        col1, col2 = st.columns(2)
        col1.download_button(
            t("⬇️ Water quality data (CSV)"),
            data=get_water_quality().to_csv(index=False).encode("utf-8"),
            file_name="aquatrack_water_quality.csv", mime="text/csv", key="export_quality",
            width="stretch",
        )
        col2.download_button(
            t("⬇️ Availability data (CSV)"),
            data=get_availability().to_csv(index=False).encode("utf-8"),
            file_name="aquatrack_availability.csv", mime="text/csv", key="export_availability",
            width="stretch",
        )

    # ---------------- SYSTEM ----------------
    else:
        st.subheader(t("️ System Status"))

        # ------------------------------------------------------------
        # REAL EMAIL / SMTP CONFIGURATION
        # ------------------------------------------------------------
        st.markdown("###  Real Email Delivery")
        st.caption(
            "AquaTrack uses SMTP for real email delivery. For Gmail, use a Google App Password "
            "instead of the normal Gmail account password."
        )

        mail_status = smtp_status()
        if mail_status["configured"]:
            st.success(
                f"SMTP ready • {mail_status['host']}:{mail_status['port']} • "
                f"Sender: {mail_status['username']}"
            )
            st.caption(f"Sender name: {mail_status['sender_name']}")
        else:
            st.warning(mail_status["message"])
            st.code(
                '[smtp]\n'
                'host = "smtp.gmail.com"\n'
                'port = 587\n'
                'username = "your-email@gmail.com"\n'
                'password = "YOUR_GOOGLE_APP_PASSWORD"\n'
                'sender_name = "AquaTrack Support"\n'
                'use_tls = true',
                language="toml",
            )

        test_col1, test_col2 = st.columns([3, 1])
        with test_col1:
            test_recipient = st.text_input(
                "Test recipient email",
                placeholder="your-email@gmail.com",
                key="smtp_test_recipient",
            )
        with test_col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            test_email_button = st.button(
                " Send Test Email",
                key="send_smtp_test_email",
                type="primary",
                width="stretch",
            )

        if test_email_button:
            if not test_recipient.strip():
                st.error("Enter a recipient email address first.")
            else:
                with st.spinner("Sending test email..."):
                    sent, result = send_test_email(test_recipient.strip())
                if sent:
                    st.success(f"Test email sent successfully to {test_recipient.strip()}.")
                else:
                    st.error(result)

        st.divider()

        st.subheader(t("️ System Status"))
        checks = [
            (t("SQLite database file"), Path(DATABASE_NAME).exists()),
            (t("Water-quality model file"), QUALITY_MODEL_PATH.exists()),
            (t("Availability model file"), AVAILABILITY_MODEL_PATH.exists()),
            (t("Report photo folder"), (BASE_DIR / "reports").exists()),
            ("Resident alerts tables", True),
            ("SMTP email configuration", mail_status["configured"]),
        ]
        for label, ok in checks:
            (st.success if ok else st.error)(f"{label}: {t('Available') if ok else t('Missing')}")

        try:
            counts = get_table_counts()
            st.success(
                f"{t('Database connection: OK (')}{counts['water_quality']:,} {t('quality and')} "
                f"{counts['availability']:,} {t('availability records)')}"
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"{t('Database connection failed:')} {error}")

        if st.button(t(" Refresh cached data"), key="refresh_cache"):
            clear_data_caches()
            st.toast(t("Cached data refreshed successfully."))
        st.caption(t("Data is cached for speed. Use this after editing the database outside the app."))
        st.divider()
        st.subheader(t("️ Database Backup"))
        backup = latest_backup()
        if backup and backup.exists():
            st.success(f"{t('Latest backup')}: {backup.name}")
        else:
            st.warning(t("No database backup has been created yet."))
        if st.button(t("Create Database Backup"), key="create_db_backup", type="primary"):
            backup_path, backup_message = create_database_backup()
            if backup_path:
                st.success(backup_message)
                st.download_button(
                    t("⬇️ Download Latest Backup"),
                    data=backup_path.read_bytes(),
                    file_name=backup_path.name,
                    mime="application/octet-stream",
                    key="download_latest_backup",
                )
            else:
                st.error(backup_message)

        st.divider()
        st.subheader(t(" Audit Log"))
        audit = db_query("SELECT audit_id, action, actor_role, actor_id, report_id, details, created_at FROM aquatrack_audit_log ORDER BY audit_id DESC LIMIT 100")
        if audit.empty:
            st.info(t("No audit events recorded yet."))
        else:
            _display_dataframe(audit, width="stretch", hide_index=True)

        st.info(t(DISCLAIMER))


st.markdown(
    """
    <style>
    /* Cross-device safety overrides. These rules are intentionally loaded
       after the page-specific viewport CSS so phone/tablet behavior wins. */

    /* Phones: never lock these pages to the viewport — let them scroll
       normally so the on-screen keyboard never fights position:fixed. */
    @media (max-width: 820px) {
        html:has(.admin-login-marker),
        html:has(.admin-login-marker) body,
        html:has(.forecast-footer-space-marker),
        html:has(.forecast-footer-space-marker) body,
        html:has(.resident-alert-marker),
        html:has(.resident-alert-marker) body {
            overflow: auto !important;
            height: auto !important;
            max-height: none !important;
        }
        body:has(.admin-login-marker) section[data-testid="stMain"] > div,
        body:has(.admin-login-marker) .main .block-container,
        body:has(.admin-login-marker) div[data-testid="stMainBlockContainer"],
        body:has(.forecast-footer-space-marker) section[data-testid="stMain"] > div,
        body:has(.forecast-footer-space-marker) .main .block-container,
        body:has(.forecast-footer-space-marker) div[data-testid="stMainBlockContainer"],
        body:has(.resident-alert-marker) section[data-testid="stMain"] > div,
        body:has(.resident-alert-marker) .main .block-container,
        body:has(.resident-alert-marker) div[data-testid="stMainBlockContainer"] {
            height: auto !important;
            min-height: 100vh !important;
            max-height: none !important;
            overflow: visible !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]),
        /* Compact desktop login cards: keep the complete login group comfortably below the navigation. */
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] {
            padding: .82rem 1rem !important;
            border-radius: 16px !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] > div {
            gap: .35rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] h2 {
            font-size: 1.75rem !important;
            margin: .35rem 0 .25rem !important;
            line-height: 1.08 !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] p {
            line-height: 1.38 !important;
            margin-bottom: .2rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] input {
            min-height: 2.35rem !important;
            background: rgba(255,255,255,.94) !important;
            color: #26384b !important;
            border: 1px solid #d9e2eb !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            box-sizing: border-box !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] button {
            min-height: 2.35rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) + div[data-testid="stHorizontalBlock"] {
            position: static !important;
            transform: none !important;
            width: 100% !important;
            margin: 1rem 0 !important;
        }
        body:has(.admin-login-marker) .aqua-footer,
        body:has(.resident-alert-marker) .aqua-footer {
            position: static !important;
            transform: none !important;
            width: 100% !important;
        }
    }

    /* Resident Alerts: use ONE page scrollbar only.
       Streamlit can otherwise create a nested scrollbar inside stMain while
       the browser creates another scrollbar for the document. */
    body:has(.resident-alert-marker) main,
    body:has(.resident-alert-marker) section[data-testid="stMain"],
    body:has(.resident-alert-marker) .stApp,
    body:has(.resident-alert-marker) .stAppViewContainer {
        overflow: visible !important;
        height: auto !important;
        max-height: none !important;
    }
    body:has(.resident-alert-marker) html {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    /* Tablets: prevent the seven-column header from squeezing button labels
       on iPad/Android-tablet portrait widths. */
    @media (min-width: 601px) and (max-width: 900px) {
        div[data-testid="stHorizontalBlock"]:has(.aqua-reference-brand) {
            display:flex !important;
            flex-wrap:wrap !important;
            gap:.5rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-reference-brand) > div:first-child {
            flex:0 0 100% !important;
            width:100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.aqua-reference-brand) > div:not(:first-child) {
            flex:1 1 calc(20% - .5rem) !important;
            min-width:90px !important;
        }
        div[data-testid="stHorizontalBlock"]:not(:has(.aqua-reference-brand)):has(.stButton) {
            flex-wrap:wrap !important;
        }
    }
        /* Worker dashboard: Start Work uses white, bold action text. */
        div[class*="st-key-worker_start_work_"] button p,
        div[class*="st-key-worker_start_work_"] button span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        /* Worker dashboard: Mark as Done / Solved uses white, bold action text. */
        div[class*="st-key-complete_"] button p,
        div[class*="st-key-complete_"] button span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        /* Requested action/download buttons: white, bold text only. */
        div.st-key-export_reports_csv button p,
        div.st-key-export_reports_csv button span,
        div.st-key-update_worker_availability button p,
        div.st-key-update_worker_availability button span,
        div.st-key-refresh_cache button p,
        div.st-key-refresh_cache button span,
        div.st-key-create_db_backup button p,
        div.st-key-create_db_backup button span,
        div.st-key-export_history button p,
        div.st-key-export_history button span,
        div.st-key-export_quality button p,
        div.st-key-export_quality button span,
        div.st-key-export_availability button p,
        div.st-key-export_availability button span,
        .stFormSubmitButton > button p,
        .stFormSubmitButton > button span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        /* Service rating: show only the five star choices, without radio circles. */
        div[class*="st-key-rating_"] [data-baseweb="radio"] {
            display: none !important;
        }
        div[class*="st-key-rating_"] [role="radiogroup"] {
            gap: 6px !important;
        }
        div[class*="st-key-rating_"] label {
            color: #ff9900 !important;
            font-size: 2rem !important;
            line-height: 1 !important;
            padding: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================

# Desktop login spacing: keep the login card and Back-to-Public button
# comfortably above the fixed footer. Mobile keeps its normal document flow.
st.markdown(
    """
    <style>
    @media (min-width: 821px) {
        /* Final desktop login sizing: smaller card, smaller controls, clear space
           below the navigation and above the footer. */
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) {
            top: calc(50% - 12px) !important;
            width: min(760px, calc(100vw - 2rem)) !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] {
            padding: .72rem .95rem !important;
            border-radius: 16px !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] > div {
            gap: .22rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] h2 {
            font-size: 1.65rem !important;
            margin: .25rem 0 .18rem !important;
            line-height: 1.05 !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] p {
            font-size: .9rem !important;
            line-height: 1.32 !important;
            margin-bottom: .05rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] [data-testid="stTextInput"] {
            margin-bottom: .15rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] input {
            min-height: 2.2rem !important;
            height: 2.45rem !important;
            background: rgba(255,255,255,.94) !important;
            color: #26384b !important;
            border: 1px solid #d9e2eb !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            padding: .55rem .8rem !important;
            box-sizing: border-box !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] input:focus {
            border-color: #14a9c7 !important;
            box-shadow: 0 0 0 2px rgba(20,169,199,.12) !important;
            outline: none !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] input::placeholder {
            color: #91a3b7 !important;
            opacity: 1 !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] label {
            font-size: .82rem !important;
            margin-bottom: .1rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] button {
            min-height: 2.25rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] [data-testid="stFormSubmitButton"] {
            margin-top: .25rem !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) + div[data-testid="stHorizontalBlock"] {
            top: calc(50% + 180px) !important;
            width: min(760px, calc(100vw - 2rem)) !important;
        }
    }
    /* Keep only the requested action buttons white and bold. */
    div.st-key-simple_water_check button,
    div.st-key-simple_water_check button p,
    div.st-key-simple_water_check button span,
    div.st-key-simple_water_check button div,
    div.st-key-citizen_mark_all_read button,
    div.st-key-citizen_mark_all_read button p,
    div.st-key-citizen_mark_all_read button span,
    div.st-key-citizen_mark_all_read button div {
        color:#fff !important;
        font-weight:700 !important;
    }

    

/* Amazon-style service rating stars.
   IMPORTANT: this CSS is scoped ONLY to the Service Rating widget.
*/
body:has(.service-rating-marker)
div[data-testid="stRadio"] > label {
    font-size: 0 !important;
}

body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"] {
    gap: 0.18rem !important;
}

body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"] label {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    display: inline-flex !important;
    align-items: center !important;
}

/* REMOVE ONLY THE RADIO CIRCLE FROM SERVICE RATING */
body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label > div:first-child,

body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label > span:first-child,

body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label [data-baseweb="radio"],

body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label [data-baseweb="radio"] *,

body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label input[type="radio"],

body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label svg {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* THE FIVE STARS */
body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    font-size: 2.15rem !important;
    line-height: 1 !important;
    color: transparent !important;
    -webkit-text-stroke: 1.5px #ff9900 !important;
    cursor: pointer !important;
    transition: color 0.12s ease, transform 0.12s ease !important;
}

/* SELECTED STAR */
body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label:has(input:checked)
div[data-testid="stMarkdownContainer"] p {
    color: #ff9900 !important;
    -webkit-text-stroke: 1.5px #ff9900 !important;
    transform: scale(1.03);
}

/* HOVER */
body:has(.service-rating-marker)
div[data-testid="stRadio"]
div[role="radiogroup"]
label:hover
div[data-testid="stMarkdownContainer"] p {
    color: #ff9900 !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# FOOTER
# ============================================================

st.markdown(
    """
    <div class="aqua-footer">
        <b>AquaTrack</b> · Predictive Water Quality &amp; Availability Analytics
        <br>
        <span>Project-level decision-support prototype · Indore · For screening and decision support</span>
        <br><span>Privacy · Terms · Data &amp; Model Transparency · Support</span>
    </div>
    """,
    unsafe_allow_html=True,
)