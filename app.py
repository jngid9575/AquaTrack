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
import sqlite3
from contextlib import closing
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
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

def db_query(sql, params=()):
    with closing(sqlite3.connect(DATABASE_NAME)) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def db_execute(sql, params=()):
    with closing(sqlite3.connect(DATABASE_NAME)) as connection:
        cursor = connection.execute(sql, params)
        connection.commit()
        return cursor.lastrowid


# ============================================================
# RESIDENT REGISTRATION + ZONE ALERT ENGINE
# ============================================================

def ensure_alert_tables():
    """Create/migrate resident-alert tables without changing existing AquaTrack tables."""
    with closing(sqlite3.connect(DATABASE_NAME)) as connection:
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
        SELECT resident_id, zone_id, name, phone, email, created_at
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


def get_smtp_config():
    """Read and validate AquaTrack SMTP settings from Streamlit secrets.

    Expected secrets.toml section:
        [smtp]
        host = "smtp.gmail.com"
        port = 587
        username = "digitalarpit78@gmail.com"
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
    sender_email = cfg["username"] if cfg else "digitalarpit78@gmail.com"
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
            sender_email = smtp_cfg["username"] if smtp_cfg else "digitalarpit78@gmail.com"
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


ensure_alert_tables()


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
    """Keep synthetic/project demo quality data mostly positive and realistic.

    The project is a prototype using synthetic data. For presentation, most
    records are kept comfortably within BIS screening limits while a small,
    deterministic minority remains moderate/high-risk so the risk analytics
    are still demonstrable. Negative numeric values are never shown.
    """
    if df is None or df.empty:
        return df

    data = df.copy()

    numeric_cols = [
        "ph", "tds", "turbidity", "hardness",
        "chloride", "fluoride", "nitrate"
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=0)

    # Deterministic row ordering makes the presentation consistent on every run.
    order = data.sort_values(
        [c for c in ["zone_name", "date"] if c in data.columns]
    ).index
    n = len(order)

    # About 85% safe, 12% moderate, 3% high-risk.
    safe_end = max(0, int(n * 0.85))
    moderate_end = max(safe_end, int(n * 0.97))

    safe_idx = order[:safe_end]
    moderate_idx = order[safe_end:moderate_end]
    high_idx = order[moderate_end:]

    # Safe presentation range.
    if len(safe_idx):
        data.loc[safe_idx, "ph"] = data.loc[safe_idx, "ph"].clip(6.8, 8.0)
        data.loc[safe_idx, "tds"] = data.loc[safe_idx, "tds"].clip(180, 430)
        data.loc[safe_idx, "turbidity"] = data.loc[safe_idx, "turbidity"].clip(0.15, 0.85)
        data.loc[safe_idx, "hardness"] = data.loc[safe_idx, "hardness"].clip(90, 185)
        data.loc[safe_idx, "chloride"] = data.loc[safe_idx, "chloride"].clip(70, 220)
        data.loc[safe_idx, "fluoride"] = data.loc[safe_idx, "fluoride"].clip(0.25, 0.85)
        data.loc[safe_idx, "nitrate"] = data.loc[safe_idx, "nitrate"].clip(5, 38)

    # Small moderate-risk group: one or two parameters slightly outside limits.
    if len(moderate_idx):
        data.loc[moderate_idx, "ph"] = data.loc[moderate_idx, "ph"].clip(6.4, 8.7)
        data.loc[moderate_idx, "tds"] = data.loc[moderate_idx, "tds"].clip(250, 560)
        data.loc[moderate_idx, "turbidity"] = data.loc[moderate_idx, "turbidity"].clip(0.3, 1.25)
        data.loc[moderate_idx, "hardness"] = data.loc[moderate_idx, "hardness"].clip(120, 235)
        data.loc[moderate_idx, "chloride"] = data.loc[moderate_idx, "chloride"].clip(100, 280)
        data.loc[moderate_idx, "fluoride"] = data.loc[moderate_idx, "fluoride"].clip(0.4, 1.15)
        data.loc[moderate_idx, "nitrate"] = data.loc[moderate_idx, "nitrate"].clip(12, 52)

    # Very small high-risk group for demonstrating alerts/risk charts.
    if len(high_idx):
        data.loc[high_idx, "ph"] = data.loc[high_idx, "ph"].clip(6.0, 9.0)
        data.loc[high_idx, "tds"] = data.loc[high_idx, "tds"].clip(420, 700)
        data.loc[high_idx, "turbidity"] = data.loc[high_idx, "turbidity"].clip(0.7, 2.0)
        data.loc[high_idx, "hardness"] = data.loc[high_idx, "hardness"].clip(180, 300)
        data.loc[high_idx, "chloride"] = data.loc[high_idx, "chloride"].clip(220, 350)
        data.loc[high_idx, "fluoride"] = data.loc[high_idx, "fluoride"].clip(0.8, 1.5)
        data.loc[high_idx, "nitrate"] = data.loc[high_idx, "nitrate"].clip(35, 65)

    return data


def prepare_availability_demo_data(df):
    """Keep synthetic availability data positive with mostly good supply."""
    if df is None or df.empty:
        return df

    data = df.copy()

    if "supply_hours" in data.columns:
        data["supply_hours"] = pd.to_numeric(
            data["supply_hours"], errors="coerce"
        ).clip(lower=0, upper=24)

        order = data.sort_values(
            [c for c in ["zone_name", "date"] if c in data.columns]
        ).index
        n = len(order)

        # Mostly good availability, with a small realistic lower-supply group.
        good_end = max(0, int(n * 0.88))
        moderate_end = max(good_end, int(n * 0.97))

        good_idx = order[:good_end]
        moderate_idx = order[good_end:moderate_end]
        low_idx = order[moderate_end:]

        if len(good_idx):
            data.loc[good_idx, "supply_hours"] = data.loc[
                good_idx, "supply_hours"
            ].clip(8.0, 18.5)

        if len(moderate_idx):
            data.loc[moderate_idx, "supply_hours"] = data.loc[
                moderate_idx, "supply_hours"
            ].clip(5.5, 7.9)

        if len(low_idx):
            data.loc[low_idx, "supply_hours"] = data.loc[
                low_idx, "supply_hours"
            ].clip(3.5, 5.0)

    if "rainfall" in data.columns:
        data["rainfall"] = pd.to_numeric(
            data["rainfall"], errors="coerce"
        ).clip(lower=0, upper=300)

    return data


# ============================================================
# CACHED DATA LAYER
# ============================================================

@st.cache_data(ttl=None, show_spinner=False)
def get_zones():
    return db_get_zones()


@st.cache_data(ttl=None, show_spinner=False)
def get_water_quality(zone_name=None):
    data = db_get_water_quality() if zone_name is None else db_get_water_quality(zone_name)
    return prepare_quality_demo_data(data)


@st.cache_data(ttl=None, show_spinner=False)
def get_availability(zone_name=None):
    data = db_get_availability() if zone_name is None else db_get_availability(zone_name)
    return prepare_availability_demo_data(data)


@st.cache_data(ttl=None, show_spinner=False)
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


@st.cache_data(ttl=None, show_spinner=False)
def get_table_counts():
    counts = {}
    for name in [
        "zones", "water_quality", "availability", "predictions",
        "water_bodies", "residents", "alerts_log"
    ]:
        counts[name] = int(db_query(f"SELECT COUNT(*) AS c FROM {name}")["c"].iloc[0])
    return counts


@st.cache_data(ttl=None, show_spinner=False)
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


@st.cache_data(ttl=None, show_spinner=False)
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
            water_body_reports.severity,
            water_body_reports.created_at,
            water_body_reports.updated_at,
            users.name AS reporter_name
        FROM water_body_reports
        LEFT JOIN water_bodies
            ON water_body_reports.water_body_id = water_bodies.water_body_id
        LEFT JOIN users
            ON water_body_reports.reporter_user_id = users.user_id
        ORDER BY water_body_reports.report_id DESC
        """
    )


def update_water_body_report_status(report_id, new_status):
    db_execute(
        """
        UPDATE water_body_reports
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE report_id = ?
        """,
        (new_status, int(report_id)),
    )


def update_water_body_report_severity(report_id, new_severity):
    db_execute(
        """
        UPDATE water_body_reports
        SET severity = ?, updated_at = CURRENT_TIMESTAMP
        WHERE report_id = ?
        """,
        (new_severity, int(report_id)),
    )


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


@st.cache_data(ttl=None, show_spinner=False)
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


@st.cache_data(ttl=None, show_spinner=False)
def get_risk_distribution():
    quality = get_water_quality()
    if quality.empty:
        return pd.DataFrame(columns=["Risk Level", "Records"])
    counts = pd.Series(predict_quality_batch(quality)).value_counts()
    counts = counts.reindex(RISK_ORDER).fillna(0).astype(int)
    return counts.rename_axis("Risk Level").reset_index(name="Records")


@st.cache_data(ttl=None, show_spinner=False)
def get_seasonal_supply():
    availability = get_availability()
    if availability.empty:
        return pd.DataFrame(columns=["season", "supply_hours"])
    return availability.groupby("season", as_index=False)["supply_hours"].mean()


@st.cache_data(ttl=None, show_spinner=False)
def get_model_insights():
    insights = {}

    quality = get_water_quality()
    if not quality.empty:
        predicted = pd.Series(predict_quality_batch(quality), index=quality.index)
        rule = violation_flags(quality).sum(axis=1).apply(rule_label)
        matrix = confusion_matrix(rule, predicted, labels=RISK_ORDER)
        report = classification_report(
            rule, predicted, labels=RISK_ORDER, output_dict=True, zero_division=0
        )
        importances = pd.DataFrame({
            "Parameter": QUALITY_FEATURES,
            "Importance": load_quality_model().feature_importances_,
        }).sort_values("Importance", ascending=True)
        insights["quality"] = {
            "records": len(quality),
            "accuracy": accuracy_score(rule, predicted),
            "matrix": pd.DataFrame(
                matrix,
                index=[f"Rule: {c}" for c in RISK_ORDER],
                columns=[f"Model: {c}" for c in RISK_ORDER],
            ),
            "report": pd.DataFrame(report).T.loc[
                RISK_ORDER + ["macro avg"], ["precision", "recall", "f1-score", "support"]
            ],
            "importances": importances,
        }

    availability = get_availability()
    if not availability.empty:
        data = availability.sort_values(["zone_name", "date"]).copy()
        data["previous_supply_hours"] = data.groupby("zone_name")["supply_hours"].shift(1)
        data["season_code"] = data["season"].map(SEASON_CODES)
        data = data.dropna(subset=["previous_supply_hours", "season_code"])
        if not data.empty:
            features = data[["previous_supply_hours", "rainfall", "season_code"]]
            predicted = np.clip(load_availability_model().predict(features), 0, 24)
            actual = data["supply_hours"].to_numpy()
            naive_mae = float(np.mean(np.abs(actual - data["previous_supply_hours"].to_numpy())))
            model = load_availability_model()
            coefficients = pd.DataFrame({
                "Feature": ["Previous supply hours", "Rainfall", "Season code"],
                "Coefficient": model.coef_,
            })
            scatter = pd.DataFrame({"Actual": actual, "Predicted": predicted})
            insights["availability"] = {
                "records": len(data),
                "mae": float(mean_absolute_error(actual, predicted)),
                "r2": float(r2_score(actual, predicted)),
                "naive_mae": naive_mae,
                "coefficients": coefficients,
                "intercept": float(model.intercept_),
                "scatter": scatter.sample(min(600, len(scatter)), random_state=42),
            }
    return insights


def get_system_health():
    """Return lightweight dashboard health information without changing project data."""
    quality = get_water_quality()
    availability = get_availability()
    quality_date = None
    availability_date = None
    if not quality.empty and "date" in quality.columns:
        quality_date = pd.to_datetime(quality["date"], errors="coerce").max()
    if not availability.empty and "date" in availability.columns:
        availability_date = pd.to_datetime(availability["date"], errors="coerce").max()
    return {
        "models_ready": QUALITY_MODEL_PATH.exists() and AVAILABILITY_MODEL_PATH.exists(),
        "quality_date": quality_date,
        "availability_date": availability_date,
    }


def clear_data_caches():
    st.cache_data.clear()


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
        get_prediction_history.clear()
        get_table_counts.clear()
        return True, ""
    except Exception as error:  # noqa: BLE001
        return False, str(error)


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

st.set_page_config(page_title="AquaTrack", page_icon="💧", layout="wide")

st.markdown(
    """
    <style>
    /* ============================================================
       AQUATRACK GOVERNMENT-STYLE TYPOGRAPHY
       One consistent Noto Sans family across the entire website.
       Noto Sans Devanagari keeps Hindi visually consistent too.
       ============================================================ */
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

    .hero-card { position:relative; overflow:hidden; min-height:424px; padding:3.35rem 3.2rem; border-radius:28px; background:linear-gradient(125deg,#083c69 0%,#0877bb 53%,#16b8c7 100%); box-shadow:0 20px 46px rgba(12,84,120,.18); color:white; margin-bottom:1.75rem; box-sizing:border-box; }
    .hero-card:before { content:""; position:absolute; width:360px; height:360px; right:-95px; top:-145px; border-radius:50%; background:rgba(255,255,255,.075); }
    .hero-card:after { content:""; position:absolute; width:210px; height:210px; right:65px; top:65px; border-radius:50%; background:rgba(255,255,255,.035); }
    .hero-badge { display:inline-flex; align-items:center; gap:.38rem; padding:.43rem .78rem; border-radius:999px; background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.18); font-size:.72rem; font-weight:700; color:#f0fcff; }
    .hero-title { margin-top:1.35rem; max-width:590px; font-size:3.55rem; line-height:1; letter-spacing:-.065em; font-weight:800; color:white; }
    .hero-title span { color:#aeeeff; }
    .hero-text { max-width:570px; margin-top:1.15rem; color:rgba(255,255,255,.83); line-height:1.65; font-size:.93rem; }
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
        border-radius:99px; background:linear-gradient(90deg,#11abc8,#36c9d0);
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
    "🏠 Home": "🏠 होम", "📝 Report": "📝 रिपोर्ट", "🗺️ Map": "🗺️ मैप", "⚙️ Admin": "⚙️ एडमिन", "Logout": "लॉगआउट",
    "📊 Analytics": "📊 एनालिटिक्स", "⚖️ Compare": "⚖️ तुलना", "💧 Quality": "💧 क्वालिटी", "📈 Forecast": "📈 फोरकास्ट", "🧠 Insights": "🧠 इनसाइट्स",
    "📊 Zone-wise Analytics": "📊 ज़ोन-वाइज़ एनालिटिक्स", "⚖️ Compare Zones": "⚖️ ज़ोन कंपेयर करें", "🗺️ Indore Zone Map": "🗺️ इंदौर ज़ोन मैप",
    "💧 Water Quality Risk Prediction": "💧 पानी की क्वालिटी और रिस्क की जाँच", "📈 Water Availability Forecast": "📈 वाटर अवेलेबिलिटी फोरकास्ट",
    "🧠 Model Insights": "🧠 मॉडल इनसाइट्स", "🚨 Report Water Body Issue": "🚨 वाटर बॉडी की प्रॉब्लम रिपोर्ट करें",
    "👨‍💻 AquaTrack Admin Panel": "👨‍💻 AquaTrack एडमिन पैनल",
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
    "Read these numbers carefully. Risk labels come from the BIS violation count, and the same records were used to train the models, so they show agreement with the screening rule and not real-world accuracy. Validation with authorised, laboratory-verified data is the next step.": "इन नंबर्स को ध्यान से देखें। रिस्क लेबल BIS वायलेशन काउंट से आते हैं और इन्हीं रिकॉर्ड्स से मॉडल को ट्रेन किया गया है, इसलिए ये सिर्फ स्क्रीनिंग रूल से मैच दिखाते हैं, रियल-वर्ल्ड एक्यूरेसी नहीं। अगला स्टेप है ऑथराइज़्ड, लैब-वेरिफाइड डेटा से वैलिडेशन करना।",
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
    "Allow location access if you want to report from your current device location.": "अगर आप अपनी करंट लोकेशन से रिपोर्ट करना चाहते हैं तो लोकेशन एक्सेस दें।",
    "🔎 Get Current Address": "🔎 करंट एड्रेस पता करें",
    "Getting location details...": "लोकेशन डिटेल्स मिल रही हैं...",
    "Select Report Location": "रिपोर्ट की लोकेशन चुनें",
    "Selected Water Body Location": "सिलेक्टेड वाटर बॉडी की लोकेशन",
    "My Current Location": "मेरी करंट लोकेशन",
    "Your detected current location": "आपकी करंट लोकेशन",
    "Selected water body location": "सिलेक्टेड वाटर बॉडी की लोकेशन",
    "🚨 Submit Report": "🚨 रिपोर्ट सबमिट करें",
    "Please enter a description of the issue.": "कृपया प्रॉब्लम का डिस्क्रिप्शन लिखें।",
    "The photo is larger than 5 MB. Please choose a smaller image.": "फोटो 5 MB से बड़ी है। कृपया छोटी फोटो चुनें।",
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
    """Return the selected-language version of UI text.

    Uses colloquial Hinglish Hindi (mixed with common English/technical words),
    not shuddh/formal Sanskritized Hindi.
    """
    if st.session_state.get("language", "English") != "हिंदी" or text is None:
        return text
    value = str(text)
    exact = TRANSLATIONS["हिंदी"].get(value)
    if exact is not None:
        return exact
    for en, hi in sorted(TRANSLATIONS["हिंदी"].items(), key=lambda item: len(item[0]), reverse=True):
        if en and en in value:
            value = value.replace(en, hi)
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


_original_dataframe = st.dataframe
_original_metric = st.metric


def _dataframe_bilingual(data, *args, **kwargs):
    return _original_dataframe(_localize_dataframe(data), *args, **kwargs)


def _metric_bilingual(label, value=None, *args, **kwargs):
    return _original_metric(t(label), t(value) if isinstance(value, str) else value, *args, **kwargs)


st.dataframe = _dataframe_bilingual
st.metric = _metric_bilingual


# ============================================================
# SESSION STATE AND NAVIGATION CALLBACKS
# ============================================================

st.session_state.setdefault("admin_logged_in", False)
st.session_state.setdefault("admin_user", None)
st.session_state.setdefault("page", "Dashboard")
st.session_state.setdefault("admin_login_error", None)
st.session_state.setdefault("language", "English")
st.session_state.setdefault("mobile_menu_open", False)


def navigate_to(page_name):
    st.session_state.page = page_name


def open_admin():
    st.session_state.page = "Admin Panel" if st.session_state.admin_logged_in else "Admin Login"


def do_logout():
    if st.session_state.admin_logged_in:
        st.session_state.admin_logged_in = False
        st.session_state.admin_user = None
        st.session_state.page = "Dashboard"


def toggle_language():
    current = st.session_state.get("language", "English")
    st.session_state.language = "हिंदी" if current == "English" else "English"


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


def mobile_logout():
    do_logout()
    st.session_state.mobile_menu_open = False


def mobile_toggle_language():
    toggle_language()
    st.session_state.mobile_menu_open = False


def attempt_admin_login():
    user = authenticate_user(
        st.session_state.get("admin_username", ""),
        st.session_state.get("admin_password", ""),
    )
    if user and user[3] == "admin":
        st.session_state.admin_logged_in = True
        st.session_state.admin_user = user
        st.session_state.admin_login_error = None
        st.session_state.page = "Admin Panel"
    elif user:
        st.session_state.admin_login_error = "This account does not have administrator privileges."
    else:
        st.session_state.admin_login_error = "Invalid admin username or password."


if st.session_state.page == "Admin Panel" and not st.session_state.admin_logged_in:
    st.session_state.page = "Admin Login"

page = st.session_state.page


# ============================================================
# HEADER NAVIGATION
# ============================================================

# Mobile navigation: kebab button on the left, centered brand, language on the right.
# Desktop navigation below remains unchanged.
mobile_header_cols = st.columns([1.0, 4.0, 1.0], gap="small", vertical_alignment="center")

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
        mobile_items = [
            ("🏠 " + t("Home"), "Dashboard", "mobile_home"),
            ("📝 " + t("Report"), "Report Water Body Issue", "mobile_report"),
            ("🔔 " + t("Alerts"), "Resident Alerts", "mobile_alerts"),
            ("🗺️ " + t("Map"), "Indore Zone Map", "mobile_map"),
            ("⚙️ " + t("Admin"), "__ADMIN__", "mobile_admin"),
            ("📊 " + t("Analytics"), "Zone Analytics", "mobile_analytics"),
            ("⚖️ " + t("Compare"), "Compare Zones", "mobile_compare"),
            ("💧 " + t("Quality"), "Water Quality Prediction", "mobile_quality"),
            ("📈 " + t("Forecast"), "Availability Forecast", "mobile_forecast"),
            ("🧠 " + t("Insights"), "Model Insights", "mobile_insights"),
        ]
        for label, target, key in mobile_items:
            if target == "__ADMIN__":
                st.button(label, key=key, width="stretch", on_click=mobile_open_admin,
                          type="primary" if page in ("Admin Panel", "Admin Login") else "secondary")
            else:
                st.button(label, key=key, width="stretch", on_click=mobile_navigate, args=(target,),
                          type="primary" if page == target else "secondary")
        if st.session_state.admin_logged_in:
            st.button(t("Logout"), key="mobile_logout", width="stretch", on_click=mobile_logout, type="secondary")

# Desktop header
if st.session_state.admin_logged_in:
    header_cols = st.columns(
        [5.55, 1.25, 1.04, 1.04, 1.04, 1.04, 1.04],
        gap="small",
        vertical_alignment="center",
    )
else:
    header_cols = st.columns(
        [3.82, 1.22, 1.08, 1.08, 1.08, 1.08, 1.08],
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
header_cols[1].button(language_button_label, key="header_language_toggle", width="stretch", on_click=toggle_language, type="secondary")
header_cols[2].button("🏠 " + t("Home"), key="header_home", width="stretch", on_click=navigate_to, args=("Dashboard",), type="primary" if page == "Dashboard" else "secondary")

if not st.session_state.admin_logged_in:
    header_cols[3].button("📝 " + t("Report"), key="header_report", width="stretch", on_click=navigate_to, args=("Report Water Body Issue",), type="primary" if page == "Report Water Body Issue" else "secondary")
    header_cols[4].button("🔔 " + t("Alerts"), key="header_resident_alerts", width="stretch", on_click=navigate_to, args=("Resident Alerts",), type="primary" if page == "Resident Alerts" else "secondary")
    map_col, admin_col, logout_col = 5, 6, 7
else:
    header_cols[3].button("🔔 " + t("Alerts"), key="header_resident_alerts_admin", width="stretch", on_click=navigate_to, args=("Resident Alerts",), type="primary" if page == "Resident Alerts" else "secondary")
    map_col, admin_col, logout_col = 4, 5, 6

header_cols[map_col].button("🗺️ " + t("Map"), key="header_map", width="stretch", on_click=navigate_to, args=("Indore Zone Map",), type="primary" if page == "Indore Zone Map" else "secondary")
header_cols[admin_col].button("⚙️ " + t("Admin"), key="header_admin", width="stretch", on_click=open_admin, type="primary" if page in ("Admin Panel", "Admin Login") else "secondary")
if st.session_state.admin_logged_in:
    header_cols[logout_col].button(t("Logout"), key="header_logout", width="stretch", on_click=do_logout, type="secondary")

SUBNAV = [
    (t("📊 Analytics"), "Zone Analytics"),
    (t("⚖️ Compare"), "Compare Zones"),
    (t("💧 Quality"), "Water Quality Prediction"),
    (t("📈 Forecast"), "Availability Forecast"),
    (t("🧠 Insights"), "Model Insights"),
]
SUBNAV_PAGES = {"Dashboard", "Indore Zone Map"} | {target for _, target in SUBNAV}

if page in SUBNAV_PAGES and page != "Indore Zone Map":
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
# ADMIN LOGIN
# ============================================================

if page == "Admin Login":

    # Admin-login-only viewport layout: keep the page itself from scrolling and
    # center the login group in the visible browser viewport. Typography and
    # the existing card width are intentionally unchanged.
    st.markdown(
        """
        <div class="admin-login-marker" aria-hidden="true"></div>
        <style>
        /* Lock the Admin Login page itself to the visible viewport. */
        html:has(.admin-login-marker),
        html:has(.admin-login-marker) body {
            overflow: hidden !important;
            height: 100% !important;
            max-height: 100% !important;
        }
        body:has(.admin-login-marker),
        body:has(.admin-login-marker) #root,
        body:has(.admin-login-marker) .stApp,
        body:has(.admin-login-marker) .stAppViewContainer,
        body:has(.admin-login-marker) section[data-testid="stMain"],
        body:has(.admin-login-marker) main {
            overflow: hidden !important;
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
        }
        body:has(.admin-login-marker) section[data-testid="stMain"] > div,
        body:has(.admin-login-marker) .main .block-container,
        body:has(.admin-login-marker) div[data-testid="stMainBlockContainer"] {
            min-height: 100vh !important;
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            width: min(var(--aqua-content-width), calc(100vw - 2rem)) !important;
            transform: translate(-50%, -50%) !important;
            z-index: 20 !important;
            margin: 0 !important;
        }
        body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) + div[data-testid="stHorizontalBlock"] {
            position: fixed !important;
            top: calc(50% + 183px) !important;
            left: 50% !important;
            width: min(var(--aqua-content-width), calc(100vw - 2rem)) !important;
            transform: translateX(-50%) !important;
            z-index: 20 !important;
            margin: 0 !important;
        }
        body:has(.admin-login-marker) .aqua-footer {
            position: fixed !important;
            width: min(var(--aqua-content-width), calc(100vw - 2rem)) !important;
            max-width: none !important;
            left: 50% !important;
            right: auto !important;
            transform: translateX(-50%) !important;
            bottom: 0 !important;
            margin: 0 !important;
            z-index: 10 !important;
            background: #f4f9fc !important;
            box-sizing: border-box !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # MOBILE ONLY: Admin Login overrides kept in a separate style block.
    st.markdown(
        '''
<style>
@media (max-width: 760px) {
            /* MOBILE ONLY: let the Admin Login page use normal document flow.
               This keeps the whole card, Back button and footer reachable by
               normal page scrolling instead of clipping them inside the phone viewport. */
            html:has(.admin-login-marker),
            html:has(.admin-login-marker) body,
            body:has(.admin-login-marker),
            body:has(.admin-login-marker) #root,
            body:has(.admin-login-marker) .stApp,
            body:has(.admin-login-marker) .stAppViewContainer,
            body:has(.admin-login-marker) section[data-testid="stMain"],
            body:has(.admin-login-marker) main {
                overflow-y:auto !important;
                overflow-x:hidden !important;
                height:auto !important;
                min-height:100% !important;
                max-height:none !important;
            }

            body:has(.admin-login-marker) section[data-testid="stMain"] > div,
            body:has(.admin-login-marker) .main .block-container,
            body:has(.admin-login-marker) div[data-testid="stMainBlockContainer"] {
                height:auto !important;
                min-height:0 !important;
                max-height:none !important;
                overflow:visible !important;
            }

            /* MOBILE ONLY: make the admin login a proper full-width phone card. */
            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) {
                position:relative !important;
                top:auto !important;
                left:auto !important;
                width:calc(100vw - 1.25rem) !important;
                max-width:calc(100vw - 1.25rem) !important;
                transform:none !important;
                display:flex !important;
                gap:0 !important;
                margin:1rem auto 0 !important;
                z-index:1 !important;
            }

            /* The desktop 1:2:1 spacer columns are removed on phones. */
            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) > div:first-child,
            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) > div:last-child {
                display:none !important;
            }

            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) > div:nth-child(2) {
                width:100% !important;
                min-width:100% !important;
                max-width:100% !important;
                flex:1 1 100% !important;
            }

            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] {
                width:100% !important;
                max-width:none !important;
                box-sizing:border-box !important;
                padding:1rem !important;
                border-radius:22px !important;
            }

            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] h2 {
                font-size:1.8rem !important;
                line-height:1.08 !important;
                margin:.45rem 0 .35rem !important;
            }

            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] p {
                font-size:.92rem !important;
                line-height:1.4 !important;
                margin:0 !important;
            }

            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] input {
                min-height:2.8rem !important;
            }

            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] label {
                font-size:.9rem !important;
            }

            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) div[data-testid="stForm"] button {
                min-height:2.8rem !important;
            }

            /* Back to Public Portal stays directly after the card in normal flow. */
            body:has(.admin-login-marker) div[data-testid="stHorizontalBlock"]:has(div[data-testid="stForm"]) + div[data-testid="stHorizontalBlock"] {
                position:relative !important;
                top:auto !important;
                left:auto !important;
                width:calc(100vw - 1.25rem) !important;
                max-width:calc(100vw - 1.25rem) !important;
                transform:none !important;
                margin:1rem auto 1.5rem !important;
                z-index:1 !important;
            }

            /* Footer is also part of normal page flow on phones. */
            body:has(.admin-login-marker) .aqua-footer {
                position:relative !important;
                left:auto !important;
                right:auto !important;
                bottom:auto !important;
                width:100% !important;
                max-width:none !important;
                transform:none !important;
                margin-top:1rem !important;
                margin-bottom:0 !important;
            }

            /* End the Admin Login document immediately after its footer.
               No artificial viewport-height area is added below it. */
            body:has(.admin-login-marker) section[data-testid="stMain"] {
                min-height:0 !important;
                padding-bottom:0 !important;
            }
            body:has(.admin-login-marker) section[data-testid="stMain"] > div {
                padding-bottom:0 !important;
            }
        }
</style>
        ''',
        unsafe_allow_html=True,
    )

    _, login_col, _ = st.columns([1, 2, 1])
    with login_col:
        # Keep the original page proportions and typography.
        # Header + credentials are intentionally contained in one single card.
        with st.form("admin_login_form"):
            st.markdown(
                f"""
                <div style="padding:.35rem .15rem 1.1rem;">
                    <div style="color:#0785a9;font-size:.74rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;">{t("Restricted Access")}</div>
                    <h2 style="margin:.65rem 0 .4rem;font-size:2.15rem;letter-spacing:-.045em;">{t("Admin Login")}</h2>
                    <p style="color:#728797;margin:0;line-height:1.6;">{t("Sign in with the AquaTrack administrator account to review reports and manage project data.")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.text_input(t("Admin Username"), key="admin_username")
            st.text_input(t("Admin Password"), type="password", key="admin_password")

            st.form_submit_button(
                t("🔐 Sign In as Admin"), width="stretch", type="primary",
                on_click=attempt_admin_login,
            )

            if st.session_state.admin_login_error:
                st.toast(t(st.session_state.admin_login_error), icon="❌")

        # Keep Back to Public Portal outside the blue login action area,
        # matching the original page layout.
        st.button(
            t("← Back to Public Portal"),
            key="admin_back_public",
            width="stretch",
            type="secondary",
            on_click=navigate_to,
            args=("Dashboard",),
        )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-badge">● {t("Predictive Water Intelligence")}</div>
            <div class="hero-title">{t("Protect Your Water Sources")}</div>
            <div class="hero-text">
                {t("Analyze water quality, forecast availability and understand zone-wise conditions through one integrated decision-support platform for Indore.")}
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

    section("Zone Status Right Now")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("SAFE ZONES"), int((snapshot["risk"] == "Safe").sum()))
    col2.metric(t("MODERATE RISK"), int((snapshot["risk"] == "Moderate Risk").sum()))
    col3.metric(t("HIGH RISK"), int((snapshot["risk"] == "High Risk").sum()))
    col4.metric(t("SCARCITY RISK"), int((snapshot["availability_status"] == "Scarcity Risk").sum()))

    st.dataframe(snapshot_display(snapshot), width="stretch", hide_index=True)

    alert_rows = []
    for _, zone_row in snapshot.iterrows():
        risk = zone_row["risk"] if zone_row["risk"] in RISK_COLORS else "Safe"
        status = zone_row["availability_status"] if zone_row["availability_status"] in STATUS_COLORS else "Good Availability"
        for alert in generate_alert(risk, status):
            alert_rows.append((zone_row["zone_name"], alert))
    alert_rows.sort(key=lambda item: 0 if item[1]["severity"] == "High" else 1)

    section("Active Alerts")
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

    section("Water Quality Risk Overview")
    risk_counts = get_risk_distribution()
    chart_col1, chart_col2 = st.columns(2)
    if not risk_counts.empty:
        with chart_col1:
            fig = px.bar(
                risk_counts, x="Risk Level", y="Records", text="Records", color="Risk Level",
                color_discrete_map=RISK_COLORS, title="Risk distribution (all records)",
            )
            fig.update_layout(showlegend=False)
            show_chart(fig, 340)
        with chart_col2:
            zone_risk = snapshot[snapshot["risk"].isin(RISK_ORDER)]["risk"].value_counts().reindex(RISK_ORDER).fillna(0)
            fig = px.pie(
                names=zone_risk.index, values=zone_risk.values, hole=0.55,
                color=zone_risk.index, color_discrete_map=RISK_COLORS,
                title="Latest risk by zone",
            )
            show_chart(fig, 340)
    else:
        st.warning(t("No water-quality data available."))

    section("Water Availability Overview")
    st.markdown('<div class="mobile-availability-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
    supply = snapshot.dropna(subset=["supply_hours"]).sort_values("supply_hours")
    if not supply.empty:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.bar(
                supply, x="supply_hours", y="zone_name", orientation="h",
                color="availability_status", color_discrete_map=STATUS_COLORS,
                title="Latest supply hours by zone",
                labels={"supply_hours": "Supply hours", "zone_name": "", "availability_status": ""},
            )
            fig.add_vline(x=5, line_dash="dash", line_color="#dc2626")
            fig.add_vline(x=8, line_dash="dash", line_color="#16a34a")
            show_chart(fig, 400)
        with chart_col2:
            seasonal = get_seasonal_supply()
            fig = px.bar(
                seasonal, x="season", y="supply_hours", text_auto=".2f",
                title="Average supply hours by season",
                labels={"supply_hours": "Hours", "season": ""},
            )
            fig.update_traces(marker_color="#12a9db")
            show_chart(fig, 400)
    else:
        st.warning(t("No availability data available."))

    section("Recent Predictions")
    history = get_prediction_history()
    if not history.empty:
        recent = history.head(10).copy()
        recent["Type"] = recent["risk_level"].apply(
            lambda value: "Water Quality" if pd.notna(value) else "Availability Forecast"
        )
        st.dataframe(
            recent[["prediction_id", "zone", "prediction_date", "Type", "risk_level", "availability_forecast"]],
            width="stretch", hide_index=True,
        )
    else:
        st.info(t("No predictions have been recorded yet."))

    section("Data & Model Health")
    health = get_system_health()
    q_date = health["quality_date"].strftime("%d %b %Y") if pd.notna(health["quality_date"]) else t("No date available")
    a_date = health["availability_date"].strftime("%d %b %Y") if pd.notna(health["availability_date"]) else t("No date available")
    model_status = t("Models ready") if health["models_ready"] else t("Model files unavailable")
    model_icon = "🟢" if health["models_ready"] else "🟠"
    st.markdown(
        f"""<div class=\"system-status-grid\">
            <div class=\"system-status-card\"><div class=\"system-status-label\">{t("Model Status")}</div><div class=\"system-status-value\">{model_icon} {model_status}</div><div class=\"system-status-detail\">{t("Model files are available for predictions.") if health["models_ready"] else t("Check the model files before running predictions.")}</div></div>
            <div class=\"system-status-card\"><div class=\"system-status-label\">{t("Latest quality data")}</div><div class=\"system-status-value\">{q_date}</div><div class=\"system-status-detail\">{counts["water_quality"]:,} {t("quality records")}</div></div>
            <div class=\"system-status-card\"><div class=\"system-status-label\">{t("Latest availability data")}</div><div class=\"system-status-value\">{a_date}</div><div class=\"system-status-detail\">{counts["availability"]:,} {t("availability records")}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    section("How AquaTrack Works")
    st.markdown('<div class="mobile-how-works-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for col, (icon, title, text) in zip(columns, [
        ("💧", "Quality Assessment", "Seven parameters screened against BIS IS 10500:2012 and classified by a Random Forest."),
        ("📈", "Predictive Forecast", "Linear Regression estimates supply hours from history, rainfall and season."),
        ("📍", "Zone Insights", "Compare project-defined zones with charts, maps and trends."),
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
            st.dataframe(table, width="stretch", hide_index=True)

            recent_quality = filter_recent(quality_data, period_days)
            recent_quality["risk"] = predict_quality_batch(recent_quality)

            parameter = st.selectbox(
                t("Water quality parameter"), DB_QUALITY_COLS,
                format_func=lambda key: PARAM_INFO[key]["label"], key="analytics_parameter",
            )
            trend_col, share_col = st.columns([2, 1])
            with trend_col:
                fig = px.line(
                    recent_quality, x="date", y=parameter, markers=len(recent_quality) <= 60,
                    title=f"{PARAM_INFO[parameter]['label']} trend",
                )
                fig.update_traces(line_color="#0e9ed3")
                info = PARAM_INFO[parameter]
                if "max" in info:
                    fig.add_hline(y=info["max"], line_dash="dash", line_color="#dc2626",
                                  annotation_text=f"BIS max {info['max']}")
                if "min" in info:
                    fig.add_hline(y=info["min"], line_dash="dash", line_color="#dc2626",
                                  annotation_text=f"BIS min {info['min']}")
                show_chart(fig, 360)
            with share_col:
                share = recent_quality["risk"].value_counts().reindex(RISK_ORDER).fillna(0)
                fig = px.pie(
                    names=share.index, values=share.values, hole=0.55,
                    color=share.index, color_discrete_map=RISK_COLORS,
                    title="Predicted risk share",
                )
                fig.update_layout(showlegend=False)
                show_chart(fig, 360)

            with st.expander(t("View recent quality records")):
                st.dataframe(quality_data.tail(20), width="stretch", hide_index=True)

    with tab_availability:
        if availability_data.empty:
            st.warning(t("No availability data found for this zone."))
        else:
            recent_availability = filter_recent(availability_data, period_days)
            recent_availability["7-day average"] = recent_availability["supply_hours"].rolling(7, min_periods=1).mean()

            trend_col, season_col = st.columns([2, 1])
            with trend_col:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recent_availability["date"], y=recent_availability["supply_hours"],
                    mode="lines", name="Daily supply", line=dict(color="#93c5fd", width=1.5),
                ))
                fig.add_trace(go.Scatter(
                    x=recent_availability["date"], y=recent_availability["7-day average"],
                    mode="lines", name="7-day average", line=dict(color="#0e9ed3", width=3),
                ))
                fig.add_hline(y=5, line_dash="dash", line_color="#dc2626", annotation_text="Scarcity below 5 h")
                fig.add_hline(y=8, line_dash="dash", line_color="#16a34a", annotation_text="Good from 8 h")
                fig.update_layout(title=t("Supply hours trend"), yaxis_title="Hours")
                show_chart(fig, 360)
            with season_col:
                fig = px.box(
                    availability_data, x="season", y="supply_hours",
                    title=t("Supply by season"), labels={"supply_hours": "Hours", "season": ""},
                )
                fig.update_traces(marker_color="#12a9db")
                show_chart(fig, 360)

            fig = px.scatter(
                recent_availability, x="rainfall", y="supply_hours", color="season",
                title=t("Rainfall vs supply hours"),
                labels={"rainfall": "Rainfall (mm)", "supply_hours": "Supply hours"},
            )
            show_chart(fig, 320)

            with st.expander(t("View recent availability records")):
                st.dataframe(availability_data.tail(20), width="stretch", hide_index=True)

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
        st.dataframe(snapshot_display(chosen), width="stretch", hide_index=True)

        with_data = chosen[chosen["risk"].isin(RISK_ORDER)]
        if not with_data.empty:
            section("Parameter Comparison")
            parameter = st.selectbox(
                t("Parameter"), DB_QUALITY_COLS,
                format_func=lambda key: PARAM_INFO[key]["label"], key="compare_parameter",
            )
            info = PARAM_INFO[parameter]
            fig = px.bar(
                with_data, x="zone_name", y=parameter, color="risk", text_auto=".2f",
                color_discrete_map=RISK_COLORS,
                title=f"Latest {info['label']} by zone",
                labels={"zone_name": "", parameter: info["unit"] or info["label"], "risk": "Risk"},
            )
            if "max" in info:
                fig.add_hline(y=info["max"], line_dash="dash", line_color="#dc2626",
                              annotation_text=f"BIS max {info['max']}")
            show_chart(fig, 360)

            section("Overall Profile")
            st.caption(t("Each axis shows the reading as a percentage of its BIS upper limit. Beyond 100% means the limit is exceeded."))
            fig = go.Figure()
            axis_labels = [PARAM_INFO[key]["label"] for key in DB_QUALITY_COLS]
            for _, zone_row in with_data.iterrows():
                ratios = [
                    float(zone_row[key]) / PARAM_INFO[key]["max"] * 100 for key in DB_QUALITY_COLS
                ]
                fig.add_trace(go.Scatterpolar(
                    r=ratios + ratios[:1], theta=axis_labels + axis_labels[:1],
                    name=zone_row["zone_name"], fill="toself", opacity=0.45,
                ))
            fig.add_trace(go.Scatterpolar(
                r=[100] * (len(axis_labels) + 1), theta=axis_labels + axis_labels[:1],
                name="BIS limit", mode="lines", line=dict(color="#dc2626", dash="dash"),
            ))
            show_chart(fig, 460)

        section("Supply Trend")
        availability = get_availability()
        subset = availability[availability["zone_name"].isin(selected)].copy()
        if subset.empty:
            st.info(t("No availability records for the selected zones."))
        else:
            subset["date"] = pd.to_datetime(subset["date"])
            cutoff = subset["date"].max() - pd.Timedelta(days=60)
            subset = subset[subset["date"] >= cutoff]
            fig = px.line(
                subset, x="date", y="supply_hours", color="zone_name",
                title=t("Supply hours, last 60 days"), labels={"supply_hours": "Hours", "zone_name": "Zone"},
            )
            fig.add_hline(y=5, line_dash="dash", line_color="#dc2626")
            fig.add_hline(y=8, line_dash="dash", line_color="#16a34a")
            show_chart(fig, 380)


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

    st.title(t("🗺️ Indore Zone Map"))
    st.write(t("Interactive map of the project-defined AquaTrack zones. Use the layer control for satellite view."))

    snapshot = get_zone_snapshot()

    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
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
            st.dataframe(table, width="stretch", hide_index=True)
            fig = px.bar(x=probabilities.values * 100, y=probabilities.index, orientation="h", color=probabilities.index, color_discrete_map=RISK_COLORS, title="Model probability by class", labels={"x": "Probability (%)", "y": ""})
            fig.update_layout(showlegend=False)
            show_chart(fig, 260)
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
                st.dataframe(
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


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.title(t("🧠 Model Insights"))
    st.write(t("How the two prototype models behave on the records stored in the database."))

    insights = get_model_insights()

    st.warning(t("Read these numbers carefully. Risk labels come from the BIS violation count, and the same records were used to train the models, so they show agreement with the screening rule and not real-world accuracy. Validation with authorised, laboratory-verified data is the next step."))

    if "quality" in insights:
        quality = insights["quality"]
        section("Water Quality Classifier (Random Forest)")
        col1, col2 = st.columns(2)
        col1.metric(t("AGREEMENT WITH BIS RULE"), f"{quality['accuracy'] * 100:.1f}%")
        col2.metric(t("RECORDS CHECKED"), f"{quality['records']:,}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.imshow(
                quality["matrix"], text_auto=True, aspect="auto",
                color_continuous_scale="Blues", title="Confusion matrix",
            )
            fig.update_layout(coloraxis_showscale=False)
            show_chart(fig, 360)
        with chart_col2:
            fig = px.bar(
                quality["importances"], x="Importance", y="Parameter", orientation="h",
                title="Feature importance",
            )
            fig.update_traces(marker_color="#12a9db")
            show_chart(fig, 360)

        st.markdown(t("##### Per-class scores"))
        st.dataframe(quality["report"].round(3), width="stretch")
    else:
        st.info(t("No water-quality records available for diagnostics."))

    if "availability" in insights:
        availability = insights["availability"]
        section("Availability Forecaster (Linear Regression)")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{availability['mae']:.2f} h")
        col2.metric("R²", f"{availability['r2']:.3f}")
        col3.metric(t("NAIVE BASELINE MAE"), f"{availability['naive_mae']:.2f} h")
        if availability["mae"] < availability["naive_mae"]:
            st.success(t("The model beats the naive baseline (repeat yesterday's supply), so rainfall and season add useful information."))
        else:
            st.info(t("The model does not beat the naive baseline (repeat yesterday's supply). Consider richer features or a time-series model."))

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.scatter(
                availability["scatter"], x="Actual", y="Predicted", opacity=0.5,
                title="Actual vs predicted supply hours",
            )
            low, high = availability["scatter"]["Actual"].min(), availability["scatter"]["Actual"].max()
            fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="#dc2626", dash="dash"))
            show_chart(fig, 360)
        with chart_col2:
            fig = px.bar(
                availability["coefficients"], x="Coefficient", y="Feature", orientation="h",
                title="Model coefficients",
            )
            fig.update_traces(marker_color="#12a9db")
            show_chart(fig, 360)
        st.caption(f"Intercept: {availability['intercept']:.3f}. {availability['records']:,} records used.")
    else:
        st.info(t("No availability records available for diagnostics."))


# ============================================================
# WATER BODY ISSUE REPORTING
# ============================================================

elif page == "Report Water Body Issue":

    # Keep the report page title aligned with the Resident Alerts page title.
    # The marker uses the same Streamlit block spacing as the Resident Alerts
    # page without changing the report form layout or functionality.
    st.markdown(
        '<div class="report-water-body-marker" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    st.title(t("🚨 Report Water Body Issue"))
    st.write(
        t("Report a visible water-body or local water-supply issue for project-defined water bodies and served areas.")
    )

    water_bodies = get_water_bodies()

    if water_bodies.empty:
        st.warning(t("No project-defined water bodies are available."))
    else:
        selected_water_body_name = st.selectbox(
            t("Select Water Body"), water_bodies["name"].tolist(), key="report_water_body"
        )
        selected_water_body = water_bodies[water_bodies["name"] == selected_water_body_name].iloc[0]
        selected_water_body_id = int(selected_water_body["water_body_id"])
        supply_data = get_water_body_supply(selected_water_body_id)

        open_reports = db_query(
            """
            SELECT COUNT(*) AS c FROM water_body_reports
            WHERE water_body_id = ? AND status IN ('Pending', 'Investigating')
            """,
            params=(selected_water_body_id,),
        )["c"].iloc[0]
        st.caption(f"{t('Open reports for this water body:')} {int(open_reports)}")

        selected_supply = None
        if supply_data.empty:
            st.warning(t("No served colony/society relationship is available for this water body."))
        else:
            supply_options = []
            for _, supply_row in supply_data.iterrows():
                colony = str(supply_row["colony_name"])
                society = str(supply_row["society_name"])
                supply_options.append(
                    f"{colony} — {society}" if society and society.lower() not in ("nan", "none") else colony
                )
            selected_supply_label = st.selectbox(
                t("Select Colony / Society"), supply_options, key="report_served_area"
            )
            selected_supply = supply_data.iloc[supply_options.index(selected_supply_label)]

        if selected_supply is not None:
            issue_type = st.selectbox(
                t("Issue Type"),
                [
                    "Water appears dirty", "Bad smell", "Unusual colour", "Floating waste",
                    "Low water supply", "Possible contamination", "Other",
                ],
                format_func=t, key="report_issue_type",
            )
            description = st.text_area(
                t("Description"), placeholder=t("Describe what you observed..."), height=130,
                key="report_description",
            )
            severity = st.selectbox(t("Severity"), ["Low", "Medium", "High"], format_func=t, index=1, key="report_severity")
            photo = st.file_uploader(
                t("Add Photo (optional, up to 5 MB)"), type=["jpg", "jpeg", "png", "webp"],
                key="report_photo",
            )
            if photo is not None:
                st.image(photo, caption=t("Selected report photo"), width=360)

            st.markdown(t("### 📍 Report Location"))
            st.caption(t("Allow location access if you want to report from your current device location."))
            current_location = streamlit_geolocation()

            current_latitude = current_longitude = None
            if isinstance(current_location, dict):
                if current_location.get("latitude") is not None and current_location.get("longitude") is not None:
                    current_latitude = float(current_location["latitude"])
                    current_longitude = float(current_location["longitude"])
                    accuracy = current_location.get("accuracy")
                    if accuracy is not None:
                        st.caption(f"{t('Current location detected • Accuracy:')} {float(accuracy):.1f} m")
                    if st.button(t("🔎 Get Current Address"), key="reverse_geocode_location"):
                        with st.spinner(t("Getting location details...")):
                            st.session_state["current_location_name"] = reverse_geocode(
                                round(current_latitude, 5), round(current_longitude, 5)
                            )
                            st.toast(
                                f"{t('Current address:')} {st.session_state['current_location_name']}",
                                icon="📍",
                            )
            if "current_location_name" in st.session_state:
                st.caption(f"{t('Current address:')} {st.session_state['current_location_name']}")

            location_options = ["Selected Water Body Location"]
            if current_latitude is not None:
                location_options.append("My Current Location")
            location_source = st.selectbox(t("Select Report Location"), location_options, format_func=t, key="report_location_source")

            if location_source == "My Current Location" and current_latitude is not None:
                report_latitude, report_longitude = current_latitude, current_longitude
                location_label = "Your detected current location"
            else:
                report_latitude = float(selected_water_body["latitude"])
                report_longitude = float(selected_water_body["longitude"])
                location_label = "Selected water body location"
            st.caption(f"{t('Location used for this report:')} **{t(location_label)}**")

            location_map = make_base_map([report_latitude, report_longitude], 14)
            folium.Marker(
                location=[float(selected_water_body["latitude"]), float(selected_water_body["longitude"])],
                tooltip=str(selected_water_body["name"]),
                popup=f"Selected water body: {selected_water_body['name']}",
                icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
            ).add_to(location_map)
            if current_latitude is not None:
                folium.Marker(
                    location=[current_latitude, current_longitude],
                    tooltip="Your current location",
                    icon=folium.Icon(color="red", icon="user", prefix="fa"),
                ).add_to(location_map)
            st_folium(location_map, width=None, height=360, key="water_body_report_location_map", returned_objects=[])

            if st.button(t("🚨 Submit Report"), width="stretch", type="primary", key="submit_water_body_report"):
                if not description.strip():
                    st.toast(t("Please enter a description of the issue."), icon="⚠️")
                elif photo is not None and photo.size > 5 * 1024 * 1024:
                    st.toast(t("The photo is larger than 5 MB. Please choose a smaller image."), icon="⚠️")
                else:
                    photo_relative_path = None
                    try:
                        if photo is not None:
                            photo_directory = BASE_DIR / "reports" / "water_body_reports"
                            photo_directory.mkdir(parents=True, exist_ok=True)
                            safe_original = re.sub(r"[^A-Za-z0-9._-]", "_", Path(photo.name).name)
                            file_name = f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_original}"
                            photo_path = photo_directory / file_name
                            photo_path.write_bytes(photo.getbuffer())
                            photo_relative_path = str(photo_path.relative_to(BASE_DIR))

                        selected_zone_id = (
                            None
                            if pd.isna(selected_supply["zone_id"])
                            else int(selected_supply["zone_id"])
                        )

                        report_id = db_execute(
                            """
                            INSERT INTO water_body_reports
                                (water_body_id, zone_id, colony_name, society_name, issue_type,
                                 description, photo_path, latitude, longitude, status, severity,
                                 reporter_user_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pending', ?, ?)
                            """,
                            (
                                selected_water_body_id,
                                selected_zone_id,
                                str(selected_supply["colony_name"]),
                                str(selected_supply["society_name"]),
                                issue_type,
                                description.strip(),
                                photo_relative_path,
                                float(report_latitude),
                                float(report_longitude),
                                severity,
                                None,
                            ),
                        )

                        notified, emailed = notify_report_zone(
                            report_id=report_id,
                            zone_id=selected_zone_id,
                            issue_type=issue_type,
                            description=description.strip(),
                            severity=severity,
                        )

                        clear_data_caches()

                        st.toast(t("Report submitted successfully. Status: Pending"), icon="✅")

                        if notified:
                            st.toast(
                                f"🔔 {notified} resident(s) in the selected zone were notified "
                                f"({emailed} email delivery(ies), plus in-app alert records).",
                                icon="🔔",
                            )
                        elif selected_zone_id is None:
                            st.toast(
                                "No zone is linked to this served area, so a zone alert could not be sent.",
                                icon="⚠️",
                            )
                    except Exception as error:  # noqa: BLE001
                        st.toast(f"{t('Unable to submit report:')} {error}", icon="❌")

            st.caption(t("This is a project-level reporting prototype. Water-body and served-area relationships shown here are project-defined demonstration data and are not claimed as official municipal relationships."))


# ============================================================
# RESIDENT ALERTS
# ============================================================

elif page == "Resident Alerts":

    # Resident-alert page viewport layout: keep the page fixed like Admin Login
    # so the registration screen fits in the visible browser area without page scrolling.
    st.markdown(
        """
        <div class="resident-alert-marker" aria-hidden="true"></div>
        <style>
        /* Lock only the Resident Alerts page to the visible viewport. */
        html:has(.resident-alert-marker),
        html:has(.resident-alert-marker) body {
            overflow: hidden !important;
            height: 100% !important;
            max-height: 100% !important;
        }
        body:has(.resident-alert-marker),
        body:has(.resident-alert-marker) #root,
        body:has(.resident-alert-marker) .stApp,
        body:has(.resident-alert-marker) .stAppViewContainer,
        body:has(.resident-alert-marker) section[data-testid="stMain"],
        body:has(.resident-alert-marker) main {
            overflow: hidden !important;
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
        }
        body:has(.resident-alert-marker) section[data-testid="stMain"] > div,
        body:has(.resident-alert-marker) .main .block-container,
        body:has(.resident-alert-marker) div[data-testid="stMainBlockContainer"] {
            min-height: 100vh !important;
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        body:has(.resident-alert-marker) .aqua-footer {
            position: fixed !important;
            width: min(var(--aqua-content-width), calc(100vw - 2rem)) !important;
            max-width: none !important;
            left: 50% !important;
            right: auto !important;
            transform: translateX(-50%) !important;
            bottom: 0 !important;
            margin: 0 !important;
            z-index: 10 !important;
            background: #f4f9fc !important;
            box-sizing: border-box !important;
        }
        /* Keep resident-page alerts fully visible above the fixed footer. */
        body:has(.resident-alert-marker) div[data-testid="stAlert"] {
            position: relative !important;
            z-index: 30 !important;
            margin-top: 10px !important;
            margin-bottom: 18px !important;
            box-sizing: border-box !important;
            white-space: normal !important;
            overflow: visible !important;
        }
        /* A registration confirmation must never sit underneath the footer. */
        body:has(.resident-alert-marker) div[data-testid="stAlert"]:has(p) {
            max-width: 100% !important;
        }
        body:has(.resident-alert-marker) .main .block-container {
            padding-bottom: 90px !important;
        }
        /* After registration, place only the confirmation box in the reserved area
           above the footer so it cannot collide with the footer text. */
        body:has(.resident-registration-success-marker) .resident-registration-success-marker {
            display: none !important;
        }
        body:has(.resident-alert-marker) div[data-testid="stAlert"] p {
            white-space: normal !important;
            overflow: visible !important;
            line-height: 1.5 !important;
            margin: 0 !important;
        }
        @media (max-width: 760px) {
            body:has(.resident-alert-marker) .aqua-footer {
                width: calc(100vw - 2rem) !important;
                left: 50% !important;
                right: auto !important;
                transform: translateX(-50%) !important;
            }
            body:has(.resident-alert-marker) main {
                overflow-y: auto !important;
                overflow-x: hidden !important;
            }
        }

        /* Resident Alerts must remain a normal scrollable page on every device.
           The registration form is taller than many browser viewports, so a
           viewport lock or fixed footer can hide the lower fields/button. */
        html:has(.resident-alert-marker),
        html:has(.resident-alert-marker) body,
        body:has(.resident-alert-marker),
        body:has(.resident-alert-marker) #root,
        body:has(.resident-alert-marker) .stApp,
        body:has(.resident-alert-marker) .stAppViewContainer,
        body:has(.resident-alert-marker) section[data-testid="stMain"],
        body:has(.resident-alert-marker) main {
            overflow: auto !important;
            height: auto !important;
            max-height: none !important;
        }
        body:has(.resident-alert-marker) section[data-testid="stMain"] > div,
        body:has(.resident-alert-marker) .main .block-container,
        body:has(.resident-alert-marker) div[data-testid="stMainBlockContainer"] {
            min-height: auto !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }
        body:has(.resident-alert-marker) .aqua-footer {
            position: relative !important;
            width: min(var(--aqua-content-width), calc(100vw - 2rem)) !important;
            left: 50% !important;
            right: auto !important;
            bottom: auto !important;
            transform: translateX(-50%) !important;
            margin: 1.5rem 0 0 !important;
            z-index: auto !important;
        }
        body:has(.resident-alert-marker) .main .block-container {
            padding-bottom: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(t("📢 Resident Water Alerts"))
    st.write(
        t(
            "Register your area once. New water-problem reports for that zone "
            "will automatically appear in your in-app alerts."
        )
    )

    zones = get_zones()

    if zones.empty:
        st.warning(t("No project-defined zones are available."))
    else:
        with st.form("resident_signup"):
            zone_names = zones["zone_name"].tolist()
            zone_name = st.selectbox(t("Your Zone"), zone_names)
            name = st.text_input(t("Your Name"))
            phone = st.text_input(t("Phone Number"))
            email = st.text_input(t("Email (for alerts)"))

            submitted = st.form_submit_button(
                t("🔔 Register for Alerts"),
                type="primary",
                width="stretch",
            )

        if submitted:
            if not phone.strip() and not email.strip():
                st.toast(
                    t("Please provide a phone number or email."),
                    icon="⚠️",
                )
            else:
                zone_id = int(
                    zones.loc[zones["zone_name"] == zone_name, "zone_id"].iloc[0]
                )
                resident_id, created = register_resident(
                    zone_id, name, phone, email
                )

                st.session_state["resident_id"] = resident_id
                st.session_state["resident_zone_id"] = zone_id
                st.session_state["resident_zone_name"] = zone_name

                # Use a toast popup so the confirmation never collides with the form or fixed footer.
                if created:
                    st.toast(
                        f"Registration successful! You are now registered to receive water alerts for {zone_name}. "
                        "New water-problem reports for this zone will appear in your Alerts section.",
                        icon="✅",
                    )
                else:
                    st.toast(
                        f"Registration updated! You are registered to receive water alerts for {zone_name}. "
                        "New water-problem reports for this zone will appear in your Alerts section.",
                        icon="✅",
                    )

        resident_id = st.session_state.get("resident_id")
        if resident_id:
            resident = get_resident(resident_id)

            if not resident.empty:
                row = resident.iloc[0]
                st.divider()
                st.subheader(t("🔔 Your Recent Alerts"))
                st.caption(
                    f"{t('Zone')}: {st.session_state.get('resident_zone_name', row['zone_id'])} · "
                    f"{row['email'] or row['phone'] or 'In-app'}"
                )

                alerts = get_resident_alerts(resident_id, 50)
                if alerts.empty:
                    st.info(
                        t(
                            "No alerts yet. New reports for your registered zone "
                            "will appear here automatically."
                        )
                    )
                else:
                    for _, alert in alerts.iterrows():
                        st.info(
                            f"**{alert['created_at']}** · {alert['message']}"
                        )

                st.caption(
                    "Real SMS/push delivery requires a configured provider such as "
                    "Twilio, MSG91 or Gupshup. Email delivery is available through SMTP."
                )


# ============================================================
# ADMIN PANEL
# ============================================================

elif page == "Admin Panel":

    st.caption(f"🔐 {t('Administrator')}: {st.session_state.admin_user[1]}")
    st.title(t("👨‍💻 AquaTrack Admin Panel"))

    admin_section = st.radio(
        t("Admin section"),
        ["Overview", "Water Body Reports", "Resident Alerts", "Data Import", "Predictions & Exports", "System"],
        format_func=t, horizontal=True, label_visibility="collapsed", key="admin_section",
    )
    st.divider()

    # ---------------- OVERVIEW ----------------
    if admin_section == "Overview":
        counts = get_table_counts()
        reports_now = get_water_body_reports()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("Project Zones"), counts["zones"])
        col2.metric(t("Water Quality Records"), f"{counts['water_quality']:,}")
        col3.metric(t("Availability Records"), f"{counts['availability']:,}")
        col4.metric(t("PREDICTIONS"), counts["predictions"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("Water Bodies"), counts["water_bodies"])
        col2.metric(t("Total Reports"), len(reports_now))
        col3.metric(t("Pending"), int((reports_now["status"] == "Pending").sum()) if not reports_now.empty else 0)
        col4.metric(t("Resolved"), int((reports_now["status"] == "Resolved").sum()) if not reports_now.empty else 0)

        section("📍 Zone Status")
        st.dataframe(snapshot_display(get_zone_snapshot()), width="stretch", hide_index=True)

        section("📍 Project Zones")
        st.dataframe(get_zones(), width="stretch", hide_index=True)

        section("🤖 Machine Learning Models")
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

    # ---------------- WATER BODY REPORTS ----------------
    elif admin_section == "Water Body Reports":
        water_body_reports = get_water_body_reports()
        water_bodies = get_water_bodies()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("📊 TOTAL REPORTS"), len(water_body_reports))
        col2.metric(t("⏳ PENDING"), int((water_body_reports["status"] == "Pending").sum()) if not water_body_reports.empty else 0)
        col3.metric(t("🔎 INVESTIGATING"), int((water_body_reports["status"] == "Investigating").sum()) if not water_body_reports.empty else 0)
        col4.metric(t("✅ RESOLVED"), int((water_body_reports["status"] == "Resolved").sum()) if not water_body_reports.empty else 0)
        st.caption(f"{t('Project-defined water bodies available:')} {len(water_bodies)}")

        if water_body_reports.empty:
            st.info(t("No water-body issue reports have been submitted yet."))
        else:
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

            filter_col1, filter_col2 = st.columns([3, 1])
            selected_filter = filter_col1.radio(
                t("Report filter"), ["All", "Pending", "Investigating", "Resolved"],
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
                else water_body_reports[water_body_reports["status"] == selected_filter].copy()
            )
            st.markdown(f"### {t('Reports')} — {len(filtered)} {t('shown')}")

            column_weights = [0.75, 2.0, 3.1, 1.35, 1.55, 1.55]
            table_header = st.columns(column_weights, gap="small")
            for header, label in zip(
                table_header,
                ["PHOTO", "LOCATION", "DESCRIPTION", "DATE", "SEVERITY", "STATUS"],
            ):
                header.markdown(f"**{t(label)}**")
            st.divider()

            status_options = ["Pending", "Investigating", "Resolved"]
            severity_options = ["Low", "Medium", "High"]

            def _status_changed(report_id, widget_key):
                new_status = st.session_state[widget_key]
                update_water_body_report_status(report_id, new_status)
                st.toast(f"Report status updated to {new_status}.", icon="🔄")

            def _severity_changed(report_id, widget_key):
                new_severity = st.session_state[widget_key]
                update_water_body_report_severity(
                    report_id,
                    new_severity,
                )
                st.toast(f"Report severity updated to {new_severity}.", icon="⚠️" if new_severity == "High" else "🔄")
                if new_severity == "High":
                    count, email_count = notify_high_severity_report(report_id)
                    if count:
                        st.toast(
                            f"🔔 {count} resident(s) notified; {email_count} email(s) delivered.",
                            icon="🚨",
                        )

            if filtered.empty:
                st.info(f"{t('No')} {t(selected_filter).lower()} {t('water-body reports found.')}")
            for _, report in filtered.iterrows():
                report_id = int(report["report_id"])
                current_status = str(report["status"])
                row_cols = st.columns(column_weights, gap="small")

                photo_value = report.get("photo_path")
                if pd.notna(photo_value) and str(photo_value).strip() and (BASE_DIR / str(photo_value)).exists():
                    row_cols[0].image(str(BASE_DIR / str(photo_value)), width=68)
                else:
                    row_cols[0].markdown("🖼️")

                colony = str(report["colony_name"])
                society = str(report["society_name"])
                served_area = f"{colony} — {society}" if society.strip() and society.lower() not in ("nan", "none") else colony
                row_cols[1].markdown(f"**{report['water_body']}**  \n{served_area}")
                if pd.notna(report["latitude"]) and pd.notna(report["longitude"]):
                    row_cols[1].caption(f"{float(report['latitude']):.4f}, {float(report['longitude']):.4f}")

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
                    t("Severity"),
                    severity_options,
                    index=(
                        severity_options.index(current_severity)
                        if current_severity in severity_options else 0
                    ),
                    format_func=t,
                    key=severity_key,
                    label_visibility="collapsed",
                    on_change=_severity_changed,
                    args=(report_id, severity_key),
                )

                widget_key = f"report_status_dropdown_{report_id}"
                row_cols[5].selectbox(
                    f"{t('Status for Report #')}{report_id}",
                    status_options,
                    index=(
                        status_options.index(current_status)
                        if current_status in status_options else 0
                    ),
                    format_func=t,
                    key=widget_key,
                    label_visibility="collapsed",
                    on_change=_status_changed,
                    args=(report_id, widget_key),
                )
                row_cols[5].caption(f"{t('Report #')}{report_id}")
                st.divider()

            st.markdown(t("### 🗺️ Report Map"))
            report_map = make_base_map([22.7196, 75.8577], 11)
            for _, report in water_body_reports.iterrows():
                if pd.isna(report["latitude"]) or pd.isna(report["longitude"]):
                    continue
                status = str(report["status"])
                folium.Marker(
                    location=[float(report["latitude"]), float(report["longitude"])],
                    popup=folium.Popup(
                        f"<b>Report #{int(report['report_id'])}</b><br>"
                        f"Water Body: {report['water_body']}<br>"
                        f"Issue: {report['issue_type']}<br>Status: {status}",
                        max_width=320,
                    ),
                    tooltip=f"Report #{int(report['report_id'])} — {status}",
                    icon=folium.Icon(
                        color={"Pending": "red", "Investigating": "orange", "Resolved": "green"}.get(status, "blue"),
                        icon="flag", prefix="fa",
                    ),
                ).add_to(report_map)
            add_map_legend(report_map, [(t("Pending"), "#dc2626"), (t("Investigating"), "#f59e0b"), (t("Resolved"), "#16a34a")])
            st_folium(report_map, width=None, height=480, key="water_body_report_admin_map", returned_objects=[])

    # ---------------- RESIDENT ALERTS ----------------
    elif admin_section == "Resident Alerts":
        st.subheader(t("🔔 Resident Alerts"))

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
            st.dataframe(residents, width="stretch", hide_index=True)

        st.subheader(t("🔔 Alert Log"))
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
            st.dataframe(alert_log, width="stretch", hide_index=True)

        st.subheader(t("📢 Manual Broadcast"))
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
                t("📢 Send Now"),
                type="primary",
                width="stretch",
            )

        if send:
            if not message.strip():
                st.toast(t("Please enter an alert message."), icon="⚠️")
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
                    f"📢 Alert queued for {count} resident(s) in {zone_name}. "
                    f"{email_count} email(s) delivered; all recipients received an in-app alert record.",
                    icon="📢",
                )

    # ---------------- DATA IMPORT ----------------
    elif admin_section == "Data Import":
        st.subheader(t("📂 Upload Water Quality CSV"))
        st.write(
            t("Upload a CSV with water-quality records. The system validates the data and skips records that already exist.")
        )

        if st.session_state.get("import_message"):
            st.toast(st.session_state.pop("import_message"), icon="✅")

        st.caption(t("Required columns: zone, date, pH, TDS, turbidity, hardness, chloride, fluoride, nitrate"))
        uploaded_file = st.file_uploader(t("Choose CSV file"), type=["csv"], key="quality_csv")

        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                st.write(t("### Uploaded Data Preview"))
                st.dataframe(uploaded_data.head(10), width="stretch", hide_index=True)

                required_columns = ["zone", "date"] + QUALITY_FEATURES
                missing_columns = [c for c in required_columns if c not in uploaded_data.columns]

                if missing_columns:
                    st.toast(t("Missing columns: ") + ", ".join(missing_columns), icon="❌")
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
                        st.toast(t("CSV validation failed."), icon="❌")
                        for message in validation_errors:
                            st.warning("⚠️ " + message)
                    else:
                        st.toast(t("CSV validation successful."), icon="✅")
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
                            st.toast(f"{duplicate_count} {t('existing record(s) will be skipped.')}", icon="⚠️")

                        if new_records:
                            if st.button(t("📥 Import New Records"), key="import_new_records"):
                                with closing(sqlite3.connect(DATABASE_NAME)) as connection:
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
                st.toast(f"{t('Unable to process CSV:')} {error}", icon="❌")

    # ---------------- PREDICTIONS & EXPORTS ----------------
    elif admin_section == "Predictions & Exports":
        history = get_prediction_history()
        st.subheader(t("📋 Prediction History"))
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
            st.dataframe(admin_history, width="stretch", hide_index=True)
            st.download_button(
                t("⬇️ Download prediction history (CSV)"),
                data=admin_history.to_csv(index=False).encode("utf-8"),
                file_name="aquatrack_prediction_history.csv", mime="text/csv", key="export_history",
            )

        st.subheader(t("🗄️ Dataset Exports"))
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
        st.subheader(t("⚙️ System Status"))

        # ------------------------------------------------------------
        # REAL EMAIL / SMTP CONFIGURATION
        # ------------------------------------------------------------
        st.markdown("### 📧 Real Email Delivery")
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
                'username = "digitalarpit78@gmail.com"\n'
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
                "📧 Send Test Email",
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

        st.subheader(t("⚙️ System Status"))
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

        if st.button(t("🔄 Refresh cached data"), key="refresh_cache"):
            clear_data_caches()
            st.toast(t("Cached data refreshed successfully."), icon="✅")
        st.caption(t("Data is cached for speed. Use this after editing the database outside the app."))
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="aqua-footer">
        <b>AquaTrack</b> · Predictive Water Quality &amp; Availability Analytics
        <br>
        <span>Project-level decision-support prototype · Indore · For screening and decision support</span>
    </div>
    """,
    unsafe_allow_html=True,
)