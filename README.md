
# 💧 AquaTrack

### A Predictive Analytics System for Water Quality Assessment & Scarcity Forecasting in Indore

**[📥 Download AquaTrack README](https://github.com/jngid9575/AquaTrack/raw/refs/heads/main/README.md)**

**IPS Academy, Indore — Institute of Engineering and Science**  
**Department of Computer Science & Information Technology • Session 2026–27 • Group G-21**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/Status-Decision--Support%20Prototype-0A7EA4)

**[🌐 Live Demo](https://aquatrack.streamlit.app/)** • **[💻 Source Code](https://github.com/jngid9575/AquaTrack)**

AquaTrack is a **Streamlit-based water intelligence and decision-support platform** for **Indore**. It screens water quality, forecasts water availability, collects citizen water-problem reports, manages field-worker service requests, and sends alerts to residents, all in one application with separate **Citizen**, **Worker** and **Admin** portals.

> **Project Status:** B.Tech Major Project (Major Project-I) / Decision-Support Prototype
> **Deployment:** Streamlit Community Cloud
> **Target Location:** Indore, Madhya Pradesh, India

---

## 🌊 Overview

Water quality and water availability vary across zones and seasons, and they are usually viewed separately. Complaints are often handled informally, with no assignment, tracking or confirmation that the problem was fixed. AquaTrack brings these together:

1. **Water Quality Screening** — a trained **Random Forest** classifier labels water as Safe, Moderate Risk or High Risk from 7 parameters, shown beside a direct **BIS IS 10500:2012 rule-based screening** and the model confidence.
2. **Water Availability Forecasting** — a **Linear Regression** model forecasts supply hours (up to 7 days) from previous supply hours, rainfall and season.
3. **Citizen Service Workflow** — citizens report problems with a photo and GPS location, admins assign field workers, workers submit completion evidence, admins verify, and citizens review the result. Every step is recorded in a status history and audit log.

On top of these, AquaTrack provides zone analytics, zone comparison, interactive maps, resident e-mail alerts, prediction history, CSV import, PDF / Excel / CSV exports, system health checks and database backup, in an **English / Hinglish** interface.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 **Dashboard** | Zone snapshot, safe / moderate / high-risk zones, scarcity-risk zones, project alerts and how the system works |
| 💧 **Water Quality Prediction** | Random Forest risk class, BIS parameters exceeded, model confidence, and a simple visual water check for users without a lab report |
| 📈 **Availability Forecast** | Linear Regression forecast with a status (Scarcity Risk / Moderate / Good Availability) and a rolling forecast of up to 7 days |
| 📍 **Zone Analytics & Comparison** | Per-zone quality and supply analysis over 30 / 60 / 90 days or all records, and comparison of 2 to 6 zones |
| 🗺️ **Interactive Maps** | Folium map of zones, risk colours, supply rings and water bodies, with OpenStreetMap and satellite layers |
| 📝 **Citizen Issue Reporting** | Report low supply, water quality, leakage, contamination or other problems with a photo (up to 5 MB), GPS location or a map pin, and an optional society / landmark detail |
| 🔎 **Request Tracking** | Citizens track status, severity, assigned worker and a full request timeline |
| 👷 **Worker Portal** | Workers see assigned requests, start work, submit remarks and completion evidence, and open the reported location in Google Maps (using the water-body location if exact coordinates are unavailable) |
| 🛠️ **Admin Panel** | Report management, worker assignment, worker accounts, citizen reviews, system insights, ML model performance, resident alerts, data import, exports and system tools |
| ⭐ **Citizen Reviews** | 1 to 5 star rating and comment on a resolved request, one review per citizen per request |
| 🔔 **Notifications** | In-app notifications for citizens, workers and admins with read / unread state |
| 🚨 **Resident Alerts** | Zone-based alerts in the app and by e-mail through SMTP, manual broadcast or automatic on high-severity reports |
| 🔐 **Authentication** | Citizen registration and login, Google sign-in, password reset by e-mail, and separate worker and admin logins |
| 🌐 **English / Hinglish Interface** | Language switch across the interface |
| 📄 **Reports & Exports** | PDF and Excel zone reports, and CSV exports of water quality, availability, predictions and reports |
| 🗂️ **Data Import** | Water-quality CSV import with validation and duplicate detection |
| 🩺 **System Health & Backup** | Database and model readiness checks, audit log, timestamped database backup and download |
| 📱 **Responsive UI** | Desktop, tablet and mobile layouts with a mobile navigation drawer |

---

## 👥 User Roles

| Role | What they can do |
|---|---|
| **Visitor** | View the dashboard, zone analytics, comparison, map, water-quality prediction, availability forecast and policy pages |
| **Citizen** | Register / log in, report problems, track requests and timelines, receive notifications, rate resolved requests, manage a profile |
| **Worker** | View assigned requests, start work, submit remarks and evidence, view notifications. Accounts are created by admins |
| **Admin** | Manage reports, workers, reviews, alerts, data import, ML evaluation, exports, health, backup and audit log |

---

## 🔄 Complaint Workflow

```text
Submitted → Assigned → In Progress → Waiting for Verification → Resolved
                                                                    │
                                                                    └─→ Reopened
```

- A citizen submits a report with photo and location (**Submitted**).
- The admin reviews it, sets severity and assigns an active worker (**Assigned**).
- The worker starts work (**In Progress**) and submits remarks and optional evidence; the request then moves to **Waiting for Verification**.
- The admin reviews the submitted work and can resolve or reopen the request. A citizen can rate a resolved request.
- Only valid transitions are allowed, and each status change is stored in the status history and audit log.

---

## 🧠 Machine Learning & Analytics

### Water Quality Classification

A **Random Forest classifier** uses 7 parameters: pH, TDS, turbidity, hardness, chloride, fluoride and nitrate. It outputs:

- 🟢 **Safe**
- 🟠 **Moderate Risk**
- 🔴 **High Risk**

AquaTrack also applies a direct **BIS rule screening** using the project limits (pH 6.5–8.5, TDS 500 mg/L, turbidity 1.0 NTU, hardness 200 mg/L, chloride 250 mg/L, fluoride 1.0 mg/L, nitrate 45 mg/L). Zero violations is Safe, 1–2 is Moderate Risk and 3 or more is High Risk. The app shows the model prediction, the exceeded parameters and the model confidence together.

Trained model: `models/water_quality_rf.joblib`

### Water Availability Forecasting

A **Linear Regression** model uses previous supply hours, rainfall and season (Summer 0, Monsoon 1, Winter 2). Forecasts are classified as:

- Below 5 hours — **Scarcity Risk**
- 5 to under 8 hours — **Moderate Availability**
- 8 hours or more — **Good Availability**

For a multi-day forecast, each day's forecast becomes the next day's previous supply while rainfall and season stay constant. Uncertainty grows with the forecast horizon.

Trained model: `models/availability_linear_regression.joblib`

### Evaluation Summary

The **ML Model Performance** page in the Admin panel reports accuracy, macro precision, recall and F1, confusion matrix and feature importance for the quality model, and MAE, RMSE and R² for the availability model (held-out test split).

| Metric | Result |
|---|---:|
| Random Forest agreement with BIS-rule labels | **95.00%** |
| F1-score (High Risk / Safe) | **0.96 / 0.98** |
| Supply forecast MAE (874 records) | **0.64 h** |
| Supply forecast R² | **0.7158** |

> **Note:** The metrics depend on the dataset used and will change if the models are retrained on different data. Quality labels are derived from BIS-rule limits on synthetic, project-generated data. Laboratory validation with real field samples is future work.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────────────┐
                    │          AquaTrack          │
                    │       Streamlit Web App     │
                    │    (English / Hinglish)     │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   Citizen Portal            Worker Portal              Admin Panel
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Authentication & Roles     │
                    │  Application Logic          │
                    │  Complaint Workflow         │
                    │  Notifications & Alerts     │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
    Random Forest           Linear Regression        BIS Rule Screening
    (Risk class)            (Supply forecast)        & Model Confidence
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ SQLite database / CSV files │
                    │ Uploaded photos / backups   │
                    └─────────────────────────────┘

External services: Gmail SMTP (e-mail alerts) • Google OAuth (sign-in)
                   • Map tiles • Google Maps links • Browser Geolocation API
```

AquaTrack is a **single Streamlit application** (`app.py`), not a microservice system. There is no API gateway and no separate backend service.

---

## 🛠️ Tech Stack

### Application
- **Python**
- **Streamlit**

### Data Science & Machine Learning
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**

### Visualization
- **Plotly**
- **Folium**
- **Streamlit-Folium**

### Database
- **SQLite** (write-ahead logging, foreign keys enforced)

### Reporting & Data Files
- **ReportLab (PDF)**
- **OpenPyXL (Excel)**
- **CSV datasets**

### External Services
- **Gmail SMTP** (e-mail alerts and password-reset links)
- **Google OAuth** (sign-in)

### Deployment
- **GitHub**
- **Streamlit Community Cloud**

---

## 📁 Project Structure

```text
AquaTrack/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── aqua_auth_background.jpg
│
├── data/
│   ├── water_quality.csv
│   ├── water_availability.csv
│   └── water_bodies.csv
│
├── models/
│   ├── availability_linear_regression.joblib
│   └── water_quality_rf.joblib
│
├── reports/                  # Runtime uploads / evidence; keep private user files out of Git
│
├── utils/
│   ├── alerts.py
│   ├── auth.py
│   └── db_functions.py
│
├── app.py
├── database.py
├── check_database.py
├── generate_availability.py
├── generate_water_quality.py
├── import_data.py
├── import_water_bodies.py
├── requirements.txt
└── README.md
```

> The SQLite database (`aquatrack.db`), backup files and uploaded evidence are runtime data. Do not commit them if they contain private or real user information. A `backups/` folder is created at run time when an admin creates a database backup.

> **Security:** Sensitive files such as Streamlit secrets, local databases containing private data, uploaded citizen files and credentials must stay outside a public repository.

---

## 📊 Main Modules

### 1. 🔐 Authentication & Access Control
Citizen registration and login, Google sign-in with OAuth state validation, password reset with hashed tokens that expire after 30 minutes, and separate worker and admin logins. Passwords are stored with PBKDF2-HMAC-SHA256, a random salt and 180,000 iterations. Every protected page checks the user role.

### 2. 📊 Dashboard
Zone counts, safe / moderate-risk / high-risk and scarcity-risk zones, latest zone snapshot and project alerts with recommendations.

### 3. 💧 Water Quality Prediction
Enter the 7 parameters to get a Random Forest risk class, the BIS parameters exceeded and the model confidence. A simple visual check is available for users without a lab report. Predictions can be saved to history.

### 4. 📈 Availability Forecast
Rolling forecast of up to 7 days with an availability status. Admins get technical controls (rainfall, season, horizon) and can save forecasts.

### 5. 🗺️ Analytics & Mapping
Zone analytics, zone comparison with automatic insights, and an interactive map of zones and water bodies.

### 6. 📝 Report Water Issue & My Requests
Citizens submit a report with photo and location, then follow status, assigned worker and timeline. Resolved requests can be reviewed once.

### 7. 👷 Worker Dashboard
Assigned, active and completed requests, start-work and completion actions, and notifications. Worker availability (Available, Busy, Offline) is kept in sync with active work.

### 8. 🛠️ Admin Panel
Ten sections: **Overview, Water Body Reports, Workers, Citizen Reviews, System Insights, ML Model Performance, Resident Alerts, Data Import, Predictions & Exports, System.**

### 9. 🚨 Resident Alerts
Zone-wise alerts in the app and by e-mail. Admins can broadcast a message manually, send a test e-mail to check SMTP, and high-severity reports can notify residents automatically.

### 10. 🩺 System
System health checks, audit log, database backup and download.

---

## 🗄️ Database

AquaTrack uses SQLite. Main tables include:

| Table | Purpose |
|---|---|
| `users` | user id, role, hashed password |
| `residents` | zone id, name, phone, email |
| `zones` | zone id, name, latitude, longitude |
| `water_quality` | zone id, date, 7 parameters |
| `availability` | zone id, date, supply hours, rainfall, season |
| `predictions` | zone id, date, risk level, forecast |
| `water_bodies` | name, type, location, latitude / longitude |
| `water_body_supply` | water body id, zone id, colony, society |
| `water_body_reports` | status, severity, photo path, location, assigned worker and workflow timestamps |
| `aquatrack_workers` | worker details, zone, availability status |
| `request_status_history` | previous / new status, actor, remarks, timestamp |
| `citizen_reviews` | report, citizen, worker, 1–5 rating, comment |
| `aquatrack_notifications` | recipient, report, message, read state |
| `alerts_log` | zone id, report id, resident id, channel, status |
| `aquatrack_audit_log` | action, actor, report id, details, timestamp |

Existing data is preserved when the database is migrated to newer workflow tables.

---

## 📦 Dataset & Parameters

The project uses structured, project-generated water datasets (12 Indore zones).

| Parameter | BIS limit used | Purpose |
|---|---|---|
| **pH** | 6.5–8.5 | Acidity / alkalinity |
| **TDS** | 500 mg/L | Total dissolved solids |
| **Turbidity** | 1.0 NTU | Water clarity |
| **Hardness** | 200 mg/L | Mineral concentration |
| **Chloride** | 250 mg/L | Water chemistry indicator |
| **Fluoride** | 1.0 mg/L | Health-related indicator |
| **Nitrate** | 45 mg/L | Health-related indicator |

Water-availability and water-body data are also used by the analytical modules. Demo data generation must be enabled explicitly through configuration.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12 recommended**
- **Git**
- **A modern web browser**

### 1. Clone the repository

```bash
git clone https://github.com/jngid9575/AquaTrack.git
cd AquaTrack
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv venv
```

### 3. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Configure secrets

Create:

```text
.streamlit/secrets.toml
```

Example (SMTP):

```toml
[smtp]
host = "smtp.gmail.com"
port = 587
username = "your-email@gmail.com"
password = "YOUR_GOOGLE_APP_PASSWORD"
sender_name = "AquaTrack Support"
use_tls = true
```

Google sign-in also needs a Google OAuth client ID, client secret and redirect URI, which are read from Streamlit secrets or environment variables.

> **Never commit real passwords, API keys, or private credentials to GitHub.**

### 6. Run AquaTrack

```powershell
python -m streamlit run app.py
```

The application normally opens in your browser.

---

## ☁️ Deployment

AquaTrack is configured for deployment on **Streamlit Community Cloud**.

**[🌐 AquaTrack · Streamlit](https://aquatrack.streamlit.app/)**

### Important Deployment Notes

- Keep secrets in **Streamlit Community Cloud Secrets**.
- Do not commit `.streamlit/secrets.toml`.
- The prototype uses SQLite and stores report photos / evidence on local disk, so database and upload data can be lost on redeploy.
- A production deployment should use a managed database such as PostgreSQL and external file storage.

---

## 🔒 Security

AquaTrack is intended to demonstrate secure application practices:

- Salted PBKDF2 password hashing with constant-time comparison
- Random password-reset tokens, stored only as hashes, expiring after 30 minutes
- Google OAuth state validation and verified e-mail requirement
- Role-based access, and ownership checks (a worker acts only on assigned requests, a citizen reviews only their own request)
- Valid workflow transitions only, and parameterised SQL queries
- Audit logging of important actions
- Secrets read from deployment configuration, never hard-coded

### Never commit

```text
.streamlit/secrets.toml
.env and other local environment files
API keys and OAuth client secrets
SMTP passwords and private credentials
private SQLite databases or backups
reports/citizen_uploads/
reports/completion_evidence/
```

---

## ⚠️ Project Disclaimer

AquaTrack is a **B.Tech major project and decision-support prototype**.

The application is intended for demonstration, academic evaluation, data analysis and prototype-level planning. Quality labels come from BIS-rule limits on synthetic data. Predictions and classifications must not replace certified laboratory testing, official water-quality measurements, engineering assessment, or government operational decisions.

---

## 🎓 Academic Project

**Project:** AquaTrack — A Predictive Analytics System for Water Quality Assessment and Scarcity Forecasting in Indore

**Project Type:** B.Tech Major Project-I

**Institute:** IPS Academy, Indore — Institute of Engineering and Science

**Department:** Computer Science & Information Technology

**Session:** 2026–27 • **Group:** G-21

**Domain:**

- Data Science
- Machine Learning
- Predictive Analytics
- Water Resource Management
- Web Application Development

**Target City:** Indore, Madhya Pradesh, India

---

## 👥 Project Team

| Name | Enrollment No. | Role |
|---|---|---|
| **Arpit Jangid** | 0808CI231038 | Project Leader |
| **Anuj Patidar** | 0808CI231031 | Team Member |
| **Anush Parmar** | 0808CI231033 | Team Member |
| **Bhushan Bondre** | 0808CI231058 | Team Member |

**Project Guide:** Mr. Sumit Kumar

---

## 🌐 Links

### Source Code

**[💻 AquaTrack · GitHub](https://github.com/jngid9575/AquaTrack)**

### Live Application

**[🌐 AquaTrack · Streamlit](https://aquatrack.streamlit.app/)**

---

## 📌 Future Scope

- Validation with lab-tested water samples
- Real-time IoT water sensors
- Live municipal water-supply APIs
- PostgreSQL production database and scheduled model retraining
- Automated SMS alerts (MSG91 / Twilio) and WhatsApp alerts
- Mobile application
- Advanced time-series forecasting
- GIS-based spatial analytics
- Satellite and remote-sensing integration
- Automated anomaly detection
- More Indore zones and a municipal administration portal

---

## ⭐ Why AquaTrack?

AquaTrack brings together **water-quality screening, machine learning, forecasting, mapping, citizen reporting, worker field service, alerts and reports** in one platform.

The goal is to move from simply viewing historical water information toward a more **predictive and proactive approach to water-resource management**.

---

## 📄 License

This project is developed as an academic B.Tech project. All rights and usage terms should be determined by the project team.
