💧 AquaTrack

A Predictive Analytics System for Water Quality Assessment & Scarcity Forecasting in Indore

📥 Download AquaTrack README

IPS Academy, Indore — Institute of Engineering and Science
Department of Computer Science & Information Technology • Session 2026–27 • Group G-21

AquaTrack is a Streamlit-based water intelligence and decision-support platform for monitoring water quality, analysing water-body conditions, forecasting water availability, collecting citizen issue reports, and sending alerts to residents in Indore.

The project combines data analytics, machine learning, forecasting, interactive maps, dashboards, reporting, and an email alert engine in a single platform.

Project Status: B.Tech Major Project (Major Project-I, Presentation-2) / Decision-Support Prototype
Deployment: Streamlit Community Cloud
Target Location: Indore, Madhya Pradesh, India

🌊 Overview

Water quality and water availability vary across locations and seasons, and they are usually viewed separately. AquaTrack brings both into one dashboard and converts raw readings into understandable risk information.

The system is built around two analytical capabilities:

Water Quality Assessment — classifies water conditions from 7 parameters using a trained Random Forest model.

Water Availability Forecasting — forecasts supply hours using a Linear Regression model.

On top of these, AquaTrack adds zone-wise analytics, interactive maps, citizen reporting of water-body issues, resident registration with email alerts, prediction history, and PDF / Excel reports.

✨ Key Features

Feature

Description

📊 Dashboard

Overview of zones, water-quality status, risks, supply information and analytics

💧 Water Quality Analysis

Analyses pH, TDS, turbidity, hardness, chloride, fluoride and nitrate

🤖 ML Risk Classification

Random Forest classifies water conditions into Safe, Moderate Risk and High Risk

📈 Availability Forecasting

Linear Regression forecasts supply hours from previous supply, rainfall and season

🗺️ Interactive Maps

Displays zones and water bodies geographically

📝 Citizen Issue Reporting

Residents report water-body issues with a photo, GPS location ("detect my location") and severity; address lookup via OpenStreetMap Nominatim

🚨 Resident Alerts

Residents register with zone and email; admins set severity and broadcast email alerts through Gmail SMTP

🛠️ Admin Panel

Admins review reports, update status and severity, and send alerts

🌐 English / Hindi Interface

Language toggle for public pages

📋 Prediction History

Stores previous analytical and forecasting results

📄 Reports

Generates PDF and Excel zone reports

🏞️ Water Bodies

Information and analysis for water bodies

🔐 Authentication

Login and role-based access (resident / admin)

☁️ Cloud Deployment

Live Streamlit Community Cloud service

🧠 Machine Learning & Analytics

Water Quality Classification

AquaTrack uses a Random Forest classifier to evaluate water-quality readings and classify them into:

🟢 Safe

🟡 Moderate Risk

🔴 High Risk

The analysis uses 7 parameters: pH, TDS, turbidity, hardness, chloride, fluoride and nitrate.

The trained model is stored in:

models/water_quality_rf.joblib

Water Availability Forecasting

Water availability (supply hours) is forecast with a Linear Regression model using previous supply hours, rainfall and season as inputs. The result is shown with a status and a rolling forecast.

The trained model is stored in:

models/availability_linear_regression.joblib

Evaluation Summary

Metric

Result

Random Forest agreement with BIS-rule labels

95.00%

F1-score (High Risk / Safe)

0.96 / 0.98

Supply forecast MAE (874 records)

0.64 h

Supply forecast R²

0.7158

Note: Quality labels are derived from BIS-rule limits on synthetic data. Laboratory validation with real field samples is future work.

🏗️ System Architecture

                    ┌─────────────────────────┐
                    │       AquaTrack         │
                    │  Streamlit Web App      │
                    │  (English / Hindi)      │
                    └────────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
       Authentication      Dashboard & UI      Reporting & Alerts
       (Resident/Admin)                        (Issues, Email, PDF/Excel)
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Application Layer     │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        Random Forest       Linear Regression   Data Analysis
        (Risk class)        (Supply forecast)
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  SQLite (10 tables) /   │
                    │  CSV files / photos     │
                    └─────────────────────────┘

   External services: Gmail SMTP (email alerts) • OpenStreetMap Nominatim (address)
                      • Map tiles • Browser Geolocation API

🛠️ Tech Stack

Application

Python

Streamlit

Data Science & Machine Learning

Pandas

NumPy

Scikit-learn

Joblib

Visualization

Plotly

Folium

Streamlit-Folium

Database

SQLite (10 tables)

Reporting & Data Files

ReportLab (PDF)

OpenPyXL (Excel)

CSV datasets

External Services

Gmail SMTP (email alerts)

OpenStreetMap Nominatim (address lookup)

Deployment

GitHub

Streamlit Community Cloud

📁 Project Structure

AquaTrack/
│
├── .streamlit/
│   └── config.toml
│
├── .vscode/
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
├── reports/
│   └── water_body_reports/
│
├── utils/
│   ├── alerts.py
│   ├── auth.py
│   └── db_functions.py
│
├── app.py
├── aquatrack.db
├── database.py
├── check_database.py
├── generate_availability.py
├── generate_water_quality.py
├── import_data.py
├── import_water_bodies.py
├── requirements.txt
└── README.md

Sensitive files such as Streamlit secrets and private credentials must stay outside the public repository.

📊 Main Modules

1. 🔐 Login & Authentication

Provides access to the application and separates protected functionality (admin panel, alert broadcast) from public pages.

2. 📊 Dashboard

Provides a high-level view of:

Total zones

Safe, moderate-risk and high-risk zones

Water supply status

Charts and key indicators

Alerts

3. 💧 Water Quality Analysis

Users enter the 7 parameters and receive a Random Forest risk class with confidence. A simple check is also available for users without a lab report.

4. 📈 Water Availability Prediction

Provides a rolling supply forecast with status to support planning and resource-management decisions.

5. 🗺️ Analytics & Mapping

Zone trends, zone comparison and interactive maps give geographical context for zones and water bodies.

6. 📝 Report Water Issue

Residents submit a water-body issue with photo, GPS location and severity. Reports are stored in the database and reviewed by admins.

7. 🚨 Resident Alerts

Residents register with their zone and email. When an admin sets a report to high severity, residents of that zone receive an email alert.

8. 🛠️ Admin Panel & Reports

Admins review reports, update status and severity, import data, broadcast alerts and export PDF / Excel reports.

9. 📜 Prediction History

Stores and presents previous predictions and analytical results for review.

🗄️ Database

AquaTrack uses SQLite with 10 tables. Main ones include:

Table

Purpose

zones

zone id, name, latitude, longitude

water_quality

zone id, date, 7 parameters

availability

zone id, date, supply hours, rainfall, season

predictions

zone id, date, risk level, forecast

users

user id, name, role

water_bodies

name, type, location, latitude / longitude

water_body_supply

water body id, zone id, colony, society

water_body_reports

status, severity, photo path, latitude / longitude

residents

zone id, name, phone, email

alerts_log

zone id, report id, resident id, channel, status

📦 Dataset & Parameters

The project uses structured water-related datasets for analysis.

Parameter

Purpose

pH

Acidity / alkalinity assessment

TDS

Total dissolved solids

Turbidity

Water clarity assessment

Hardness

Mineral concentration indicator

Chloride

Water chemistry indicator

Fluoride

Water-quality indicator

Nitrate

Water-quality indicator

The project also contains water-availability and water-body data used by the analytical modules.

🚀 Getting Started

Prerequisites

Python 3.12 recommended

Git

A modern web browser

1. Clone the repository

git clone https://github.com/jngid9575/AquaTrack.git
cd AquaTrack

2. Create a virtual environment

Windows:

python -m venv venv

3. Activate the environment

.\venv\Scripts\Activate.ps1

4. Install dependencies

pip install -r requirements.txt

5. Configure secrets

Create:

.streamlit/secrets.toml

Example:

[smtp]
host = "smtp.gmail.com"
port = 587
username = "your-email@gmail.com"
password = "YOUR_GOOGLE_APP_PASSWORD"
sender_name = "AquaTrack Support"
use_tls = true

Never commit real passwords, API keys, or private credentials to GitHub.

6. Run AquaTrack

streamlit run app.py

The application normally opens in your browser.

☁️ Deployment

AquaTrack is deployed on Streamlit Community Cloud.

Streamlit Community Cloud

The live application is available here:

🌐 AquaTrack · Streamlit

Important Deployment Notes

Keep secrets in Streamlit Community Cloud Secrets.

Do not commit .streamlit/secrets.toml.

The prototype uses SQLite and stores report photos on local disk.

A production deployment should use a managed database such as PostgreSQL and external file storage.

🔒 Security

AquaTrack is intended to demonstrate secure application practices.

Never commit:

.streamlit/secrets.toml
.env
API keys
SMTP passwords
private credentials

The public repository should contain only configuration examples and source code that is safe to share.

⚠️ Project Disclaimer

AquaTrack is a B.Tech major project and decision-support prototype.

The application is intended for demonstration, academic evaluation, data analysis and prototype-level planning. Quality labels come from BIS-rule limits on synthetic data. Predictions and classifications must not replace certified laboratory testing, official water-quality measurements, engineering assessment, or government operational decisions.

🎓 Academic Project

Project: AquaTrack — A Predictive Analytics System for Water Quality Assessment and Scarcity Forecasting in Indore

Project Type: B.Tech Major Project-I (Presentation-2, Final Project Defense)

Institute: IPS Academy, Indore — Institute of Engineering and Science

Department: Computer Science & Information Technology

Session: 2026–27 • Group: G-21

Domain:

Data Science

Machine Learning

Predictive Analytics

Water Resource Management

Web Application Development

Target City: Indore, Madhya Pradesh, India

👥 Project Team

Name

Enrollment No.

Role

Anush Parmar

0808CI231033

Project Leader

Anuj Patidar

0808CI231031

Team Member

Arpit Jangid

0808CI231038

Team Member

Bhushan Bondre

0808CI231058

Team Member

Project Guide: Mr. Sumit Kumar

🌐 Links

Source Code

💻 AquaTrack · GitHub

Live Application

🌐 AquaTrack · Streamlit

📌 Future Scope

Validation with lab-tested water samples

Real-time IoT water sensors

Live municipal water-supply APIs

PostgreSQL production database and scheduled model retraining

Automated SMS alerts (MSG91 / Twilio) and WhatsApp alerts

Mobile application

Advanced time-series forecasting

GIS-based spatial analytics

Satellite and remote-sensing integration

Automated anomaly detection

More Indore zones and a municipal administration portal

⭐ Why AquaTrack?

AquaTrack brings together water-quality analysis, machine learning, forecasting, mapping, citizen reporting, alerts and reports in one platform.

The goal is to move from simply viewing historical water information toward a more predictive and proactive approach to water-resource management.

📄 License

This project is developed as an academic B.Tech project. All rights and usage terms should be determined by the project team.
