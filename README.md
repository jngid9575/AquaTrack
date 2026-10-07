[README.md](https://github.com/user-attachments/files/33155623/README.md)
# 💧 AquaTrack

### A Predictive Analytics System for Water Quality Assessment & Scarcity Forecasting in Indore

AquaTrack is a **Streamlit-based water intelligence and decision-support platform** designed for monitoring water quality, analysing water-body conditions, forecasting water availability, and generating alerts for water-related risks in **Indore**.

The project combines data analytics, machine learning, forecasting, interactive maps, dashboards, reporting, and alert management into a single platform.

> **Project Status:** B.Tech Final-Year Project / Decision-Support Prototype  
> **Deployment:** Streamlit application deployed on Render  
> **Target Location:** Indore, Madhya Pradesh, India

---

## 🌊 Overview

Water quality and water availability can vary across locations and seasons. AquaTrack provides a centralized dashboard where water-related data can be analysed and converted into understandable risk information.

The system is designed around two major analytical capabilities:

1. **Water Quality Assessment** — classifies water conditions using a trained Random Forest model.
2. **Water Availability Forecasting** — forecasts future availability using regression and time-series forecasting techniques.

AquaTrack also provides dashboards, interactive visualizations, water-body information, alerts, prediction history, and report generation.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Dashboard | Overview of zones, water-quality status, risks, supply information and analytics |
| 💧 Water Quality Analysis | Analyses parameters such as pH, TDS, turbidity, hardness, chloride, fluoride and nitrate |
| 🤖 ML Risk Classification | Random Forest model classifies water conditions into Safe, Moderate Risk and High Risk |
| 📈 Availability Forecasting | Predicts future water availability using regression and forecasting methods |
| 🗺️ Interactive Maps | Displays zone and water-body information geographically |
| 🚨 Alert System | Generates alerts for important water-quality and availability conditions |
| 📋 Prediction History | Maintains previous analytical and forecasting results |
| 📄 Reports | Supports generation of analytical reports |
| 🏞️ Water Bodies | Provides information and analysis for water bodies |
| 🔐 Authentication | Login and role-based access for application users |
| 📱 Responsive Dashboard | Streamlit-based interface designed for practical monitoring |
| ☁️ Cloud Deployment | Application deployed as a live Streamlit service on Render |

---

## 🧠 Machine Learning & Analytics

### Water Quality Classification

AquaTrack uses a **Random Forest classifier** to evaluate water-quality readings and classify them into:

- 🟢 **Safe**
- 🟡 **Moderate Risk**
- 🔴 **High Risk**

The analysis considers water-quality parameters including:

- pH
- TDS
- Turbidity
- Hardness
- Chloride
- Fluoride
- Nitrate

The trained model is stored in:

```text
models/water_quality_rf.joblib
```

### Water Availability Forecasting

The project includes forecasting functionality for future water availability.

The project architecture includes:

- Linear Regression
- Prophet-based forecasting

The trained regression model is stored in:

```text
models/availability_linear_regression.joblib
```

Forecasting can be used for future planning horizons such as:

- Next month
- Next 3 months
- Next 6 months

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │       AquaTrack         │
                    │    Streamlit Web App    │
                    └────────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
       Authentication      Dashboard & UI      Data Management
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Analytics Layer     │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        Random Forest       Regression/        Data Analysis
        Classification      Forecasting
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   SQLite / Data Files   │
                    └─────────────────────────┘
```

---

## 🛠️ Tech Stack

### Application

- **Python**
- **Streamlit**

### Data Science & Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Joblib
- Prophet

### Visualization

- Plotly
- Folium
- Streamlit-Folium

### Database

- SQLite

### Reporting & Data Files

- ReportLab
- OpenPyXL
- CSV datasets

### Deployment

- GitHub
- Render

---

## 📁 Project Structure

```text
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
```

> Sensitive files such as Streamlit secrets and private credentials should remain outside the public repository.

---

## 📊 Main Modules

### 1. 🔐 Login & Authentication

Provides secure access to the application and separates protected application functionality from the login interface.

### 2. 📊 Dashboard

Provides a high-level view of:

- Total zones
- Safe zones
- Moderate-risk zones
- High-risk zones
- Water supply status
- Charts
- Alerts
- Key indicators

### 3. 💧 Water Quality Analysis

Users can analyse water-quality readings and obtain machine-learning-based risk classifications.

### 4. 📈 Water Availability Prediction

Provides future availability predictions to support planning and resource-management decisions.

### 5. 🗺️ Water & Zone Mapping

Interactive maps provide geographical context for zones and water bodies.

### 6. 🚨 Alerts

The alert system highlights important conditions requiring attention.

### 7. 📜 Prediction History

Stores and presents previous predictions and analytical results for review.

### 8. 📄 Reports

The application can generate reports for analytical and project documentation purposes.

---

## 📦 Dataset & Parameters

The project uses structured water-related datasets containing measurements and records for analysis.

Important water-quality parameters include:

| Parameter | Purpose |
|---|---|
| pH | Acidity/alkalinity assessment |
| TDS | Total dissolved solids |
| Turbidity | Water clarity assessment |
| Hardness | Mineral concentration indicator |
| Chloride | Water chemistry indicator |
| Fluoride | Water-quality indicator |
| Nitrate | Water-quality indicator |

The project also contains water-availability and water-body data used by the analytical modules.

---

## 🚀 Getting Started

### Prerequisites

Install:

- Python 3.12 recommended
- Git
- A modern web browser

### 1. Clone the repository

```bash
git clone https://github.com/jngid9575/AquaTrack.git
cd AquaTrack
```

### 2. Create a virtual environment

Windows:

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

Example:

```toml
[smtp]
host = "smtp.gmail.com"
port = 587
username = "your-email@gmail.com"
password = "YOUR_GOOGLE_APP_PASSWORD"
sender_name = "AquaTrack Support"
use_tls = true
```

**Never commit real passwords, API keys, or private credentials to GitHub.**

### 6. Run AquaTrack

```powershell
streamlit run app.py
```

The application will normally open in your browser.

---

## ☁️ Deployment

AquaTrack is configured for deployment as a Streamlit web service on **Render**.

### Render Build Command

```bash
pip install -r requirements.txt
```

### Render Start Command

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

### Important Deployment Notes

- Keep secrets in Render Secret Files / environment configuration.
- Do not commit `.streamlit/secrets.toml`.
- The application uses SQLite for the project prototype.
- A persistent production deployment should use a managed database such as PostgreSQL instead of relying on a local SQLite file.

---

## 🔒 Security

AquaTrack is intended to demonstrate secure application practices.

### Never commit:

```text
.streamlit/secrets.toml
.env
API keys
SMTP passwords
private credentials
```

The public repository should contain only configuration examples and source code that is safe to share.

---

## ⚠️ Project Disclaimer

AquaTrack is a **B.Tech final-year project and decision-support prototype**.

The application is intended for demonstration, academic evaluation, data analysis and prototype-level planning. Predictions and classifications should not be treated as a replacement for certified laboratory testing, official water-quality measurements, engineering assessment, or government operational decisions.

---

## 🎓 Academic Project

**Project:** AquaTrack — A Predictive Analytics System for Water Quality Assessment and Scarcity Forecasting in Indore

**Project Type:** B.Tech Major Project

**Domain:**

- Data Science
- Machine Learning
- Predictive Analytics
- Water Resource Management
- Web Application Development

**Target City:** Indore, Madhya Pradesh, India

---

## 👥 Project Team

- **Anush Parmar** — Project Leader
- **Anuj Patidar** — Team Member
- **Arpit Jangid** — Team Member

**Project Guide:** Mr. Sumit Kumar

---

## 🌐 Links

### Source Code

https://github.com/jngid9575/AquaTrack

### Live Application

Add the Render URL here after deployment:

```text
https://YOUR-AQUATRACK-RENDER-URL.onrender.com
```

---

## 📌 Future Scope

Possible future improvements include:

- Real-time IoT water sensors
- Live municipal water-supply APIs
- PostgreSQL production database
- Automated SMS and WhatsApp alerts
- Mobile application
- Advanced time-series forecasting
- GIS-based spatial analytics
- Satellite and remote-sensing integration
- Automated anomaly detection
- Role-based municipal administration portal
- Real-time monitoring dashboards

---

## ⭐ Why AquaTrack?

AquaTrack brings together **water-quality analysis, machine learning, forecasting, mapping, alerts and reporting** in one platform.

The goal is to move from simply viewing historical water information toward a more **predictive and proactive approach to water-resource management**.

---

## 📄 License

This project is developed as an academic B.Tech project. All rights and usage terms should be determined by the project team.
