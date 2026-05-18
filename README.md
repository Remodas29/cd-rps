# Chronic Disease Risk Prediction System (CD-RPS)

**IGNOU BCA Final Year Project**
**Student:** Prosun Kumar Das
**Enrolment:** 2300990309
**Study Centre:** 1314
**Programme:** BCA
**Guide:** Mr. Nirupam Singh
**School:** School of Computer and Information Sciences, IGNOU, New Delhi

---

## Project Overview

The Chronic Disease Risk Prediction System (CD-RPS) is an AI-powered web application that predicts a patient's risk of developing 9 chronic diseases from a single set of clinical vitals. It uses trained Random Forest machine learning models, a SQLite database, a Flask REST API backend, and a responsive HTML/CSS/JS frontend.

---

## Diseases Covered

| # | Disease |
|---|---|
| 1 | Type 2 Diabetes |
| 2 | Cardiovascular Disease |
| 3 | Chronic Kidney Disease |
| 4 | Stroke |
| 5 | High Blood Pressure (Hypertension) |
| 6 | High Cholesterol |
| 7 | Obesity |
| 8 | Coronary Heart Disease |
| 9 | COPD (Chronic Obstructive Pulmonary Disease) |

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.x |
| Machine Learning | scikit-learn (Random Forest Classifier) |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Database | SQLite (via DB Browser for SQLite) |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js |
| Testing | Python unittest |

---

## Project Structure

```
cd_rps/
├── database/
│   ├── cd_rps.db              — SQLite database file
│   ├── db_connection.py       — Database connection and CRUD helpers
│   └── schema_sqlite.sql      — SQL schema (reference)
│
├── model/
│   ├── train_all_models.py    — Trains all 9 disease models
│   ├── model_1.pkl            — Type 2 Diabetes model
│   ├── model_2.pkl            — Cardiovascular Disease model
│   ├── model_3.pkl            — Chronic Kidney Disease model
│   ├── model_4.pkl            — Stroke model
│   ├── model_5.pkl            — High Blood Pressure model
│   ├── model_6.pkl            — High Cholesterol model
│   ├── model_7.pkl            — Obesity model
│   ├── model_8.pkl            — Coronary Heart Disease model
│   ├── model_9.pkl            — COPD model
│   ├── scaler_1.pkl           — Scaler for model 1
│   └── scaler_2.pkl ... 9     — Scalers for models 2–9
│
├── backend/
│   └── app.py                 — Flask REST API (6 routes)
│
├── frontend/
│   ├── index.html             — Login and Registration page
│   └── predict.html           — Vitals form, Results, History
│
├── tests/
│   └── test_cd_rps.py         — 45 unit and integration tests
│
├── requirements.txt           — Python dependencies
├── run.bat                    — One-click launcher (Windows)
└── README.md                  — This file
```

---

## Quick Start (Windows)

### Method 1 — Double-click launcher (easiest)

1. Make sure Python is installed (https://www.python.org/downloads/)
2. Double-click **`run.bat`** in the `cd_rps` folder
3. The browser opens automatically at `http://127.0.0.1:5000`

### Method 2 — Manual (VS Code terminal)

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Train all 9 AI models (run once):**
```bash
cd model
python train_all_models.py
cd ..
```

**Step 3 — Start the Flask server:**
```bash
cd backend
python app.py
```

**Step 4 — Open in browser:**
```
http://127.0.0.1:5000
```

---

## Running the Tests

```bash
cd "project Final\cd_rps"
python tests/test_cd_rps.py
```

Expected output: **45 tests, 0 failures, 0 errors**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status and model count |
| POST | `/api/register` | Register a new patient account |
| POST | `/api/login` | Login and get patient session |
| POST | `/api/predict_all` | Submit vitals, get all 9 disease predictions |
| GET | `/api/history/<id>` | Get full prediction history for a patient |
| GET | `/api/diseases` | List all 9 diseases |

---

## Database Tables

| Table | Purpose |
|---|---|
| PATIENT | Stores patient registration details |
| VITALS_RECORD | Stores clinical parameters submitted per session |
| DISEASE | Reference table for all 9 chronic diseases |
| PREDICTION_RESULT | Stores AI model output for each prediction |

---

## Input Parameters

| Parameter | Unit | Normal Range |
|---|---|---|
| Age | years | — |
| BMI | kg/m² | 18.5 – 24.9 |
| Glucose Level | mg/dL | 70 – 100 (fasting) |
| Systolic Blood Pressure | mmHg | Below 120 |
| Diastolic Blood Pressure | mmHg | Below 80 |
| Cholesterol | mg/dL | Below 200 |
| Insulin Level | µU/mL | 2 – 25 (fasting) |
| Skin Thickness | mm | Triceps skinfold |
| Pregnancies | count | 0 (enter 0 if male) |
| Diabetes Pedigree Function | 0–2.5 | Family history score |

---

## References

- Pima Indians Diabetes Dataset — Jason Brownlee / UCI ML Repository
- Framingham Heart Study Dataset — Kaggle
- Python Documentation — https://docs.python.org
- Flask Documentation — https://flask.palletsprojects.com
- scikit-learn Documentation — https://scikit-learn.org
- Bootstrap 5 — https://getbootstrap.com
- Chart.js — https://www.chartjs.org
- Stack Overflow, GeeksforGeeks, GitHub

---

*IGNOU BCA Final Project | School of Computer and Information Sciences | 2300990309*
