# Chronic Disease Risk Prediction System (CD-RPS)

An AI-powered full-stack healthcare web application that predicts the risk of multiple chronic diseases using Machine Learning models trained on clinical health parameters.

## Project Overview

The Chronic Disease Risk Prediction System (CD-RPS) is designed to assist in early identification of chronic disease risks through predictive analytics and machine learning. The system accepts patient vital parameters and simultaneously predicts the risk levels for nine chronic diseases.

The project was developed as part of the IGNOU BCA Final Year Project (BCSP-064).

---

# Features

* Predicts risk for 9 chronic diseases simultaneously
* Machine Learning powered prediction engine
* Flask REST API backend
* SQLite relational database integration
* Interactive frontend dashboard
* Patient registration and login system
* Historical prediction tracking
* Chart.js visual analytics
* Personalized health recommendations
* Responsive Bootstrap-based UI

---

# Diseases Covered

1. Type 2 Diabetes
2. Cardiovascular Disease
3. Chronic Kidney Disease
4. Stroke
5. Hypertension
6. High Cholesterol
7. Obesity
8. Coronary Heart Disease
9. COPD

---

# Tech Stack

## Backend

* Python
* Flask
* Flask-CORS

## Machine Learning

* scikit-learn
* Random Forest Classifier
* NumPy
* Pandas

## Frontend

* HTML5
* CSS3
* JavaScript
* Bootstrap 5
* Chart.js

## Database

* SQLite

---

# Project Structure

```text
cd-rps/
│
├── backend/
│   └── app.py
│
├── database/
│   ├── cd_rps.db
│   ├── db_connection.py
│   └── schema_sqlite.sql
│
├── frontend/
│   ├── index.html
│   └── predict.html
│
├── model/
│   ├── train_all_models.py
│   ├── model_1.pkl
│   ├── scaler_1.pkl
│   └── ...
│
├── tests/
│   └── test_cd_rps.py
│
├── requirements.txt
├── README.md
└── run.bat
```

---

# Installation Guide

## Step 1 — Clone Repository

```bash
git clone https://github.com/Remodas29/cd-rps.git
```

## Step 2 — Open Project Folder

```bash
cd cd-rps
```

## Step 3 — Create Virtual Environment

```bash
python -m venv venv
```

## Step 4 — Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Machine Learning Training

```bash
python model/train_all_models.py
```

This generates:

* model_1.pkl → model_9.pkl
* scaler_1.pkl → scaler_9.pkl

---

# Run Flask Application

```bash
python backend/app.py
```

---

# Open Application

Open browser:

```text
http://127.0.0.1:5000
```

---

# Database Setup

Open:

```text
database/schema_sqlite.sql
```

Execute the SQL using:

* DB Browser for SQLite

IMPORTANT FIX:

Replace:

```sql
RecordID INTEGER NOT NULL UNIQUE
```

with:

```sql
RecordID INTEGER NOT NULL
```

---

# Machine Learning Architecture

The system uses:

* Random Forest Classifier
* StandardScaler normalization
* Multi-model prediction architecture
* Probability-based classification

Each disease model:

* Accepts 8 clinical features
* Returns probability score
* Produces High Risk / Low Risk classification

---

# Input Parameters

The system accepts:

* Age
* BMI
* Glucose Level
* Blood Pressure
* Cholesterol
* Insulin Level
* Skin Thickness
* Pregnancies
* Diabetes Pedigree Function

---

# Testing

The project includes:

* Unit Testing
* Integration Testing
* System Testing

Implemented using:

```text
Python unittest
```

---

# Screenshots

## Login Page

<img width="962" height="607" alt="image" src="https://github.com/user-attachments/assets/2c3f5df3-4ca5-40f1-bc28-3b3f3eb33e04" />
<img width="935" height="553" alt="image" src="https://github.com/user-attachments/assets/93a5ea8c-99cb-4cbb-bfc7-7b9301121d1b" />



## Prediction Dashboard

<img width="956" height="559" alt="image" src="https://github.com/user-attachments/assets/2d0f455a-7cd7-444e-bf53-652777037b51" />
<img width="959" height="566" alt="image" src="https://github.com/user-attachments/assets/6ce7a0cf-fb88-4184-9e1e-ae1960fa710f" />


## Risk Analysis Charts

   
Figure UI Wireframe — screenshot of the Detail Modal with Charts
<img width="978" height="569" alt="image" src="https://github.com/user-attachments/assets/631f7b2e-853d-40b4-91e6-97adea963f7e" />
<img width="428" height="440" alt="image" src="https://github.com/user-attachments/assets/8571af00-88c4-4961-a6b1-80ca26cd5097" /> <img width="490" height="441" alt="image" src="https://github.com/user-attachments/assets/058aac90-57ee-4f54-baeb-a2f9f81bd052" />

<img width="922" height="902" alt="image" src="https://github.com/user-attachments/assets/1036e8bf-602a-4f3f-8493-df32f5b9c82d" />
 Prediction History Panel — Showing sessions grouped by date with High Risk/Low Risk counts, with one session expanded to show all 9 disease results<img width="971" height="528" alt="image" src="https://github.com/user-attachments/assets/c51f6c7a-5f92-4d0c-b080-0a4d643bbafb" />



---

# Future Improvements

* JWT Authentication
* Cloud Deployment
* Docker Support
* PostgreSQL Migration
* Real-time API Integration
* Advanced ML Models
* Mobile Application
* Doctor/Admin Dashboard

---

# Author

## Prosun Kumar Das


GitHub:
https://github.com/Remodas29

---

# License

This project is developed for educational and academic purposes.

---

# Disclaimer

This project is intended for educational and research purposes only. It should not be considered a substitute for professional medical diagnosis or treatment.
