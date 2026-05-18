# ============================================================
# CD-RPS Project | database/db_connection.py
# SQLite version — works with DB Browser for SQLite
# ============================================================

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cd_rps.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def insert_patient(name, age, gender, email, password_hash, contact=None):
    sql = """INSERT INTO PATIENT (Name, Age, Gender, Email, PasswordHash, ContactNumber)
             VALUES (?, ?, ?, ?, ?, ?)"""
    with get_connection() as conn:
        cur = conn.execute(sql, (name, age, gender, email, password_hash, contact))
        conn.commit()
        return cur.lastrowid


def get_patient_by_email(email):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM PATIENT WHERE Email = ?", (email,)).fetchone()
        return dict(row) if row else None


def get_patient_by_id(patient_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM PATIENT WHERE PatientID = ?", (patient_id,)).fetchone()
        return dict(row) if row else None


def insert_vitals_record(patient_id, bmi, glucose, systolic_bp,
                          diastolic_bp=None, cholesterol=None,
                          insulin=None, skin_thickness=None, pregnancies=0):
    sql = """INSERT INTO VITALS_RECORD
                (PatientID, BMI, GlucoseLevel, SystolicBP, DiastolicBP,
                 Cholesterol, InsulinLevel, SkinThickness, Pregnancies)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    with get_connection() as conn:
        cur = conn.execute(sql, (patient_id, bmi, glucose, systolic_bp,
                                  diastolic_bp, cholesterol, insulin,
                                  skin_thickness, pregnancies))
        conn.commit()
        return cur.lastrowid


def insert_prediction_result(record_id, disease_id, risk_score,
                               classification, raw_probability,
                               model_used='Random Forest Classifier',
                               recommendations=''):
    sql = """INSERT INTO PREDICTION_RESULT
                (RecordID, DiseaseID, RiskScore, Classification,
                 RawProbability, MLModelUsed, Recommendations)
             VALUES (?, ?, ?, ?, ?, ?, ?)"""
    with get_connection() as conn:
        cur = conn.execute(sql, (record_id, disease_id, risk_score,
                                  classification, raw_probability,
                                  model_used, recommendations))
        conn.commit()
        return cur.lastrowid


def get_patient_history(patient_id):
    sql = """SELECT pr.ResultID, pr.PredictionDate, pr.RiskScore,
                    pr.Classification, pr.MLModelUsed, pr.Recommendations,
                    d.DiseaseName, vr.BMI, vr.GlucoseLevel, vr.SystolicBP, vr.Cholesterol
             FROM PREDICTION_RESULT pr
             JOIN VITALS_RECORD vr ON pr.RecordID  = vr.RecordID
             JOIN DISEASE        d  ON pr.DiseaseID = d.DiseaseID
             WHERE vr.PatientID = ?
             ORDER BY pr.PredictionDate DESC"""
    with get_connection() as conn:
        rows = conn.execute(sql, (patient_id,)).fetchall()
        return [dict(r) for r in rows]


def get_all_diseases():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM DISEASE").fetchall()
        return [dict(r) for r in rows]


if __name__ == '__main__':
    try:
        conn = get_connection()
        print(f'✅ Database connection successful!')
        print(f'   Database file: {DB_PATH}')
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        print(f'   Tables found: {[t[0] for t in tables]}')
        diseases = conn.execute("SELECT DiseaseName FROM DISEASE").fetchall()
        print(f'   Diseases seeded: {[d[0] for d in diseases]}')
        conn.close()
        print('\n✅ Phase 1 database setup is complete and working!')
    except Exception as e:
        print(f'❌ Error: {e}')
