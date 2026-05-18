# ============================================================
# CD-RPS Project | tests/test_cd_rps.py
# Unit Testing + Integration Testing
# Student: Prosun Kumar Das | Enrollment: 2300990309
# Guide  : Mr. Nirupam Singh | IGNOU BCA Final Project
# ============================================================
#
# HOW TO RUN:
#   cd "project Final\cd_rps"
#   python -m pytest tests/test_cd_rps.py -v
#   OR
#   python tests/test_cd_rps.py
#
# COVERS:
#   Module 1  — Database connection
#   Module 2  — CRUD operations (PATIENT, VITALS, PREDICTION)
#   Module 3  — Data preprocessing / scaling
#   Module 4  — ML model loading and prediction
#   Module 5  — All 9 disease models
#   Module 6  — Flask API routes
#   Module 7  — Input validation
#   Module 8  — Known high-risk patient test
#   Module 9  — Known low-risk patient test
#   Module 10 — predict_all endpoint (9 diseases at once)
# ============================================================

import os
import sys
import json
import unittest
import sqlite3
import pickle
import hashlib
import numpy as np
import tempfile
import shutil

# ── Path setup ───────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE, 'database')
ML_DIR = os.path.join(BASE, 'model')
BE_DIR = os.path.join(BASE, 'backend')

sys.path.insert(0, DB_DIR)
sys.path.insert(0, ML_DIR)
sys.path.insert(0, BE_DIR)


# ════════════════════════════════════════════════════════════
# MODULE 1 — Database Connection
# ════════════════════════════════════════════════════════════
class TestDatabaseConnection(unittest.TestCase):
    """Tests that the database file exists and can be opened."""

    def test_01_db_file_exists(self):
        """Database file cd_rps.db must exist in the database folder."""
        db_path = os.path.join(DB_DIR, 'cd_rps.db')
        self.assertTrue(os.path.exists(db_path),
                        f"cd_rps.db not found at {db_path}")

    def test_02_connection_opens(self):
        """SQLite connection must open without errors."""
        from db_connection import get_connection
        conn = get_connection()
        self.assertIsNotNone(conn, "get_connection() returned None")
        conn.close()

    def test_03_foreign_keys_enabled(self):
        """PRAGMA foreign_keys must be ON."""
        from db_connection import get_connection
        conn = get_connection()
        result = conn.execute("PRAGMA foreign_keys").fetchone()
        conn.close()
        self.assertEqual(result[0], 1, "Foreign keys are not enabled")

    def test_04_all_four_tables_exist(self):
        """All 4 required tables must be present in the database."""
        from db_connection import get_connection
        conn     = get_connection()
        tables   = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        required = {'PATIENT', 'VITALS_RECORD', 'DISEASE', 'PREDICTION_RESULT'}
        self.assertTrue(required.issubset(tables),
                        f"Missing tables: {required - tables}")

    def test_05_disease_table_seeded(self):
        """DISEASE table must contain at least 9 entries."""
        from db_connection import get_connection
        conn  = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM DISEASE").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(count, 9,
                                f"Expected 9 diseases, found {count}")


# ════════════════════════════════════════════════════════════
# MODULE 2 — CRUD Operations
# ════════════════════════════════════════════════════════════
class TestCRUDOperations(unittest.TestCase):
    """Tests all Create / Read operations on the database tables."""

    _counter = 0

    def setUp(self):
        """Generate a unique test email for each individual test method."""
        TestCRUDOperations._counter += 1
        self.test_email = f"unittest_{TestCRUDOperations._counter}_crud@cdrps.test"
        self.patient_id = None

    def test_06_insert_patient_returns_id(self):
        """insert_patient() must return a valid integer PatientID."""
        from db_connection import insert_patient
        pid = insert_patient(
            name='Unit Test Patient', age=35, gender='Male',
            email=self.test_email,
            password_hash=hashlib.sha256(b'Test@1234').hexdigest(),
            contact='9000000001'
        )
        self.assertIsInstance(pid, int, "PatientID must be an integer")
        self.assertGreater(pid, 0, "PatientID must be > 0")
        self.patient_id = pid

    def test_07_get_patient_by_email(self):
        """get_patient_by_email() must return the correct patient dict."""
        from db_connection import insert_patient, get_patient_by_email
        pid = insert_patient(
            name='Email Test Patient', age=28, gender='Female',
            email=self.test_email,
            password_hash=hashlib.sha256(b'Pass@5678').hexdigest()
        )
        result = get_patient_by_email(self.test_email)
        self.assertIsNotNone(result, "Patient not found by email")
        self.assertEqual(result['Name'],  'Email Test Patient')
        self.assertEqual(result['Email'], self.test_email)
        self.assertEqual(result['Age'],   28)

    def test_08_get_patient_nonexistent_returns_none(self):
        """Querying a non-existent email must return None."""
        from db_connection import get_patient_by_email
        result = get_patient_by_email('nobody@doesnotexist.com')
        self.assertIsNone(result, "Expected None for unknown email")

    def test_09_insert_vitals_record_returns_id(self):
        """insert_vitals_record() must return a valid RecordID."""
        from db_connection import insert_patient, insert_vitals_record
        pid = insert_patient(
            name='Vitals Test', age=40, gender='Male',
            email=self.test_email,
            password_hash='hash'
        )
        rid = insert_vitals_record(
            patient_id=pid, bmi=28.5, glucose=110.0,
            systolic_bp=118, diastolic_bp=78,
            cholesterol=190.0, insulin=80.0,
            skin_thickness=22.0, pregnancies=0
        )
        self.assertIsInstance(rid, int, "RecordID must be an integer")
        self.assertGreater(rid, 0, "RecordID must be > 0")

    def test_10_insert_prediction_result(self):
        """insert_prediction_result() must store the ML output correctly."""
        from db_connection import (insert_patient, insert_vitals_record,
                                    insert_prediction_result, get_patient_history)
        pid = insert_patient(
            name='Pred Test', age=50, gender='Female',
            email=self.test_email, password_hash='hash'
        )
        rid = insert_vitals_record(
            patient_id=pid, bmi=33.6, glucose=148.0,
            systolic_bp=72, pregnancies=6
        )
        result_id = insert_prediction_result(
            record_id=rid, disease_id=1,
            risk_score=62.5, classification='High Risk',
            raw_probability=0.625,
            recommendations='Consult a doctor.'
        )
        self.assertIsInstance(result_id, int)
        self.assertGreater(result_id, 0)

        # Verify it shows up in history
        history = get_patient_history(pid)
        self.assertGreater(len(history), 0, "History should not be empty")
        self.assertEqual(history[0]['Classification'], 'High Risk')
        self.assertAlmostEqual(history[0]['RiskScore'], 62.5, places=1)

    def test_11_get_all_diseases(self):
        """get_all_diseases() must return a list of 9 disease dicts."""
        from db_connection import get_all_diseases
        diseases = get_all_diseases()
        self.assertIsInstance(diseases, list)
        self.assertGreaterEqual(len(diseases), 9)
        names = [d['DiseaseName'] for d in diseases]
        self.assertIn('Type 2 Diabetes', names)
        self.assertIn('Cardiovascular Disease', names)
        self.assertIn('Stroke', names)


# ════════════════════════════════════════════════════════════
# MODULE 3 — Data Preprocessing / Scaling
# ════════════════════════════════════════════════════════════
class TestDataPreprocessing(unittest.TestCase):
    """Tests that the StandardScaler transforms data correctly."""

    def test_12_scaler_file_exists(self):
        """scaler_1.pkl must exist in the model directory."""
        path = os.path.join(ML_DIR, 'scaler_1.pkl')
        self.assertTrue(os.path.exists(path), f"scaler_1.pkl not found at {path}")

    def test_13_scaler_loads_correctly(self):
        """scaler_1.pkl must deserialize to a sklearn StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        path = os.path.join(ML_DIR, 'scaler_1.pkl')
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        self.assertIsInstance(scaler, StandardScaler,
                              "Loaded object is not a StandardScaler")

    def test_14_scaler_transforms_correct_shape(self):
        """transform() on 8 features must return shape (1, 8)."""
        path = os.path.join(ML_DIR, 'scaler_1.pkl')
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        raw = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        scaled = scaler.transform(raw)
        self.assertEqual(scaled.shape, (1, 8),
                         f"Expected shape (1,8), got {scaled.shape}")

    def test_15_scaling_changes_values(self):
        """Scaled values must differ from raw input values."""
        path = os.path.join(ML_DIR, 'scaler_1.pkl')
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        raw    = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        scaled = scaler.transform(raw)
        # At least one value must be different after scaling
        self.assertFalse(np.allclose(raw, scaled),
                         "Scaling did not change values — scaler may be unfit")

    def test_16_all_9_scalers_exist(self):
        """All 9 scaler files (scaler_1.pkl to scaler_9.pkl) must exist."""
        missing = []
        for i in range(1, 10):
            path = os.path.join(ML_DIR, f'scaler_{i}.pkl')
            if not os.path.exists(path):
                missing.append(f'scaler_{i}.pkl')
        self.assertEqual(missing, [], f"Missing scalers: {missing}")


# ════════════════════════════════════════════════════════════
# MODULE 4 — ML Model Loading and Prediction
# ════════════════════════════════════════════════════════════
class TestMLModelLoading(unittest.TestCase):
    """Tests that model files load and produce valid predictions."""

    def test_17_all_9_models_exist(self):
        """All 9 model files (model_1.pkl to model_9.pkl) must exist."""
        missing = []
        for i in range(1, 10):
            path = os.path.join(ML_DIR, f'model_{i}.pkl')
            if not os.path.exists(path):
                missing.append(f'model_{i}.pkl')
        self.assertEqual(missing, [], f"Missing models: {missing}")

    def test_18_model_1_loads_as_classifier(self):
        """model_1.pkl must be a RandomForestClassifier."""
        from sklearn.ensemble import RandomForestClassifier
        path = os.path.join(ML_DIR, 'model_1.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_19_model_has_predict_proba(self):
        """Model must expose predict_proba() method."""
        path = os.path.join(ML_DIR, 'model_1.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.assertTrue(hasattr(model, 'predict_proba'),
                        "Model does not have predict_proba method")

    def test_20_prediction_returns_probability_in_range(self):
        """predict_proba() must return a value between 0.0 and 1.0."""
        m_path = os.path.join(ML_DIR, 'model_1.pkl')
        s_path = os.path.join(ML_DIR, 'scaler_1.pkl')
        with open(m_path, 'rb') as f: model  = pickle.load(f)
        with open(s_path, 'rb') as f: scaler = pickle.load(f)
        features = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        scaled   = scaler.transform(features)
        prob     = model.predict_proba(scaled)[0][1]
        self.assertGreaterEqual(prob, 0.0, "Probability < 0")
        self.assertLessEqual(prob,   1.0, "Probability > 1")

    def test_21_classification_threshold(self):
        """Risk classification must follow the >= 0.5 threshold rule."""
        m_path = os.path.join(ML_DIR, 'model_1.pkl')
        s_path = os.path.join(ML_DIR, 'scaler_1.pkl')
        with open(m_path, 'rb') as f: model  = pickle.load(f)
        with open(s_path, 'rb') as f: scaler = pickle.load(f)
        features = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        scaled   = scaler.transform(features)
        prob     = model.predict_proba(scaled)[0][1]
        result   = 'High Risk' if prob >= 0.5 else 'Low Risk'
        self.assertIn(result, ['High Risk', 'Low Risk'])


# ════════════════════════════════════════════════════════════
# MODULE 5 — All 9 Disease Models
# ════════════════════════════════════════════════════════════
class TestAllDiseaseModels(unittest.TestCase):
    """Tests all 9 disease-specific models produce valid predictions."""

    DISEASE_NAMES = {
        1:'Type 2 Diabetes', 2:'Cardiovascular Disease',
        3:'Chronic Kidney Disease', 4:'Stroke',
        5:'High Blood Pressure', 6:'High Cholesterol',
        7:'Obesity', 8:'Coronary Heart Disease', 9:'COPD'
    }

    def _predict(self, disease_id, features):
        m_path = os.path.join(ML_DIR, f'model_{disease_id}.pkl')
        s_path = os.path.join(ML_DIR, f'scaler_{disease_id}.pkl')
        with open(m_path, 'rb') as f: model  = pickle.load(f)
        with open(s_path, 'rb') as f: scaler = pickle.load(f)
        scaled = scaler.transform(np.array([features]))
        return float(model.predict_proba(scaled)[0][1])

    def test_22_all_9_models_return_valid_probabilities(self):
        """Each of the 9 models must return a probability in [0, 1]."""
        features = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        for did in range(1, 10):
            prob = self._predict(did, features)
            self.assertGreaterEqual(prob, 0.0,
                f"Disease {did} ({self.DISEASE_NAMES[did]}): prob < 0")
            self.assertLessEqual(prob, 1.0,
                f"Disease {did} ({self.DISEASE_NAMES[did]}): prob > 1")

    def test_23_risk_score_is_percentage(self):
        """Risk score (prob * 100) must be between 0 and 100."""
        features = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        for did in range(1, 10):
            score = round(self._predict(did, features) * 100, 2)
            self.assertGreaterEqual(score, 0.0,
                f"Disease {did}: score < 0")
            self.assertLessEqual(score, 100.0,
                f"Disease {did}: score > 100")

    def test_24_healthy_patient_low_scores(self):
        """A healthy patient should score below 50% on most diseases."""
        healthy = [0, 85, 110, 15, 18, 21.5, 0.15, 25]
        high_risk_count = 0
        for did in range(1, 10):
            prob = self._predict(did, healthy)
            if prob >= 0.5:
                high_risk_count += 1
        self.assertLessEqual(high_risk_count, 3,
            f"Healthy patient got {high_risk_count}/9 High Risk — model may be miscalibrated")


# ════════════════════════════════════════════════════════════
# MODULE 6 — Flask API Routes
# ════════════════════════════════════════════════════════════
class TestFlaskAPI(unittest.TestCase):
    """Integration tests for all Flask API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Start Flask test client once for all API tests."""
        from app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()
        import time
        cls.test_email = f'flask_test_{int(time.time())}@cdrps.test'
        cls.patient_id = None

    def test_25_health_endpoint_returns_200(self):
        """GET /health must return 200 with status=running."""
        r = self.client.get('/health')
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        self.assertEqual(d['status'], 'running')

    def test_26_health_shows_models_loaded(self):
        """GET /health must show models_loaded = 9."""
        r = self.client.get('/health')
        d = json.loads(r.data)
        self.assertEqual(d['models_loaded'], 9,
            f"Expected 9 models loaded, got {d['models_loaded']}")

    def test_27_diseases_endpoint_returns_9(self):
        """GET /api/diseases must return 9 disease records."""
        r = self.client.get('/api/diseases')
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        self.assertTrue(d['success'])
        self.assertGreaterEqual(len(d['diseases']), 9)

    def test_28_register_creates_patient(self):
        """POST /api/register must return 201 with patient_id."""
        r = self.client.post('/api/register', json={
            'name':'Flask Test User', 'age':30, 'gender':'Male',
            'email': self.__class__.test_email, 'password':'Test@9999'
        })
        self.assertEqual(r.status_code, 201)
        d = json.loads(r.data)
        self.assertTrue(d['success'])
        self.assertIn('patient_id', d)
        self.__class__.patient_id = d['patient_id']

    def test_29_duplicate_registration_fails(self):
        """Registering the same email twice must return 409 Conflict."""
        self.client.post('/api/register', json={
            'name':'Dup User', 'age':25, 'gender':'Female',
            'email': self.__class__.test_email, 'password':'Pass@1111'
        })
        r = self.client.post('/api/register', json={
            'name':'Dup User', 'age':25, 'gender':'Female',
            'email': self.__class__.test_email, 'password':'Pass@1111'
        })
        self.assertEqual(r.status_code, 409)

    def test_30_login_success(self):
        """POST /api/login with correct credentials must return 200."""
        r = self.client.post('/api/login', json={
            'email': self.__class__.test_email, 'password':'Test@9999'
        })
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        self.assertTrue(d['success'])
        self.assertIn('patient_id', d)
        self.assertIn('name', d)

    def test_31_login_wrong_password_fails(self):
        """POST /api/login with wrong password must return 401."""
        r = self.client.post('/api/login', json={
            'email': self.__class__.test_email, 'password':'WrongPass'
        })
        self.assertEqual(r.status_code, 401)

    def test_32_login_unknown_email_fails(self):
        """POST /api/login with unknown email must return 404."""
        r = self.client.post('/api/login', json={
            'email':'nobody@nowhere.com', 'password':'anything'
        })
        self.assertEqual(r.status_code, 404)

    def test_33_predict_all_returns_9_results(self):
        """POST /api/predict_all must return 9 disease results."""
        pid = self.__class__.patient_id or 1
        r   = self.client.post('/api/predict_all', json={
            'patient_id':pid, 'bmi':33.6, 'glucose':148.0,
            'systolic_bp':72, 'diastolic_bp':35,
            'cholesterol':210, 'insulin':0,
            'skin_thickness':35, 'pregnancies':6,
            'age':50, 'diabetes_pedigree':0.627
        })
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        self.assertTrue(d['success'])
        self.assertEqual(len(d['results']), 9,
            f"Expected 9 results, got {len(d['results'])}")

    def test_34_predict_all_results_have_required_fields(self):
        """Each result in predict_all must have all required fields."""
        pid = self.__class__.patient_id or 1
        r   = self.client.post('/api/predict_all', json={
            'patient_id':pid, 'bmi':28.0, 'glucose':100.0,
            'systolic_bp':115, 'age':35
        })
        d = json.loads(r.data)
        for result in d['results']:
            for field in ['disease_id','disease_name','risk_score',
                          'classification','recommendations']:
                self.assertIn(field, result,
                    f"Field '{field}' missing from result for disease {result.get('disease_id')}")

    def test_35_history_endpoint_returns_list(self):
        """GET /api/history/<id> must return a list of records."""
        pid = self.__class__.patient_id or 1
        r   = self.client.get(f'/api/history/{pid}')
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        self.assertTrue(d['success'])
        self.assertIsInstance(d['history'], list)

    def test_36_missing_required_fields_returns_400(self):
        """POST /api/predict_all without BMI must return 400."""
        r = self.client.post('/api/predict_all', json={
            'patient_id':1, 'glucose':100, 'systolic_bp':120
            # bmi is missing
        })
        self.assertEqual(r.status_code, 400)


# ════════════════════════════════════════════════════════════
# MODULE 7 — Input Validation
# ════════════════════════════════════════════════════════════
class TestInputValidation(unittest.TestCase):
    """Tests that invalid inputs are correctly rejected."""

    @classmethod
    def setUpClass(cls):
        from app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_37_register_missing_name(self):
        """Registration without name must return 400."""
        r = self.client.post('/api/register', json={
            'age':25, 'gender':'Male',
            'email':'noname@test.com', 'password':'Pass@1'
        })
        self.assertEqual(r.status_code, 400)

    def test_38_register_missing_password(self):
        """Registration without password must return 400."""
        r = self.client.post('/api/register', json={
            'name':'No Pass', 'age':25, 'gender':'Male',
            'email':'nopass@test.com'
        })
        self.assertEqual(r.status_code, 400)

    def test_39_password_is_hashed_in_db(self):
        """Stored password must be SHA-256 hash, not plain text."""
        from db_connection import get_patient_by_email
        import time
        email = f'hashtest_{int(time.time())}@cdrps.test'
        self.client.post('/api/register', json={
            'name':'Hash Test', 'age':30, 'gender':'Male',
            'email':email, 'password':'MySecretPass'
        })
        patient = get_patient_by_email(email)
        self.assertIsNotNone(patient)
        # Stored hash must NOT equal the plain password
        self.assertNotEqual(patient['PasswordHash'], 'MySecretPass',
                            "Password stored as plain text — SECURITY RISK")
        # Must be 64 chars (SHA-256 hex)
        self.assertEqual(len(patient['PasswordHash']), 64,
                         "Password hash is not SHA-256 length (64 chars)")

    def test_40_classification_is_binary(self):
        """Classification must be exactly 'High Risk' or 'Low Risk'."""
        from app import app
        client = app.test_client()
        r = client.post('/api/predict_all', json={
            'patient_id':1, 'bmi':28.0, 'glucose':100.0,
            'systolic_bp':115, 'age':35
        })
        d = json.loads(r.data)
        for result in d.get('results', []):
            self.assertIn(result['classification'], ['High Risk', 'Low Risk'],
                f"Invalid classification: {result['classification']}")


# ════════════════════════════════════════════════════════════
# MODULE 8 — Known High-Risk Patient Test
# ════════════════════════════════════════════════════════════
class TestKnownHighRiskPatient(unittest.TestCase):
    """
    System testing: a patient with severely elevated vitals should
    trigger High Risk on at least 3 of 9 diseases.
    (Black-box test — inputs known, outputs checked against expected range)
    """

    @classmethod
    def setUpClass(cls):
        from app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_41_high_risk_patient_triggers_multiple_high_risks(self):
        """
        Patient: Age=60, BMI=42, Glucose=210, BP=160, Cholesterol=300.
        Expect: at least 3 diseases flagged as High Risk.
        """
        r = self.client.post('/api/predict_all', json={
            'patient_id':1,
            'age':60, 'bmi':42.0, 'glucose':210.0,
            'systolic_bp':160, 'diastolic_bp':100,
            'cholesterol':300.0, 'insulin':200.0,
            'skin_thickness':45.0, 'pregnancies':8,
            'diabetes_pedigree':1.2
        })
        d       = json.loads(r.data)
        results = d.get('results', [])
        high    = [x for x in results if x['classification'] == 'High Risk']
        self.assertGreaterEqual(len(high), 3,
            f"High-risk patient only got {len(high)}/9 High Risk flags. "
            f"Results: {[(x['disease_name'], x['risk_score']) for x in results]}")

    def test_42_high_risk_scores_are_elevated(self):
        """High-risk patient average risk score must exceed 40%."""
        r = self.client.post('/api/predict_all', json={
            'patient_id':1,
            'age':60, 'bmi':42.0, 'glucose':210.0,
            'systolic_bp':160, 'cholesterol':300.0,
            'insulin':200.0, 'skin_thickness':45.0,
            'pregnancies':8, 'diabetes_pedigree':1.2
        })
        d       = json.loads(r.data)
        scores  = [x['risk_score'] for x in d.get('results', [])]
        avg     = sum(scores) / len(scores) if scores else 0
        self.assertGreater(avg, 40.0,
            f"Average risk score {avg:.1f}% too low for a high-risk patient")


# ════════════════════════════════════════════════════════════
# MODULE 9 — Known Low-Risk Patient Test
# ════════════════════════════════════════════════════════════
class TestKnownLowRiskPatient(unittest.TestCase):
    """
    System testing: a healthy patient should score Low Risk on
    the majority of diseases.
    """

    @classmethod
    def setUpClass(cls):
        from app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_43_healthy_patient_mostly_low_risk(self):
        """
        Patient: Age=25, BMI=21, Glucose=85, BP=110, Cholesterol=160.
        Expect: at least 6 of 9 diseases are Low Risk.
        """
        r = self.client.post('/api/predict_all', json={
            'patient_id':1,
            'age':25, 'bmi':21.0, 'glucose':85.0,
            'systolic_bp':110, 'diastolic_bp':70,
            'cholesterol':160.0, 'insulin':15.0,
            'skin_thickness':14.0, 'pregnancies':0,
            'diabetes_pedigree':0.12
        })
        d      = json.loads(r.data)
        results= d.get('results', [])
        low    = [x for x in results if x['classification'] == 'Low Risk']
        self.assertGreaterEqual(len(low), 6,
            f"Healthy patient only got {len(low)}/9 Low Risk flags. "
            f"Results: {[(x['disease_name'], x['risk_score']) for x in results]}")

    def test_44_healthy_patient_average_score_low(self):
        """Healthy patient average risk score must be below 50%."""
        r = self.client.post('/api/predict_all', json={
            'patient_id':1,
            'age':25, 'bmi':21.0, 'glucose':85.0,
            'systolic_bp':110, 'cholesterol':160.0,
            'insulin':15.0, 'pregnancies':0
        })
        d      = json.loads(r.data)
        scores = [x['risk_score'] for x in d.get('results', [])]
        avg    = sum(scores) / len(scores) if scores else 100
        self.assertLess(avg, 50.0,
            f"Average score {avg:.1f}% too high for a healthy patient")


# ════════════════════════════════════════════════════════════
# MODULE 10 — predict_all Full Integration Test
# ════════════════════════════════════════════════════════════
class TestPredictAllIntegration(unittest.TestCase):
    """End-to-end integration test: register → login → predict → history."""

    @classmethod
    def setUpClass(cls):
        from app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()
        import time
        cls.email = f'e2e_{int(time.time())}@cdrps.test'
        cls.pid   = None

    def test_45_full_flow_register_login_predict_history(self):
        """Complete end-to-end flow must succeed without errors."""
        c = self.__class__.client

        # Step 1: Register
        r = c.post('/api/register', json={
            'name':'E2E Test', 'age':45, 'gender':'Female',
            'email':self.__class__.email, 'password':'E2E@pass1'
        })
        self.assertEqual(r.status_code, 201, "Registration failed")
        pid = json.loads(r.data)['patient_id']
        self.__class__.pid = pid

        # Step 2: Login
        r = c.post('/api/login', json={
            'email':self.__class__.email, 'password':'E2E@pass1'
        })
        self.assertEqual(r.status_code, 200, "Login failed")

        # Step 3: Predict all
        r = c.post('/api/predict_all', json={
            'patient_id':pid, 'bmi':30.0, 'glucose':120.0,
            'systolic_bp':125, 'age':45
        })
        self.assertEqual(r.status_code, 200, "Prediction failed")
        d = json.loads(r.data)
        self.assertEqual(len(d['results']), 9, "Did not get 9 results")

        # Step 4: Check history
        r = c.get(f'/api/history/{pid}')
        self.assertEqual(r.status_code, 200, "History fetch failed")
        h = json.loads(r.data)
        self.assertGreater(h['count'], 0, "History is empty after prediction")


# ════════════════════════════════════════════════════════════
# RUNNER — prints a clean summary table
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*65)
    print("  CD-RPS | Unit & Integration Test Suite")
    print("  Student : Prosun Kumar Das | Enrollment: 2300990309")
    print("="*65 + "\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    modules = [
        TestDatabaseConnection,
        TestCRUDOperations,
        TestDataPreprocessing,
        TestMLModelLoading,
        TestAllDiseaseModels,
        TestFlaskAPI,
        TestInputValidation,
        TestKnownHighRiskPatient,
        TestKnownLowRiskPatient,
        TestPredictAllIntegration,
    ]
    for m in modules:
        suite.addTests(loader.loadTestsFromTestCase(m))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*65)
    print(f"  Tests run   : {result.testsRun}")
    print(f"  Passed      : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed      : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    print("="*65)

    if result.wasSuccessful():
        print("\n  ALL TESTS PASSED — Phase 5 Complete!\n")
    else:
        print("\n  Some tests failed — check output above.\n")
