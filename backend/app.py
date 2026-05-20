# ============================================================
# CD-RPS Project | backend/app.py
# Flask REST API + Frontend Server
# Student: Prosun Kumar Das | Enrollment: 2300990309
# ============================================================

import os, sys, pickle, hashlib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR   = os.path.join(BASE_DIR, '..', 'database')
ML_DIR   = os.path.join(BASE_DIR, '..', 'model')
FE_DIR   = os.path.join(BASE_DIR, '..', 'frontend')
sys.path.insert(0, DB_DIR)

from db_connection import (
    insert_patient, get_patient_by_email, get_patient_by_id,
    insert_vitals_record, insert_prediction_result,
    get_patient_history, get_all_diseases
)

app = Flask(__name__, static_folder=FE_DIR)
CORS(app)

# ── Load all 9 disease models ────────────────────────────────
MODELS  = {}
SCALERS = {}
for did in range(1, 10):
    mp = os.path.join(ML_DIR, f'model_{did}.pkl')
    sp = os.path.join(ML_DIR, f'scaler_{did}.pkl')
    if os.path.exists(mp) and os.path.exists(sp):
        with open(mp, 'rb') as f: MODELS[did]  = pickle.load(f)
        with open(sp, 'rb') as f: SCALERS[did] = pickle.load(f)
        print(f'[OK] model_{did}.pkl loaded.')
    else:
        print(f'[WARN] model_{did}.pkl missing — run train_all_models.py')
print(f'[OK] {len(MODELS)}/9 disease models ready.\n')

DISEASE_NAMES = {
    1:'Type 2 Diabetes', 2:'Cardiovascular Disease',
    3:'Chronic Kidney Disease', 4:'Stroke',
    5:'High Blood Pressure (Hypertension)', 6:'High Cholesterol',
    7:'Obesity', 8:'Coronary Heart Disease', 9:'COPD'
}

RECS_HIGH = {
    1:"Consult a doctor immediately. Reduce sugar and refined carbs. Monitor glucose daily.",
    2:"See a cardiologist. Reduce saturated fat. Exercise 30 min/day. Quit smoking if applicable.",
    3:"Consult a nephrologist. Reduce protein and salt intake. Monitor blood pressure closely.",
    4:"Seek emergency care if symptoms arise. Control BP strictly. Avoid smoking and alcohol.",
    5:"Reduce salt intake. Exercise regularly. Monitor BP daily. Limit alcohol consumption.",
    6:"Reduce saturated fats. Increase fibre intake. Exercise 150 min/week. Get a lipid panel.",
    7:"Set a gradual weight-loss goal. Follow a calorie-controlled diet. Increase physical activity.",
    8:"Consult a cardiologist. Quit smoking. Take prescribed medications. Exercise regularly.",
    9:"Consult a pulmonologist. Quit smoking immediately. Use prescribed inhalers. Avoid pollutants.",
}
RECS_LOW = "Your current risk is low. Maintain a healthy lifestyle with regular exercise and a balanced diet. Schedule annual checkups."

def get_rec(disease_id, classification):
    return RECS_HIGH.get(disease_id, "Consult your doctor.") if classification == 'High Risk' else RECS_LOW

# ── Helpers ──────────────────────────────────────────────────
def hash_password(pw):     return hashlib.sha256(pw.encode()).hexdigest()
def check_password(pw, h): return hashlib.sha256(pw.encode()).hexdigest() == h

# ── Serve Frontend ───────────────────────────────────────────
@app.route('/')
def index(): return send_from_directory(FE_DIR, 'index.html')

@app.route('/predict.html')
def predict_page(): return send_from_directory(FE_DIR, 'predict.html')

@app.route('/<path:filename>')
def static_files(filename): return send_from_directory(FE_DIR, filename)

# ── Health ───────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status':'running','models_loaded':len(MODELS),
                    'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

# ── Register ─────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    for f in ['name','age','gender','email','password']:
        if not data.get(f):
            return jsonify({'success':False,'message':f'Missing field: {f}'}), 400
    if get_patient_by_email(data['email']):
        return jsonify({'success':False,'message':'Email already registered.'}), 409
    try:
        age = int(data['age'])
        assert 1 <= age <= 120
    except:
        return jsonify({'success':False,'message':'Age must be between 1 and 120.'}), 400
    pid = insert_patient(data['name'], age, data['gender'], data['email'],
                         hash_password(data['password']), data.get('contact',''))
    return jsonify({'success':True,'message':'Registration successful!','patient_id':pid}), 201

# ── Login ────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    p    = get_patient_by_email(data.get('email',''))
    if not p: return jsonify({'success':False,'message':'Email not found.'}), 404
    if not check_password(data.get('password',''), p['PasswordHash']):
        return jsonify({'success':False,'message':'Incorrect password.'}), 401
    return jsonify({'success':True,'message':'Login successful!',
                    'patient_id':p['PatientID'],'name':p['Name'],
                    'email':p['Email'],'age':p['Age'],'gender':p['Gender']})

# ── Predict ALL 9 diseases at once ───────────────────────────
@app.route('/api/predict_all', methods=['POST'])
def predict_all():
    data = request.get_json()
    for field in ['patient_id','bmi','glucose','systolic_bp']:
        if data.get(field) is None:
            return jsonify({'success':False,'message':f'Missing field: {field}'}), 400
    try:
        patient_id  = int(data['patient_id'])
        bmi         = float(data['bmi'])
        glucose     = float(data['glucose'])
        systolic_bp = int(data['systolic_bp'])
        diastolic_bp= int(data.get('diastolic_bp', 80))
        cholesterol = float(data.get('cholesterol', 200))
        insulin     = float(data.get('insulin', 79))
        skin_thick  = float(data.get('skin_thickness', 20))
        pregnancies = int(data.get('pregnancies', 0))
        age         = int(data.get('age', 40))
        dpf         = float(data.get('diabetes_pedigree', 0.5))
    except Exception as e:
        return jsonify({'success':False,'message':f'Invalid value: {e}'}), 400

    # Save one vitals record for this session
    record_id = insert_vitals_record(patient_id, bmi, glucose, systolic_bp,
                                      diastolic_bp, cholesterol, insulin,
                                      skin_thick, pregnancies)

    features = np.array([[pregnancies, glucose, systolic_bp, skin_thick,
                          insulin, bmi, dpf, age]])

    results = []
    for did in range(1, 10):
        if did not in MODELS:
            continue
        scaled      = SCALERS[did].transform(features)
        raw_prob    = float(MODELS[did].predict_proba(scaled)[0][1])
        risk_score  = round(raw_prob * 100, 2)
        classif     = 'High Risk' if raw_prob >= 0.5 else 'Low Risk'
        rec         = get_rec(did, classif)

        # Save each disease prediction to DB
        insert_prediction_result(record_id, did, risk_score, classif, raw_prob,
                                  recommendations=rec)
        results.append({
            'disease_id':   did,
            'disease_name': DISEASE_NAMES[did],
            'risk_score':   risk_score,
            'classification': classif,
            'probability':  raw_prob,
            'recommendations': rec
        })

    # Sort: High Risk first, then by score descending
    results.sort(key=lambda x: (-int(x['classification']=='High Risk'), -x['risk_score']))

    return jsonify({
        'success':  True,
        'record_id': record_id,
        'vitals': {'bmi':bmi,'glucose':glucose,'systolic_bp':systolic_bp,
                   'diastolic_bp':diastolic_bp,'cholesterol':cholesterol,
                   'insulin':insulin,'age':age},
        'results': results
    })

# ── History (full detail) ─────────────────────────────────────
@app.route('/api/history/<int:patient_id>')
def history(patient_id):
    records = get_patient_history(patient_id)
    return jsonify({'success':True,'count':len(records),'history':records})

# ── Diseases list ─────────────────────────────────────────────
@app.route('/api/diseases')
def diseases():
    return jsonify({'success':True,'diseases':get_all_diseases()})

# ── Run Application ─────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
