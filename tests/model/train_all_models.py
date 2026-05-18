# ============================================================
# CD-RPS Project | model/train_all_models.py
# Multi-Disease ML Model Trainer
# Student: Prosun Kumar Das | Enrollment: 2300990309
# ============================================================
# Trains one Random Forest model per disease and saves:
#   model_<disease_id>.pkl
#   scaler_<disease_id>.pkl
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Common features across all diseases ─────────────────────
# [Pregnancies, Glucose, BloodPressure, SkinThickness,
#  Insulin, BMI, DiabetesPedigreeFunction, Age]
# These 8 features are collected from the user on the form.
# Each disease model is trained on a dataset that maps to
# these same 8 features (sometimes with proxy columns).

FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']

np.random.seed(42)

# ════════════════════════════════════════════════════════════
# DATASET GENERATORS
# Each function returns X (features) and y (0/1 labels)
# using real dataset URLs where possible, synthetic fallback.
# ════════════════════════════════════════════════════════════

def load_diabetes():
    """Disease ID 1 — Type 2 Diabetes (Pima Indians dataset)"""
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    try:
        df = pd.read_csv(url, names=cols)
        print("   [diabetes] Downloaded real Pima dataset.")
    except Exception:
        df = _synthetic_diabetes()
        print("   [diabetes] Using synthetic dataset.")
    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    for c in zero_cols:
        df[c] = df[c].fillna(df[c].median())
    return df[FEATURES].values, df['Outcome'].values


def load_cardiovascular():
    """Disease ID 2 — Cardiovascular Disease (UCI Heart Disease)"""
    url = ("https://raw.githubusercontent.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/"
           "master/data/heart.csv")
    try:
        df = pd.read_csv(url)
        # heart.csv columns: age,sex,cp,trestbps,chol,fbs,restecg,
        #                    thalach,exang,oldpeak,slope,ca,thal,target
        # Map to our 8 features
        out = pd.DataFrame()
        out['Pregnancies']            = (df['sex'] == 0).astype(int) * np.random.randint(0,5,len(df))
        out['Glucose']                = np.random.normal(105, 20, len(df)).clip(70, 200)
        out['BloodPressure']          = df['trestbps']
        out['SkinThickness']          = np.random.normal(22, 8, len(df)).clip(10, 50)
        out['Insulin']                = np.random.normal(80, 40, len(df)).clip(10, 300)
        out['BMI']                    = np.random.normal(28, 6, len(df)).clip(15, 50)
        out['DiabetesPedigreeFunction']= df['chol'] / 500.0
        out['Age']                    = df['age']
        y = df['target'].values
        print("   [cardiovascular] Downloaded real Heart Disease dataset.")
        return out[FEATURES].values, y
    except Exception:
        print("   [cardiovascular] Using synthetic dataset.")
        return _synthetic_disease(
            high_bmi=True, high_bp=True, high_chol=True, prev=0.46)


def load_kidney():
    """Disease ID 3 — Chronic Kidney Disease"""
    url = ("https://raw.githubusercontent.com/dsaks/chronic-kidney-disease-prediction/"
           "main/kidney_disease.csv")
    try:
        df = pd.read_csv(url)
        df = df.replace('?', np.nan)
        # Map CKD features to our standard 8
        out = pd.DataFrame()
        out['Pregnancies']             = 0
        out['Glucose']                 = pd.to_numeric(df.get('bgr', pd.Series(np.random.normal(110,30,len(df)))), errors='coerce').fillna(110)
        out['BloodPressure']           = pd.to_numeric(df.get('bp',  pd.Series(np.random.normal(75, 15,len(df)))), errors='coerce').fillna(75)
        out['SkinThickness']           = np.random.normal(22, 8, len(df)).clip(10, 50)
        out['Insulin']                 = pd.to_numeric(df.get('sod', pd.Series(np.random.normal(137,5,len(df)))), errors='coerce').fillna(137) * 0.5
        out['BMI']                     = np.random.normal(27, 6, len(df)).clip(15, 50)
        out['DiabetesPedigreeFunction']= pd.to_numeric(df.get('sc',  pd.Series(np.random.normal(1.2,0.5,len(df)))), errors='coerce').fillna(1.2) / 5
        out['Age']                     = pd.to_numeric(df.get('age', pd.Series(np.random.randint(30,80,len(df)))), errors='coerce').fillna(50)
        classification = df.get('classification', pd.Series(['ckd']*len(df)))
        y = (classification.astype(str).str.strip() == 'ckd').astype(int).values
        print("   [kidney] Downloaded real CKD dataset.")
        return out[FEATURES].values, y
    except Exception:
        print("   [kidney] Using synthetic dataset.")
        return _synthetic_disease(high_bp=True, prev=0.40)


def load_stroke():
    """Disease ID 4 — Stroke"""
    url = ("https://raw.githubusercontent.com/dsrscientist/dataset1/"
           "master/stroke.csv")
    try:
        df = pd.read_csv(url).dropna()
        out = pd.DataFrame()
        out['Pregnancies']             = 0
        out['Glucose']                 = pd.to_numeric(df.get('avg_glucose_level', 100), errors='coerce').fillna(100)
        out['BloodPressure']           = np.random.normal(78, 18, len(df)).clip(50, 180)
        out['SkinThickness']           = np.random.normal(22, 8, len(df)).clip(10, 50)
        out['Insulin']                 = np.random.normal(80, 40, len(df)).clip(10, 300)
        out['BMI']                     = pd.to_numeric(df.get('bmi', 28), errors='coerce').fillna(28)
        out['DiabetesPedigreeFunction']= pd.to_numeric(df.get('hypertension', 0), errors='coerce').fillna(0) * 0.5
        out['Age']                     = pd.to_numeric(df.get('age', 50), errors='coerce').fillna(50)
        y = pd.to_numeric(df.get('stroke', 0), errors='coerce').fillna(0).astype(int).values
        print("   [stroke] Downloaded real Stroke dataset.")
        return out[FEATURES].values, y
    except Exception:
        print("   [stroke] Using synthetic dataset.")
        return _synthetic_disease(high_bp=True, high_glucose=True, old_age=True, prev=0.05)


def load_hypertension():
    """Disease ID 5 — High Blood Pressure / Hypertension"""
    print("   [hypertension] Using clinically-calibrated synthetic dataset.")
    n = 1000
    age  = np.random.randint(20, 80, n)
    bmi  = np.random.normal(28, 7, n).clip(15, 55)
    bp   = np.random.normal(118, 22, n).clip(60, 220)
    chol = np.random.normal(200, 45, n).clip(100, 400)
    gluc = np.random.normal(105, 25, n).clip(60, 300)

    prob = (
        0.30 * (bp   > 130).astype(float) +
        0.20 * (bmi  > 30).astype(float)  +
        0.20 * (age  > 55).astype(float)  +
        0.15 * (chol > 240).astype(float) +
        0.15 * (gluc > 126).astype(float)
    )
    y = (np.random.rand(n) < prob).astype(int)

    X = np.column_stack([
        np.zeros(n),                         # Pregnancies
        gluc,                                # Glucose
        bp,                                  # BloodPressure
        np.random.normal(22, 8, n).clip(5,60), # SkinThickness
        np.random.normal(80, 40, n).clip(0,300), # Insulin
        bmi,                                 # BMI
        (chol / 500.0).clip(0.1, 1.5),      # DiabetesPedigreeFunction proxy
        age                                  # Age
    ])
    return X, y


def load_cholesterol():
    """Disease ID 6 — High Cholesterol"""
    print("   [cholesterol] Using clinically-calibrated synthetic dataset.")
    n = 1000
    age  = np.random.randint(20, 80, n)
    bmi  = np.random.normal(28, 7, n).clip(15, 55)
    bp   = np.random.normal(118, 22, n).clip(60, 200)
    gluc = np.random.normal(105, 25, n).clip(60, 300)
    smk  = (np.random.rand(n) < 0.30).astype(float)  # smoking proxy

    prob = (
        0.25 * (age  > 50).astype(float)  +
        0.25 * (bmi  > 30).astype(float)  +
        0.20 * smk                         +
        0.15 * (bp   > 130).astype(float) +
        0.15 * (gluc > 110).astype(float)
    )
    y = (np.random.rand(n) < prob).astype(int)

    X = np.column_stack([
        np.zeros(n),
        gluc, bp,
        np.random.normal(22, 8, n).clip(5, 60),
        np.random.normal(80, 40, n).clip(0, 300),
        bmi, smk * 0.8 + 0.1, age
    ])
    return X, y


def load_obesity():
    """Disease ID 7 — Obesity"""
    print("   [obesity] Using clinically-calibrated synthetic dataset.")
    n = 1000
    age  = np.random.randint(18, 75, n)
    bmi  = np.random.normal(28, 8, n).clip(15, 60)
    gluc = np.random.normal(108, 28, n).clip(60, 300)
    bp   = np.random.normal(118, 20, n).clip(60, 200)

    prob = (
        0.60 * (bmi  >= 30).astype(float) +
        0.15 * (gluc > 110).astype(float) +
        0.15 * (age  > 40).astype(float)  +
        0.10 * (bp   > 130).astype(float)
    )
    y = (np.random.rand(n) < prob.clip(0,1)).astype(int)

    X = np.column_stack([
        np.zeros(n), gluc, bp,
        np.random.normal(22, 8, n).clip(5, 60),
        np.random.normal(80, 40, n).clip(0, 300),
        bmi,
        np.random.uniform(0.1, 1.0, n),
        age
    ])
    return X, y


def load_coronary():
    """Disease ID 8 — Coronary Heart Disease (Framingham-based)"""
    url = ("https://raw.githubusercontent.com/rashida048/Datasets/"
           "master/framingham.csv")
    try:
        df = pd.read_csv(url).dropna()
        out = pd.DataFrame()
        out['Pregnancies']             = df.get('prevalentHyp', 0).astype(float)
        out['Glucose']                 = pd.to_numeric(df.get('glucose', 100), errors='coerce').fillna(100)
        out['BloodPressure']           = pd.to_numeric(df.get('sysBP', 120), errors='coerce').fillna(120)
        out['SkinThickness']           = np.random.normal(22, 8, len(df)).clip(5, 60)
        out['Insulin']                 = np.random.normal(80, 40, len(df)).clip(0, 300)
        out['BMI']                     = pd.to_numeric(df.get('BMI', 27), errors='coerce').fillna(27)
        out['DiabetesPedigreeFunction']= pd.to_numeric(df.get('totChol', 200), errors='coerce').fillna(200) / 500
        out['Age']                     = pd.to_numeric(df.get('age', 50), errors='coerce').fillna(50)
        y = pd.to_numeric(df.get('TenYearCHD', 0), errors='coerce').fillna(0).astype(int).values
        print("   [coronary] Downloaded real Framingham dataset.")
        return out[FEATURES].values, y
    except Exception:
        print("   [coronary] Using synthetic dataset.")
        return _synthetic_disease(high_bp=True, high_chol=True, old_age=True, prev=0.15)


def load_copd():
    """Disease ID 9 — COPD (Chronic Obstructive Pulmonary Disease)"""
    print("   [COPD] Using clinically-calibrated synthetic dataset.")
    n = 1000
    age    = np.random.randint(30, 80, n)
    bmi    = np.random.normal(26, 7, n).clip(14, 50)
    bp     = np.random.normal(115, 20, n).clip(60, 200)
    gluc   = np.random.normal(102, 22, n).clip(60, 300)
    smk_yr = np.random.randint(0, 40, n)   # smoking years proxy via DPF

    prob = (
        0.35 * (smk_yr > 10).astype(float) +
        0.25 * (age    > 55).astype(float) +
        0.20 * (bmi    < 22).astype(float) +
        0.10 * (bp     > 130).astype(float)+
        0.10 * (gluc   > 120).astype(float)
    )
    y = (np.random.rand(n) < prob).astype(int)

    X = np.column_stack([
        np.zeros(n), gluc, bp,
        np.random.normal(18, 6, n).clip(5, 50),
        np.random.normal(60, 30, n).clip(0, 200),
        bmi,
        (smk_yr / 40.0).clip(0, 1),
        age
    ])
    return X, y


# ── Synthetic fallback helper ────────────────────────────────
def _synthetic_diabetes():
    n = 768
    return pd.DataFrame({
        'Pregnancies':              np.random.randint(0,17,n),
        'Glucose':                  np.random.normal(120,32,n).clip(44,199),
        'BloodPressure':            np.random.normal(69,19,n).clip(24,122),
        'SkinThickness':            np.random.normal(20,16,n).clip(0,99),
        'Insulin':                  np.random.normal(79,115,n).clip(0,846),
        'BMI':                      np.random.normal(32,7,n).clip(18,67),
        'DiabetesPedigreeFunction': np.random.normal(0.47,0.33,n).clip(0.08,2.42),
        'Age':                      np.random.randint(21,82,n),
        'Outcome':                  np.random.binomial(1,0.35,n),
    })


def _synthetic_disease(high_bmi=False, high_bp=False, high_glucose=False,
                        high_chol=False, old_age=False, prev=0.35, n=800):
    age  = np.random.randint(25, 80, n)
    bmi  = np.random.normal(29, 7, n).clip(15, 55)
    bp   = np.random.normal(118, 22, n).clip(60, 200)
    gluc = np.random.normal(108, 28, n).clip(60, 300)
    chol = np.random.normal(200, 45, n).clip(100, 400)

    factors = []
    if high_bmi:     factors.append((bmi   > 30).astype(float))
    if high_bp:      factors.append((bp    > 130).astype(float))
    if high_glucose: factors.append((gluc  > 126).astype(float))
    if high_chol:    factors.append((chol  > 240).astype(float))
    if old_age:      factors.append((age   > 55).astype(float))
    if not factors:  factors.append(np.zeros(n))

    base = np.mean(factors, axis=0) * 0.7 + prev * 0.3
    y = (np.random.rand(n) < base.clip(0.05, 0.95)).astype(int)

    X = np.column_stack([
        np.zeros(n), gluc, bp,
        np.random.normal(22, 8, n).clip(5, 60),
        np.random.normal(80, 40, n).clip(0, 300),
        bmi, chol/500.0, age
    ])
    return X, y


# ════════════════════════════════════════════════════════════
# DISEASE REGISTRY
# disease_id must match what's in your DISEASE table
# ════════════════════════════════════════════════════════════
DISEASES = [
    {'id': 1, 'name': 'Type 2 Diabetes',                        'loader': load_diabetes},
    {'id': 2, 'name': 'Cardiovascular Disease',                  'loader': load_cardiovascular},
    {'id': 3, 'name': 'Chronic Kidney Disease',                  'loader': load_kidney},
    {'id': 4, 'name': 'Stroke',                                  'loader': load_stroke},
    {'id': 5, 'name': 'High Blood Pressure (Hypertension)',      'loader': load_hypertension},
    {'id': 6, 'name': 'High Cholesterol',                        'loader': load_cholesterol},
    {'id': 7, 'name': 'Obesity',                                 'loader': load_obesity},
    {'id': 8, 'name': 'Coronary Heart Disease',                  'loader': load_coronary},
    {'id': 9, 'name': 'COPD',                                    'loader': load_copd},
]

# ════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CD-RPS | Multi-Disease Model Trainer")
print("  Training 9 Random Forest models...")
print("="*60)

results = []

for d in DISEASES:
    print(f"\n[Disease {d['id']}/9] {d['name']}")
    print("  Loading data...")

    X, y = d['loader']()

    # Balance check
    pos = y.sum()
    neg = len(y) - pos
    print(f"  Samples: {len(y)} | Positive: {pos} | Negative: {neg}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Accuracy={acc*100:.1f}%  Precision={prec*100:.1f}%  Recall={rec*100:.1f}%  F1={f1*100:.1f}%")

    # Save
    model_path  = os.path.join(BASE_DIR, f'model_{d["id"]}.pkl')
    scaler_path = os.path.join(BASE_DIR, f'scaler_{d["id"]}.pkl')
    with open(model_path,  'wb') as f: pickle.dump(clf,    f)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    print(f"  Saved: model_{d['id']}.pkl + scaler_{d['id']}.pkl")

    results.append({
        'id': d['id'], 'name': d['name'],
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1
    })

# ── Summary ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  TRAINING COMPLETE — SUMMARY")
print("="*60)
print(f"  {'ID':<4} {'Disease':<38} {'Acc':>6} {'F1':>6}")
print(f"  {'-'*4} {'-'*38} {'-'*6} {'-'*6}")
for r in results:
    print(f"  {r['id']:<4} {r['name']:<38} {r['acc']*100:>5.1f}% {r['f1']*100:>5.1f}%")
print("="*60)
print(f"\n  {len(DISEASES)*2} files saved to: {BASE_DIR}")
print("  Next: restart Flask and test all diseases in the browser.\n")
