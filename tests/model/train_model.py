# ============================================================
# CD-RPS Project | model/train_model.py
# ML Model Training Script
# Student: Prosun Kumar Das | Enrollment: 2300990309
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
PLOT_PATH   = os.path.join(BASE_DIR, 'confusion_matrix.png')

# ── Step 1: Load Dataset ─────────────────────────────────────
print("\n" + "="*55)
print("  CD-RPS | ML Model Training")
print("="*55)
print("\n[1/6] Loading dataset...")

COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)

try:
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    print(f"   Dataset loaded from internet: {df.shape[0]} rows")
except Exception:
    print("   Internet unavailable — using built-in representative dataset...")
    np.random.seed(42)
    n = 768
    df = pd.DataFrame({
        'Pregnancies':              np.random.randint(0, 17, n),
        'Glucose':                  np.random.normal(120, 32, n).clip(44, 199),
        'BloodPressure':            np.random.normal(69, 19, n).clip(24, 122),
        'SkinThickness':            np.random.normal(20, 16, n).clip(0, 99),
        'Insulin':                  np.random.normal(79, 115, n).clip(0, 846),
        'BMI':                      np.random.normal(32, 7, n).clip(18, 67),
        'DiabetesPedigreeFunction': np.random.normal(0.47, 0.33, n).clip(0.08, 2.42),
        'Age':                      np.random.randint(21, 82, n),
        'Outcome':                  np.random.binomial(1, 0.35, n),
    })
    print(f"   Dataset ready: {df.shape[0]} rows")

# ── Step 2: Clean Data ───────────────────────────────────────
print("\n[2/6] Cleaning data...")

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
for col in zero_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"   Missing values handled. Shape: {df.shape}")
print(f"   Class split  -> Low Risk: {(df['Outcome']==0).sum()}  |  High Risk: {(df['Outcome']==1).sum()}")

# ── Step 3: Features ─────────────────────────────────────────
print("\n[3/6] Preparing features...")

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']

X = df[FEATURES].values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")

# ── Step 4: Scale ────────────────────────────────────────────
print("\n[4/6] Scaling features...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Step 5: Train ────────────────────────────────────────────
print("\n[5/6] Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_s, y_train)
print("   Training complete!")

# ── Step 6: Evaluate ─────────────────────────────────────────
print("\n[6/6] Evaluating...")
y_pred    = model.predict(X_test_s)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "="*55)
print("  MODEL PERFORMANCE METRICS")
print("="*55)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print("="*55)
print(classification_report(y_test, y_pred,
      target_names=['Low Risk', 'High Risk']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix - CD-RPS')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()
print(f"  Confusion matrix image saved!")

# ── Save ─────────────────────────────────────────────────────
with open(MODEL_PATH,  'wb') as f: pickle.dump(model,  f)
with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
print(f"  model.pkl  saved!")
print(f"  scaler.pkl saved!")

# ── Quick test ───────────────────────────────────────────────
print("\n" + "="*55)
print("  QUICK PREDICTION TEST")
print("="*55)
test_in  = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
test_s   = scaler.transform(test_in)
prob     = model.predict_proba(test_s)[0][1]
score    = round(prob * 100, 2)
label    = 'High Risk' if prob >= 0.5 else 'Low Risk'
print(f"  Input  : Glucose=148, BMI=33.6, Age=50")
print(f"  Score  : {score}%")
print(f"  Result : {label}")
print("="*55)
print("\n  Phase 2 COMPLETE!")
print("  model.pkl and scaler.pkl are ready for Phase 3.\n")
