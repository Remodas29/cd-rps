-- ============================================================
-- Chronic Disease Risk Prediction System (CD-RPS)
-- IGNOU BCA Final Project | Student: Prosun Kumar Das
-- Enrollment: 2300990309 | Guide: Mr. Nirupam Singh
-- DATABASE: SQLite (use with DB Browser for SQLite)
-- ============================================================

PRAGMA foreign_keys = ON;

-- TABLE 1: PATIENT
CREATE TABLE IF NOT EXISTS PATIENT (
    PatientID     INTEGER PRIMARY KEY AUTOINCREMENT,
    Name          TEXT    NOT NULL,
    Age           INTEGER NOT NULL,
    Gender        TEXT    NOT NULL,
    Email         TEXT    NOT NULL UNIQUE,
    PasswordHash  TEXT    NOT NULL,
    ContactNumber TEXT,
    CreatedAt     TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt     TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 2: DISEASE
CREATE TABLE IF NOT EXISTS DISEASE (
    DiseaseID   INTEGER PRIMARY KEY AUTOINCREMENT,
    DiseaseName TEXT    NOT NULL UNIQUE,
    Description TEXT,
    RiskFactors TEXT,
    CreatedAt   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 3: VITALS_RECORD
CREATE TABLE IF NOT EXISTS VITALS_RECORD (
    RecordID      INTEGER PRIMARY KEY AUTOINCREMENT,
    PatientID     INTEGER NOT NULL,
    RecordDate    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    BMI           REAL    NOT NULL,
    GlucoseLevel  REAL    NOT NULL,
    SystolicBP    INTEGER NOT NULL,
    DiastolicBP   INTEGER,
    Cholesterol   REAL,
    InsulinLevel  REAL,
    SkinThickness REAL,
    Pregnancies   INTEGER DEFAULT 0,
    Notes         TEXT,
    FOREIGN KEY (PatientID) REFERENCES PATIENT(PatientID) ON DELETE CASCADE
);

-- TABLE 4: PREDICTION_RESULT
CREATE TABLE IF NOT EXISTS PREDICTION_RESULT (
    ResultID        INTEGER PRIMARY KEY AUTOINCREMENT,
    RecordID        INTEGER NOT NULL UNIQUE,
    DiseaseID       INTEGER NOT NULL,
    RiskScore       REAL    NOT NULL,
    Classification  TEXT    NOT NULL,
    RawProbability  REAL    NOT NULL,
    MLModelUsed     TEXT    NOT NULL DEFAULT 'Random Forest Classifier',
    ModelVersion    TEXT    DEFAULT '1.0',
    PredictionDate  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    Recommendations TEXT,
    FOREIGN KEY (RecordID)  REFERENCES VITALS_RECORD(RecordID) ON DELETE CASCADE,
    FOREIGN KEY (DiseaseID) REFERENCES DISEASE(DiseaseID)
);

-- INDEXES
CREATE INDEX IF NOT EXISTS IDX_VR_PatientID      ON VITALS_RECORD(PatientID);
CREATE INDEX IF NOT EXISTS IDX_PR_Classification ON PREDICTION_RESULT(Classification);
CREATE INDEX IF NOT EXISTS IDX_PR_Date           ON PREDICTION_RESULT(PredictionDate);

-- SEED: Reference diseases
INSERT OR IGNORE INTO DISEASE (DiseaseName, Description, RiskFactors) VALUES
('Type 2 Diabetes',       'Chronic condition affecting blood sugar processing.',    'High BMI, high glucose, high BP, family history'),
('Hypertension',          'High force of blood against artery walls.',              'High BMI, high salt, stress, age, family history'),
('Cardiovascular Disease','Diseases involving the heart or blood vessels.',         'High cholesterol, high BP, high BMI, smoking, diabetes');

-- SEED: Sample patient for testing
INSERT OR IGNORE INTO PATIENT (Name, Age, Gender, Email, PasswordHash, ContactNumber)
VALUES ('Test Patient', 45, 'Female', 'test@cdrps.com', 'hashed_password', '9876543210');

-- SEED: Sample vitals for testing
INSERT OR IGNORE INTO VITALS_RECORD (PatientID, BMI, GlucoseLevel, SystolicBP, DiastolicBP, Cholesterol, InsulinLevel, SkinThickness, Pregnancies)
VALUES (1, 33.6, 148.0, 72, 35, 210.0, 0.0, 35.0, 6);
