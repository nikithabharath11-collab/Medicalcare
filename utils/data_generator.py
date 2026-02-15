"""
Generate synthetic medical datasets for training ML models.
In production, replace with real medical datasets from sources like:
- UCI ML Repository, Kaggle, PhysioNet, MIMIC-III
"""

import numpy as np
import pandas as pd
import os

def generate_heart_disease_data(n_samples=1000, save_path=None):
    """Generate synthetic heart disease dataset similar to UCI Heart Disease dataset."""
    np.random.seed(42)

    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),           # chest pain type
        'trestbps': np.random.randint(94, 200, n_samples),   # resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),      # serum cholesterol
        'fbs': np.random.randint(0, 2, n_samples),           # fasting blood sugar > 120
        'restecg': np.random.randint(0, 3, n_samples),       # resting ECG results
        'thalach': np.random.randint(71, 202, n_samples),    # max heart rate achieved
        'exang': np.random.randint(0, 2, n_samples),         # exercise induced angina
        'oldpeak': np.round(np.random.uniform(0, 6.2, n_samples), 1),  # ST depression
        'slope': np.random.randint(0, 3, n_samples),         # slope of ST segment
        'ca': np.random.randint(0, 5, n_samples),            # number of major vessels
        'thal': np.random.randint(0, 4, n_samples),          # thalassemia
    }

    # Generate target based on risk factors
    risk_score = (
        (data['age'] > 55).astype(int) * 0.2 +
        (data['cp'] >= 2).astype(int) * 0.15 +
        (data['trestbps'] > 140).astype(int) * 0.15 +
        (data['chol'] > 240).astype(int) * 0.1 +
        data['fbs'] * 0.05 +
        (data['thalach'] < 120).astype(int) * 0.1 +
        data['exang'] * 0.1 +
        (data['oldpeak'] > 2).astype(int) * 0.1 +
        (data['ca'] > 0).astype(int) * 0.05
    )
    data['target'] = (risk_score + np.random.normal(0, 0.15, n_samples) > 0.45).astype(int)

    df = pd.DataFrame(data)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def generate_diabetes_data(n_samples=1000, save_path=None):
    """Generate synthetic diabetes dataset similar to Pima Indians Diabetes dataset."""
    np.random.seed(43)

    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(44, 199, n_samples),
        'BloodPressure': np.random.randint(24, 122, n_samples),
        'SkinThickness': np.random.randint(7, 99, n_samples),
        'Insulin': np.random.randint(14, 846, n_samples),
        'BMI': np.round(np.random.uniform(18.2, 67.1, n_samples), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.078, 2.42, n_samples), 3),
        'Age': np.random.randint(21, 81, n_samples),
    }

    risk_score = (
        (data['Glucose'] > 140).astype(int) * 0.25 +
        (data['BMI'] > 30).astype(int) * 0.2 +
        (data['Age'] > 45).astype(int) * 0.15 +
        (data['Insulin'] > 200).astype(int) * 0.1 +
        (data['BloodPressure'] > 90).astype(int) * 0.1 +
        (data['DiabetesPedigreeFunction'] > 0.5).astype(int) * 0.1 +
        (data['Pregnancies'] > 5).astype(int) * 0.1
    )
    data['Outcome'] = (risk_score + np.random.normal(0, 0.15, n_samples) > 0.4).astype(int)

    df = pd.DataFrame(data)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def generate_general_disease_data(n_samples=1000, save_path=None):
    """Generate synthetic general disease symptom dataset."""
    np.random.seed(44)

    diseases = ['Common Cold', 'Flu', 'Malaria', 'Typhoid', 'Pneumonia',
                'Migraine', 'Hypertension', 'Diabetes', 'Anemia', 'Gastritis']
    symptoms = [
        'fever', 'cough', 'fatigue', 'headache', 'body_pain',
        'nausea', 'vomiting', 'diarrhea', 'shortness_of_breath', 'chest_pain',
        'dizziness', 'loss_of_appetite', 'sweating', 'chills', 'sore_throat',
        'runny_nose', 'joint_pain', 'abdominal_pain', 'weight_loss', 'skin_rash'
    ]

    # Symptom profiles for each disease
    disease_profiles = {
        'Common Cold':   [0.3, 0.8, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.1, 0.0, 0.0, 0.0],
        'Flu':           [0.9, 0.7, 0.8, 0.6, 0.8, 0.3, 0.2, 0.1, 0.2, 0.1, 0.3, 0.5, 0.5, 0.7, 0.4, 0.3, 0.5, 0.1, 0.1, 0.0],
        'Malaria':       [0.9, 0.1, 0.7, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.3, 0.4, 0.8, 0.9, 0.1, 0.0, 0.4, 0.2, 0.2, 0.0],
        'Typhoid':       [0.9, 0.1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.3, 0.1, 0.0, 0.2, 0.6, 0.3, 0.4, 0.1, 0.0, 0.3, 0.7, 0.3, 0.2],
        'Pneumonia':     [0.8, 0.9, 0.7, 0.3, 0.4, 0.2, 0.1, 0.0, 0.8, 0.6, 0.2, 0.4, 0.4, 0.6, 0.2, 0.1, 0.2, 0.1, 0.1, 0.0],
        'Migraine':      [0.1, 0.0, 0.5, 0.9, 0.2, 0.6, 0.4, 0.0, 0.0, 0.0, 0.7, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0],
        'Hypertension':  [0.1, 0.0, 0.4, 0.7, 0.2, 0.3, 0.1, 0.0, 0.3, 0.4, 0.6, 0.2, 0.3, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0],
        'Diabetes':      [0.1, 0.0, 0.6, 0.3, 0.2, 0.3, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1, 0.0, 0.0, 0.2, 0.1, 0.5, 0.1],
        'Anemia':        [0.1, 0.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0, 0.4, 0.2, 0.6, 0.3, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.3, 0.2],
        'Gastritis':     [0.1, 0.0, 0.3, 0.2, 0.1, 0.7, 0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.1, 0.1, 0.0, 0.0, 0.1, 0.8, 0.2, 0.0],
    }

    records = []
    for _ in range(n_samples):
        disease = np.random.choice(diseases)
        profile = disease_profiles[disease]
        symptom_values = [int(np.random.random() < p) for p in profile]
        records.append(symptom_values + [disease])

    columns = symptoms + ['disease']
    df = pd.DataFrame(records, columns=columns)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def generate_breast_cancer_data(n_samples=1000, save_path=None):
    """Generate synthetic breast cancer dataset similar to Wisconsin Breast Cancer dataset."""
    np.random.seed(45)

    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]

    data = {}
    # Benign characteristics (lower values generally)
    n_benign = n_samples // 2
    n_malignant = n_samples - n_benign

    for feat in feature_names:
        benign_vals = np.random.normal(12, 3, n_benign)
        malignant_vals = np.random.normal(18, 4, n_malignant)
        data[feat] = np.abs(np.concatenate([benign_vals, malignant_vals]))

    data['diagnosis'] = np.array([0]*n_benign + [1]*n_malignant)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def generate_all_datasets(data_dir):
    """Generate all datasets and save to data directory."""
    os.makedirs(data_dir, exist_ok=True)

    print("Generating Heart Disease dataset...")
    generate_heart_disease_data(1000, os.path.join(data_dir, 'heart_disease.csv'))

    print("Generating Diabetes dataset...")
    generate_diabetes_data(1000, os.path.join(data_dir, 'diabetes.csv'))

    print("Generating General Disease dataset...")
    generate_general_disease_data(2000, os.path.join(data_dir, 'general_disease.csv'))

    print("Generating Breast Cancer dataset...")
    generate_breast_cancer_data(1000, os.path.join(data_dir, 'breast_cancer.csv'))

    print("All datasets generated successfully!")


if __name__ == '__main__':
    generate_all_datasets(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
