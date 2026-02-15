"""Training script - generates data and trains all ML/DL models."""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.data_generator import generate_all_datasets
from models.heart_disease_model import HeartDiseasePredictor
from models.diabetes_model import DiabetesPredictor
from models.disease_predictor import DiseasePredictor
from models.breast_cancer_model import BreastCancerPredictor

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')


def train_all():
    """Train all models and save them."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Generate datasets
    print("=" * 60)
    print("STEP 1: Generating Synthetic Medical Datasets")
    print("=" * 60)
    generate_all_datasets(DATA_DIR)
    print()

    # Step 2: Train Heart Disease Model
    print("=" * 60)
    print("STEP 2: Training Heart Disease Prediction Model")
    print("=" * 60)
    start = time.time()
    heart_model = HeartDiseasePredictor()
    heart_results = heart_model.train(os.path.join(DATA_DIR, 'heart_disease.csv'))
    heart_model.save(os.path.join(MODEL_DIR, 'heart_disease_model.pkl'))
    print(f"  Model: Gradient Boosting Classifier")
    print(f"  Accuracy: {heart_results['accuracy']}%")
    print(f"  Time: {time.time() - start:.2f}s")
    print()

    # Step 3: Train Diabetes Model
    print("=" * 60)
    print("STEP 3: Training Diabetes Prediction Model")
    print("=" * 60)
    start = time.time()
    diabetes_model = DiabetesPredictor()
    diabetes_results = diabetes_model.train(os.path.join(DATA_DIR, 'diabetes.csv'))
    diabetes_model.save(os.path.join(MODEL_DIR, 'diabetes_model.pkl'))
    print(f"  Model: Random Forest Classifier")
    print(f"  Accuracy: {diabetes_results['accuracy']}%")
    print(f"  Time: {time.time() - start:.2f}s")
    print()

    # Step 4: Train Disease Predictor
    print("=" * 60)
    print("STEP 4: Training General Disease Prediction Model")
    print("=" * 60)
    start = time.time()
    disease_model = DiseasePredictor()
    disease_results = disease_model.train(os.path.join(DATA_DIR, 'general_disease.csv'))
    disease_model.save(os.path.join(MODEL_DIR, 'disease_model.pkl'))
    print(f"  Model: Random Forest Classifier (Multi-class)")
    print(f"  Accuracy: {disease_results['accuracy']}%")
    print(f"  Diseases: {', '.join(disease_results['diseases'])}")
    print(f"  Time: {time.time() - start:.2f}s")
    print()

    # Step 5: Train Breast Cancer Model (Deep Learning)
    print("=" * 60)
    print("STEP 5: Training Breast Cancer Detection Model (Deep Learning)")
    print("=" * 60)
    start = time.time()
    cancer_model = BreastCancerPredictor()
    cancer_results = cancer_model.train(os.path.join(DATA_DIR, 'breast_cancer.csv'))
    cancer_model.save(os.path.join(MODEL_DIR, 'breast_cancer_model'))
    print(f"  Model: {cancer_results['model_type']}")
    print(f"  Accuracy: {cancer_results['accuracy']}%")
    print(f"  Time: {time.time() - start:.2f}s")
    print()

    # Summary
    print("=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"  Heart Disease Model:     {heart_results['accuracy']}%")
    print(f"  Diabetes Model:          {diabetes_results['accuracy']}%")
    print(f"  Disease Predictor:       {disease_results['accuracy']}%")
    print(f"  Breast Cancer Model:     {cancer_results['accuracy']}%")
    print(f"\n  Models saved to: {MODEL_DIR}")
    print("=" * 60)

    return {
        'heart': heart_results,
        'diabetes': diabetes_results,
        'disease': disease_results,
        'cancer': cancer_results
    }


if __name__ == '__main__':
    train_all()
