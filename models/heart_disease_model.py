"""Heart Disease Prediction using Random Forest Classifier."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class HeartDiseasePredictor:
    FEATURE_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    def __init__(self, model_path=None):
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.is_trained = False
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, data_path):
        """Train the heart disease model."""
        df = pd.read_csv(data_path)
        X = df[self.FEATURE_NAMES]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            'accuracy': round(accuracy * 100, 2),
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def predict(self, features):
        """Predict heart disease risk.
        features: dict with keys matching FEATURE_NAMES
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        input_data = np.array([[features[f] for f in self.FEATURE_NAMES]])
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]

        return {
            'prediction': int(prediction),
            'risk': 'High Risk' if prediction == 1 else 'Low Risk',
            'confidence': round(float(max(probability)) * 100, 2),
            'probabilities': {
                'low_risk': round(float(probability[0]) * 100, 2),
                'high_risk': round(float(probability[1]) * 100, 2)
            }
        }

    def save(self, path):
        """Save model and scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load(self, path):
        """Load model and scaler."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
