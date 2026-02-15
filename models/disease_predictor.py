"""General Disease Prediction from Symptoms using Multi-class Classification."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


class DiseasePredictor:
    SYMPTOM_LIST = [
        'fever', 'cough', 'fatigue', 'headache', 'body_pain',
        'nausea', 'vomiting', 'diarrhea', 'shortness_of_breath', 'chest_pain',
        'dizziness', 'loss_of_appetite', 'sweating', 'chills', 'sore_throat',
        'runny_nose', 'joint_pain', 'abdominal_pain', 'weight_loss', 'skin_rash'
    ]

    DISEASE_INFO = {
        'Common Cold': {
            'description': 'A viral infectious disease of the upper respiratory tract.',
            'precautions': ['Rest well', 'Stay hydrated', 'Use saline nasal drops', 'Avoid cold exposure'],
            'severity': 'Mild'
        },
        'Flu': {
            'description': 'Influenza is a viral infection that attacks the respiratory system.',
            'precautions': ['Get vaccinated', 'Rest and hydrate', 'Take antiviral medications', 'Avoid contact with others'],
            'severity': 'Moderate'
        },
        'Malaria': {
            'description': 'A mosquito-borne infectious disease affecting humans.',
            'precautions': ['Use mosquito nets', 'Take antimalarial drugs', 'Eliminate standing water', 'Use insect repellent'],
            'severity': 'Severe'
        },
        'Typhoid': {
            'description': 'A bacterial infection caused by Salmonella typhi.',
            'precautions': ['Drink clean water', 'Eat properly cooked food', 'Get vaccinated', 'Maintain hygiene'],
            'severity': 'Severe'
        },
        'Pneumonia': {
            'description': 'An infection that inflames air sacs in one or both lungs.',
            'precautions': ['Get vaccinated', 'Practice good hygiene', 'Avoid smoking', 'Seek medical treatment'],
            'severity': 'Severe'
        },
        'Migraine': {
            'description': 'A headache disorder characterized by recurrent moderate to severe headaches.',
            'precautions': ['Avoid triggers', 'Maintain regular sleep', 'Manage stress', 'Stay hydrated'],
            'severity': 'Moderate'
        },
        'Hypertension': {
            'description': 'A condition in which blood pressure is persistently elevated.',
            'precautions': ['Reduce salt intake', 'Exercise regularly', 'Monitor blood pressure', 'Take prescribed medications'],
            'severity': 'Moderate'
        },
        'Diabetes': {
            'description': 'A metabolic disease causing high blood sugar over a prolonged period.',
            'precautions': ['Monitor blood sugar', 'Maintain healthy diet', 'Exercise regularly', 'Take medications as prescribed'],
            'severity': 'Moderate'
        },
        'Anemia': {
            'description': 'A condition where blood lacks adequate healthy red blood cells.',
            'precautions': ['Eat iron-rich foods', 'Take supplements', 'Get regular checkups', 'Avoid excessive bleeding'],
            'severity': 'Moderate'
        },
        'Gastritis': {
            'description': 'Inflammation of the lining of the stomach.',
            'precautions': ['Avoid spicy food', 'Eat smaller meals', 'Avoid alcohol', 'Take antacids as prescribed'],
            'severity': 'Mild'
        },
    }

    def __init__(self, model_path=None):
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=15, random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, data_path):
        """Train the disease prediction model."""
        df = pd.read_csv(data_path)
        X = df[self.SYMPTOM_LIST]
        y = self.label_encoder.fit_transform(df['disease'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        target_names = self.label_encoder.classes_.tolist()
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

        return {
            'accuracy': round(accuracy * 100, 2),
            'report': report,
            'diseases': target_names
        }

    def predict(self, symptoms):
        """Predict disease from symptoms.
        symptoms: list of symptom names that are present
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        input_vector = [1 if s in symptoms else 0 for s in self.SYMPTOM_LIST]
        input_data = np.array([input_vector])

        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]

        disease_name = self.label_encoder.inverse_transform([prediction])[0]

        # Top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            name = self.label_encoder.inverse_transform([idx])[0]
            info = self.DISEASE_INFO.get(name, {})
            top_predictions.append({
                'disease': name,
                'probability': round(float(probabilities[idx]) * 100, 2),
                'description': info.get('description', 'N/A'),
                'precautions': info.get('precautions', []),
                'severity': info.get('severity', 'Unknown')
            })

        return {
            'primary_prediction': disease_name,
            'confidence': round(float(max(probabilities)) * 100, 2),
            'top_predictions': top_predictions
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.is_trained = True
