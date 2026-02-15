"""Breast Cancer Detection using Neural Network (Deep Learning with TensorFlow/Keras)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Graceful TF import
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class BreastCancerPredictor:
    FEATURE_NAMES = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]

    def __init__(self, model_path=None):
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.use_dl = TF_AVAILABLE
        if model_path:
            scaler_exists = os.path.exists(model_path + '_scaler.pkl')
            model_exists = os.path.exists(model_path + '_sklearn_model.pkl') or \
                           os.path.exists(model_path + '_keras_model.keras')
            if scaler_exists and model_exists:
                self.load(model_path)

    def _build_neural_network(self, input_dim):
        """Build a deep neural network for classification."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, data_path):
        """Train the breast cancer detection model."""
        df = pd.read_csv(data_path)
        X = df[self.FEATURE_NAMES].values
        y = df['diagnosis'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.use_dl:
            self.model = self._build_neural_network(X_train_scaled.shape[1])
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=50, batch_size=32,
                validation_split=0.2, verbose=0
            )
            y_pred_prob = self.model.predict(X_test_scaled, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            training_history = {
                'loss': [round(float(v), 4) for v in history.history['loss'][-5:]],
                'accuracy': [round(float(v), 4) for v in history.history['accuracy'][-5:]]
            }
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            training_history = None

        self.is_trained = True
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=['Benign', 'Malignant'],
                                       output_dict=True)

        return {
            'accuracy': round(accuracy * 100, 2),
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'model_type': 'Deep Neural Network' if self.use_dl else 'Gradient Boosting',
            'training_history': training_history
        }

    def predict(self, features):
        """Predict breast cancer (benign/malignant)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        input_data = np.array([[features[f] for f in self.FEATURE_NAMES]])
        input_scaled = self.scaler.transform(input_data)

        if self.use_dl:
            prob = float(self.model.predict(input_scaled, verbose=0)[0][0])
            prediction = 1 if prob > 0.5 else 0
            probabilities = {'benign': round((1 - prob) * 100, 2),
                             'malignant': round(prob * 100, 2)}
        else:
            prediction = int(self.model.predict(input_scaled)[0])
            prob_arr = self.model.predict_proba(input_scaled)[0]
            probabilities = {'benign': round(float(prob_arr[0]) * 100, 2),
                             'malignant': round(float(prob_arr[1]) * 100, 2)}

        return {
            'prediction': prediction,
            'diagnosis': 'Malignant' if prediction == 1 else 'Benign',
            'confidence': max(probabilities.values()),
            'probabilities': probabilities
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scaler_path = path + '_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        if self.use_dl and self.model:
            self.model.save(path + '_keras_model.keras')
        elif self.model:
            joblib.dump(self.model, path + '_sklearn_model.pkl')

    def load(self, path):
        scaler_path = path + '_scaler.pkl'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        keras_path = path + '_keras_model.keras'
        sklearn_path = path + '_sklearn_model.pkl'

        if self.use_dl and os.path.exists(keras_path):
            self.model = keras.models.load_model(keras_path)
            self.is_trained = True
        elif os.path.exists(sklearn_path):
            self.model = joblib.load(sklearn_path)
            self.use_dl = False
            self.is_trained = True
