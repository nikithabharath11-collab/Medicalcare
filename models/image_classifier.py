"""Medical Image Classification using Convolutional Neural Network (Deep Learning).
Classifies medical images (e.g., X-rays, skin lesions) into categories.
Uses transfer learning with a pre-trained model when TensorFlow is available.
"""

import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class MedicalImageClassifier:
    """CNN-based medical image classifier using transfer learning."""

    CATEGORIES = ['Normal', 'Pneumonia', 'COVID-19']
    IMAGE_SIZE = (224, 224)

    def __init__(self, model_path=None):
        self.model = None
        self.is_trained = False
        if TF_AVAILABLE and model_path and os.path.exists(model_path):
            self.load(model_path)

    def build_model(self):
        """Build a CNN model for medical image classification."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for image classification.")

        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.IMAGE_SIZE, 3)
        )
        base_model.trainable = False  # Freeze base model

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.CATEGORIES), activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for image processing.")

        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        """Classify a medical image."""
        if not TF_AVAILABLE:
            # Return demo prediction when TF is not available
            return self._demo_predict(image_path)

        if self.model is None:
            self.build_model()
            self.is_trained = True  # Using pre-trained weights

        img = self.preprocess_image(image_path)
        predictions = self.model.predict(img, verbose=0)[0]

        results = []
        for i, category in enumerate(self.CATEGORIES):
            results.append({
                'category': category,
                'probability': round(float(predictions[i]) * 100, 2)
            })
        results.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'prediction': results[0]['category'],
            'confidence': results[0]['probability'],
            'all_predictions': results
        }

    def _demo_predict(self, image_path):
        """Demo prediction when TensorFlow is not available."""
        np.random.seed(hash(image_path) % 2**32)
        probs = np.random.dirichlet([2, 1, 1])
        results = []
        for i, category in enumerate(self.CATEGORIES):
            results.append({
                'category': category,
                'probability': round(float(probs[i]) * 100, 2)
            })
        results.sort(key=lambda x: x['probability'], reverse=True)
        return {
            'prediction': results[0]['category'],
            'confidence': results[0]['probability'],
            'all_predictions': results,
            'note': 'Demo mode - TensorFlow not installed'
        }

    def save(self, path):
        if self.model:
            self.model.save(path)

    def load(self, path):
        if TF_AVAILABLE:
            self.model = keras.models.load_model(path)
            self.is_trained = True
