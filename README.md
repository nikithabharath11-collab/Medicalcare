# Medical Care & Health Science
## Machine Learning and Deep Learning Project

A comprehensive web application that uses **Machine Learning** and **Deep Learning** to predict diseases, assess health risks, and classify medical images.

---

## Features

| # | Model | Algorithm | Type |
|---|-------|-----------|------|
| 1 | Heart Disease Prediction | Gradient Boosting Classifier | ML - Binary Classification |
| 2 | Diabetes Prediction | Random Forest Classifier | ML - Binary Classification |
| 3 | Disease from Symptoms | Random Forest (Multi-class) | ML - Multi-class (10 diseases) |
| 4 | Breast Cancer Detection | Deep Neural Network (Keras) | DL - Binary Classification |
| 5 | Medical Image Classification | CNN + MobileNetV2 Transfer Learning | DL - Image Classification |

---

## Tech Stack

- **Backend:** Python, Flask
- **ML:** Scikit-Learn (Random Forest, Gradient Boosting, SVM)
- **DL:** TensorFlow / Keras (DNN, CNN, Transfer Learning)
- **Data:** Pandas, NumPy, Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, JavaScript

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train.py
```

This will:
- Generate synthetic medical datasets
- Train all 4 ML/DL models
- Save trained models to `models/saved/`

### 3. Run the Application

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## Project Structure

```
Medicalcare/
├── app.py                        # Flask web application
├── train.py                      # Model training script
├── config.py                     # Configuration
├── requirements.txt              # Dependencies
├── models/
│   ├── heart_disease_model.py    # Gradient Boosting (ML)
│   ├── diabetes_model.py         # Random Forest (ML)
│   ├── disease_predictor.py      # Multi-class RF (ML)
│   ├── breast_cancer_model.py    # Deep Neural Network (DL)
│   ├── image_classifier.py       # CNN Transfer Learning (DL)
│   └── saved/                    # Trained model files
├── data/                         # CSV datasets
├── utils/
│   └── data_generator.py         # Synthetic data generation
├── static/
│   ├── css/style.css
│   ├── images/
│   └── uploads/
├── templates/                    # HTML pages
│   ├── base.html, index.html, dashboard.html
│   ├── heart_disease.html, diabetes.html
│   ├── disease_prediction.html, breast_cancer.html
│   └── image_classification.html, about.html
└── notebooks/
    └── analysis.ipynb            # Data analysis notebook
```

---

## How It Works

### Machine Learning Pipeline
1. **Data Generation** - Synthetic datasets mimicking real medical data
2. **Preprocessing** - Feature scaling with StandardScaler
3. **Training** - Models trained with cross-validation
4. **Serialization** - Models saved using Joblib
5. **Serving** - Flask REST API serves predictions

### Deep Learning Pipeline
1. **Neural Network** - Dense layers + BatchNormalization + Dropout
2. **CNN** - MobileNetV2 pre-trained backbone + custom classification head
3. **Image Processing** - Resize to 224x224, normalize pixel values
4. **Inference** - Real-time predictions via API

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict/heart` | POST | Heart disease prediction |
| `/api/predict/diabetes` | POST | Diabetes risk assessment |
| `/api/predict/disease` | POST | Disease from symptoms |
| `/api/predict/breast-cancer` | POST | Breast cancer detection |
| `/api/predict/image` | POST | Medical image classification |
| `/api/model-info` | GET | Model information & status |

---

## Disclaimer

This project is for **educational purposes only**. Predictions should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
