"""
Medical Care & Health Science - ML/DL Web Application
Created by: B NIKITHA | 25BCE2517 | VIT Vellore
"""

import os
import sys
import json
from datetime import datetime
from io import BytesIO

import numpy as np
from flask import (Flask, render_template, request, jsonify, redirect,
                   url_for, flash, send_file, session)
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import Config
from database import db, login_manager, User, PredictionHistory, LoginHistory, SavedReport
from models.heart_disease_model import HeartDiseasePredictor
from models.diabetes_model import DiabetesPredictor
from models.disease_predictor import DiseasePredictor
from models.breast_cancer_model import BreastCancerPredictor
from models.image_classifier import MedicalImageClassifier

app = Flask(__name__)
app.config.from_object(Config)
DATABASE_URL = os.environ.get('DATABASE_URL', '')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
if DATABASE_URL:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'medicalcare.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Init extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_message_category = 'error'
login_manager.login_message = 'Please login to access MedicalCare AI.'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create tables
with app.app_context():
    db.create_all()

# --- Auto-train models if not found ---
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
if not os.path.exists(os.path.join(MODEL_DIR, 'heart_disease_model.pkl')):
    print("Models not found. Training now...")
    from train import train_all
    train_all()
    print("Training complete!\n")
else:
    print("Models already trained. Skipping training.")

# --- Load Models ---
heart_model = HeartDiseasePredictor(os.path.join(MODEL_DIR, 'heart_disease_model.pkl'))
diabetes_model = DiabetesPredictor(os.path.join(MODEL_DIR, 'diabetes_model.pkl'))
disease_model = DiseasePredictor(os.path.join(MODEL_DIR, 'disease_model.pkl'))
cancer_model = BreastCancerPredictor(os.path.join(MODEL_DIR, 'breast_cancer_model'))
image_classifier = MedicalImageClassifier()

# --- Gemini AI Setup ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
gemini_model = None

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_prediction(model_name, input_data, result, confidence, risk_level):
    """Save prediction to history if user is logged in."""
    if current_user.is_authenticated:
        pred = PredictionHistory(
            user_id=current_user.id,
            model_name=model_name,
            input_data=json.dumps(input_data) if isinstance(input_data, dict) else str(input_data),
            result=result,
            confidence=confidence,
            risk_level=risk_level
        )
        db.session.add(pred)
        db.session.commit()


# ===================== AUTH ROUTES =====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            # Save login history
            log = LoginHistory(user_id=user.id, ip_address=request.remote_addr)
            db.session.add(log)
            db.session.commit()
            flash('Welcome back, ' + user.name + '!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not name or not email or not password:
            flash('All fields are required.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


# ===================== PAGE ROUTES =====================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/home')
@login_required
def home():
    return render_template('index.html')


@app.route('/heart-disease')
@login_required
def heart_disease_page():
    return render_template('heart_disease.html')


@app.route('/diabetes')
@login_required
def diabetes_page():
    return render_template('diabetes.html')


@app.route('/disease-prediction')
@login_required
def disease_prediction_page():
    return render_template('disease_prediction.html',
                           symptoms=DiseasePredictor.SYMPTOM_LIST)


@app.route('/breast-cancer')
@login_required
def breast_cancer_page():
    return render_template('breast_cancer.html')


@app.route('/image-classification')
@login_required
def image_classification_page():
    return render_template('image_classification.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/about')
@login_required
def about():
    return render_template('about.html')


@app.route('/bmi')
@login_required
def bmi_page():
    return render_template('bmi.html')


@app.route('/ai-chat')
@login_required
def ai_chat_page():
    return render_template('ai_chat.html')


@app.route('/history')
@login_required
def history_page():
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id) \
        .order_by(PredictionHistory.created_at.desc()).limit(50).all()
    return render_template('history.html', predictions=predictions)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.name = request.form.get('name', current_user.name).strip()
        current_user.age = request.form.get('age', type=int)
        current_user.gender = request.form.get('gender', '').strip()
        current_user.blood_group = request.form.get('blood_group', '').strip()
        current_user.phone = request.form.get('phone', '').strip()
        current_user.address = request.form.get('address', '').strip()
        current_user.emergency_contact = request.form.get('emergency_contact', '').strip()
        current_user.medical_conditions = request.form.get('medical_conditions', '').strip()
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    return render_template('profile.html')


@app.route('/patient-history')
@login_required
def patient_history():
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id) \
        .order_by(PredictionHistory.created_at.desc()).all()
    logins = LoginHistory.query.filter_by(user_id=current_user.id) \
        .order_by(LoginHistory.login_time.desc()).limit(20).all()
    reports = SavedReport.query.filter_by(user_id=current_user.id) \
        .order_by(SavedReport.created_at.desc()).all()
    return render_template('patient_history.html',
                           predictions=predictions, logins=logins, reports=reports)


@app.route('/download-report/<int:report_id>')
@login_required
def download_saved_report(report_id):
    report = SavedReport.query.filter_by(id=report_id, user_id=current_user.id).first()
    if report and report.pdf_data:
        return send_file(
            BytesIO(report.pdf_data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=report.report_name
        )
    flash('Report not found.', 'error')
    return redirect(url_for('patient_history'))


# ===================== API ROUTES =====================

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.get_json()
        features = {
            'age': int(data['age']), 'sex': int(data['sex']),
            'cp': int(data['cp']), 'trestbps': int(data['trestbps']),
            'chol': int(data['chol']), 'fbs': int(data['fbs']),
            'restecg': int(data['restecg']), 'thalach': int(data['thalach']),
            'exang': int(data['exang']), 'oldpeak': float(data['oldpeak']),
            'slope': int(data['slope']), 'ca': int(data['ca']),
            'thal': int(data['thal']),
        }
        result = heart_model.predict(features)
        save_prediction('Heart Disease', features, result['risk'], result['confidence'], result['risk'])
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()
        features = {
            'Pregnancies': int(data['pregnancies']), 'Glucose': int(data['glucose']),
            'BloodPressure': int(data['bloodPressure']), 'SkinThickness': int(data['skinThickness']),
            'Insulin': int(data['insulin']), 'BMI': float(data['bmi']),
            'DiabetesPedigreeFunction': float(data['dpf']), 'Age': int(data['age']),
        }
        result = diabetes_model.predict(features)
        save_prediction('Diabetes', features, result['risk'], result['confidence'], result['risk'])
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/predict/disease', methods=['POST'])
def predict_disease():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        if not symptoms:
            return jsonify({'status': 'error', 'message': 'No symptoms selected'}), 400
        result = disease_model.predict(symptoms)
        severity = result['top_predictions'][0]['severity'] if result['top_predictions'] else 'Unknown'
        save_prediction('Disease Symptom', symptoms, result['primary_prediction'], result['confidence'], severity)
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/predict/breast-cancer', methods=['POST'])
def predict_breast_cancer():
    try:
        data = request.get_json()
        features = {k: float(data[k]) for k in BreastCancerPredictor.FEATURE_NAMES}
        result = cancer_model.predict(features)
        save_prediction('Breast Cancer', features, result['diagnosis'], result['confidence'], result['diagnosis'])
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = image_classifier.predict(filepath)
            save_prediction('Image Classification', filename, result['prediction'], result['confidence'], result['prediction'])
            return jsonify({'status': 'success', 'result': result})
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/chat', methods=['POST'])
def ai_chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'status': 'error', 'message': 'Empty message'}), 400

        system_prompt = (
            "You are MedCare AI, a helpful medical health assistant. "
            "Provide informative, accurate health guidance. "
            "Always remind users to consult a doctor for serious concerns. "
            "Keep responses concise (2-4 sentences). "
            "Be empathetic and professional."
        )

        if gemini_model and GEMINI_API_KEY:
            response = gemini_model.generate_content(
                f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
            )
            reply = response.text
        else:
            reply = get_fallback_response(user_message)

        return jsonify({'status': 'success', 'reply': reply})
    except Exception as e:
        return jsonify({'status': 'success', 'reply': get_fallback_response(data.get('message', ''))})


def get_fallback_response(message):
    """Rule-based fallback when Gemini is not available."""
    msg = message.lower()
    if any(w in msg for w in ['headache', 'head pain', 'migraine']):
        return "Headaches can be caused by stress, dehydration, or lack of sleep. Try resting in a dark room and staying hydrated. If headaches are severe or persistent, please consult a doctor."
    elif any(w in msg for w in ['fever', 'temperature']):
        return "For mild fever, rest and stay hydrated. You can take paracetamol as directed. If fever exceeds 103F (39.4C) or persists for more than 3 days, seek medical attention immediately."
    elif any(w in msg for w in ['cold', 'cough', 'flu']):
        return "For cold and cough, get plenty of rest, drink warm fluids, and consider over-the-counter remedies. If symptoms worsen or last more than 10 days, please see a healthcare provider."
    elif any(w in msg for w in ['diabetes', 'blood sugar', 'glucose']):
        return "Managing diabetes involves monitoring blood sugar, eating a balanced diet, regular exercise, and taking medications as prescribed. Regular checkups are essential. Consult your endocrinologist for personalized advice."
    elif any(w in msg for w in ['heart', 'chest pain', 'cardiac']):
        return "Heart health is crucial. Maintain a healthy diet, exercise regularly, manage stress, and monitor blood pressure. If you experience chest pain or shortness of breath, seek emergency medical care immediately."
    elif any(w in msg for w in ['stress', 'anxiety', 'depression', 'mental']):
        return "Mental health is just as important as physical health. Practice relaxation techniques, maintain social connections, and consider speaking with a counselor. You are not alone - help is available."
    elif any(w in msg for w in ['diet', 'nutrition', 'food', 'eat']):
        return "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins supports overall health. Limit processed foods, sugar, and sodium. Consider consulting a nutritionist for personalized dietary advice."
    elif any(w in msg for w in ['exercise', 'workout', 'fitness']):
        return "Regular physical activity (150 minutes/week moderate exercise) improves cardiovascular health, mood, and overall wellbeing. Start slowly and gradually increase intensity. Always warm up before exercising."
    elif any(w in msg for w in ['sleep', 'insomnia']):
        return "Quality sleep is vital for health. Aim for 7-9 hours nightly. Maintain a consistent sleep schedule, limit screen time before bed, and create a comfortable sleeping environment. Consult a doctor if insomnia persists."
    elif any(w in msg for w in ['hello', 'hi', 'hey']):
        return "Hello! I'm MedCare AI, your health assistant. How can I help you today? You can ask me about symptoms, health tips, nutrition, exercise, or any medical concerns."
    elif any(w in msg for w in ['thank', 'thanks']):
        return "You're welcome! Remember, I'm here to provide general health information. For specific medical concerns, always consult a qualified healthcare professional. Stay healthy!"
    else:
        return "Thank you for your question. While I can provide general health information, I recommend consulting a healthcare professional for specific medical advice. Is there a particular health topic I can help you with?"


@app.route('/api/bmi', methods=['POST'])
def calculate_bmi():
    try:
        data = request.get_json()
        weight = float(data['weight'])
        height = float(data['height']) / 100  # cm to m
        bmi = round(weight / (height ** 2), 1)

        if bmi < 18.5:
            category, risk, color = 'Underweight', 'Low', '#64b5f6'
            tips = ['Increase caloric intake with nutrient-dense foods', 'Add strength training exercises', 'Eat more protein-rich foods', 'Consult a nutritionist']
        elif bmi < 25:
            category, risk, color = 'Normal', 'Healthy', '#69f0ae'
            tips = ['Maintain your current healthy lifestyle', 'Continue regular exercise', 'Eat a balanced diet', 'Stay hydrated']
        elif bmi < 30:
            category, risk, color = 'Overweight', 'Moderate', '#ffd740'
            tips = ['Increase physical activity', 'Reduce processed food intake', 'Monitor portion sizes', 'Consider consulting a dietitian']
        else:
            category, risk, color = 'Obese', 'High', '#ff5252'
            tips = ['Consult a healthcare provider', 'Start a supervised exercise program', 'Follow a calorie-controlled diet', 'Monitor blood pressure and sugar levels']

        save_prediction('BMI Calculator', {'weight': weight, 'height': data['height']}, category, bmi, risk)

        return jsonify({
            'status': 'success',
            'result': {
                'bmi': bmi, 'category': category, 'risk': risk,
                'color': color, 'tips': tips
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate PDF report for a prediction."""
    try:
        from fpdf import FPDF

        data = request.get_json()
        model_name = data.get('model', 'Prediction')
        result_text = data.get('result', '')
        confidence = data.get('confidence', 0)
        details = data.get('details', '')

        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_fill_color(10, 22, 40)
        pdf.rect(0, 0, 210, 45, 'F')
        pdf.set_text_color(212, 168, 67)
        pdf.set_font('Helvetica', 'B', 22)
        pdf.cell(0, 20, 'MedicalCare AI', ln=True, align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(180, 180, 180)
        pdf.cell(0, 8, 'Health Prediction Report', ln=True, align='C')
        pdf.ln(15)

        # Report Info
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'Model: {model_name}', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, f'Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', ln=True)
        if current_user.is_authenticated:
            pdf.cell(0, 8, f'Patient: {current_user.name}', ln=True)
        pdf.ln(8)

        # Result
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 12, '  Prediction Result', ln=True, fill=True)
        pdf.set_font('Helvetica', '', 12)
        pdf.ln(4)
        pdf.cell(0, 8, f'Result: {result_text}', ln=True)
        pdf.cell(0, 8, f'Confidence: {confidence}%', ln=True)
        pdf.ln(4)

        if details:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 10, 'Details:', ln=True)
            pdf.set_font('Helvetica', '', 10)
            for line in str(details).split('\n'):
                pdf.cell(0, 7, f'  {line}', ln=True)

        # Disclaimer
        pdf.ln(15)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(150, 150, 150)
        pdf.multi_cell(0, 5,
            'Disclaimer: This report is generated by an AI prediction system for educational purposes only. '
            'It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. '
            'Always consult a qualified healthcare provider.')

        # Footer
        pdf.ln(5)
        pdf.set_font('Helvetica', '', 8)
        pdf.cell(0, 5, 'Created by B NIKITHA | 25BCE2517 | VIT Vellore', ln=True, align='C')

        pdf_bytes = pdf.output()
        filename = f'MedicalCare_Report_{model_name.replace(" ", "_")}.pdf'

        # Save report to database if user is logged in
        if current_user.is_authenticated:
            saved = SavedReport(
                user_id=current_user.id,
                report_name=filename,
                model_name=model_name,
                result=result_text,
                pdf_data=bytes(pdf_bytes)
            )
            db.session.add(saved)
            db.session.commit()

        return send_file(
            BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/model-info')
def model_info():
    return jsonify({
        'models': [
            {'name': 'Heart Disease Prediction', 'type': 'Gradient Boosting Classifier',
             'category': 'Machine Learning', 'features': len(HeartDiseasePredictor.FEATURE_NAMES), 'trained': heart_model.is_trained},
            {'name': 'Diabetes Prediction', 'type': 'Random Forest Classifier',
             'category': 'Machine Learning', 'features': len(DiabetesPredictor.FEATURE_NAMES), 'trained': diabetes_model.is_trained},
            {'name': 'Disease from Symptoms', 'type': 'Random Forest (Multi-class)',
             'category': 'Machine Learning', 'features': len(DiseasePredictor.SYMPTOM_LIST), 'trained': disease_model.is_trained},
            {'name': 'Breast Cancer Detection', 'type': 'Deep Neural Network',
             'category': 'Deep Learning', 'features': len(BreastCancerPredictor.FEATURE_NAMES), 'trained': cancer_model.is_trained},
            {'name': 'Medical Image Classification', 'type': 'CNN (MobileNetV2)',
             'category': 'Deep Learning', 'features': 'Image 224x224', 'trained': image_classifier.is_trained},
        ]
    })


@app.route('/api/health-news')
def health_news():
    """Return health news/tips."""
    news = [
        {'title': 'WHO: Regular Exercise Reduces Disease Risk by 30%',
         'summary': 'World Health Organization confirms that 150 minutes of weekly moderate exercise significantly reduces risk of chronic diseases.',
         'category': 'Fitness', 'date': 'Feb 2026'},
        {'title': 'AI in Healthcare: Transforming Early Diagnosis',
         'summary': 'Machine learning models are achieving over 95% accuracy in detecting diseases from medical images and patient data.',
         'category': 'Technology', 'date': 'Feb 2026'},
        {'title': 'Mediterranean Diet Linked to Lower Heart Disease Risk',
         'summary': 'New research shows that a Mediterranean-style diet rich in olive oil, nuts, and fish can reduce cardiovascular events by 25%.',
         'category': 'Nutrition', 'date': 'Jan 2026'},
        {'title': 'Mental Health Awareness: Breaking the Stigma',
         'summary': 'Studies show 1 in 4 people experience mental health issues. Early intervention and open conversations can make a significant difference.',
         'category': 'Mental Health', 'date': 'Jan 2026'},
        {'title': 'Breakthrough in Diabetes Management with Continuous Monitoring',
         'summary': 'New continuous glucose monitoring devices provide real-time data, helping patients manage blood sugar levels more effectively.',
         'category': 'Diabetes', 'date': 'Feb 2026'},
        {'title': 'Sleep Quality: The Foundation of Good Health',
         'summary': 'Research confirms that poor sleep increases risk of obesity, diabetes, and cardiovascular disease. Aim for 7-9 hours of quality sleep.',
         'category': 'Wellness', 'date': 'Jan 2026'},
    ]
    return jsonify({'status': 'success', 'news': news})


if __name__ == '__main__':
    if not heart_model.is_trained:
        print("WARNING: Models not trained! Run 'python train.py' first.\n")

    print("=" * 55)
    print("  MedicalCare AI - Health Science Platform")
    print("  Created by: B NIKITHA | 25BCE2517 | VIT Vellore")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)
