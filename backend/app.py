from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables to store models
nb_model = None
svm_model = None
vectorizer = None
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def extract_features(text):
    spam_keywords = [
        'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 'congratulations',
        'click here', 'limited time', 'act now', 'offer', 'discount', 'guarantee',
        'money', 'credit', 'loan', 'viagra', 'pharmacy', 'weight loss', 'mlm'
    ]
    lower_text = text.lower()
    found_keywords = [keyword for keyword in spam_keywords if keyword in lower_text]
    exclamation_count = text.count('!')
    all_caps_words = len([word for word in text.split() if len(word) > 2 and word.isupper()])
    return {
        'keywordCount': len(found_keywords),
        'exclamationMarks': exclamation_count,
        'capsWords': all_caps_words,
        'textLength': len(text),
        'foundKeywords': found_keywords
    }

def train_models():
    global nb_model, svm_model, vectorizer
    print("Loading dataset...")
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'emails.csv'))
        print(f"Dataset loaded: {len(df)} emails")
        if 'lable' in df.columns:
            df.rename(columns={'lable': 'label'}, inplace=True)
        emails = df['text'].tolist()
        labels = df['label'].tolist()
    except FileNotFoundError:
        print("Warning: data/emails.csv not found. Using sample data for demo.")
        emails = [
            "Congratulations! You've won a free iPhone. Click here to claim now!",
            "Hi John, can we schedule a meeting tomorrow at 3pm?",
            "URGENT: Your account will be closed. Verify your information immediately!",
            "Hey, thanks for sending the report. I'll review it by end of day.",
            "Make money fast! Work from home and earn $5000 per week!",
            "The project deadline has been extended to next Friday.",
            "Get viagra and cialis at lowest prices! No prescription needed!",
            "Please find attached the quarterly financial statements.",
        ]
        labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
    print("Preprocessing text...")
    processed_emails = [preprocess_text(email) for email in emails]
    print("Training models...")
    vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(processed_emails)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X, labels)
    print("✓ Naive Bayes model trained")
    svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    svm_model.fit(X, labels)
    print("✓ SVM model trained")

    # ✅ Use absolute path relative to this file
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'nb_model.pkl'), 'wb') as f:
        pickle.dump(nb_model, f)
    with open(os.path.join(model_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Models saved to 'models' directory")

def load_models():
    global nb_model, svm_model, vectorizer
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        print("Loading pre-trained models...")
        nb_path = os.path.join(model_dir, 'nb_model.pkl')
        svm_path = os.path.join(model_dir, 'svm_model.pkl')
        vec_path = os.path.join(model_dir, 'vectorizer.pkl')
        if not os.path.exists(nb_path):
            raise FileNotFoundError(f"Model file not found at: {nb_path}")
        with open(nb_path, 'rb') as f:
            nb_model = pickle.load(f)
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Models loaded successfully")
        return True
    except FileNotFoundError:
        print("Models not found. Training new models...")
        train_models()
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/analyze', methods=['POST'])
def analyze_email():
    try:
        data = request.get_json()
        email_text = data.get('emailText', '')
        model_type = data.get('model', 'naive_bayes')
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        processed_text = preprocess_text(email_text)
        X = vectorizer.transform([processed_text])
        model = nb_model if model_type == 'naive_bayes' else svm_model
        prediction = model.predict(X)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = proba[1] * 100 if prediction == 'spam' else proba[0] * 100
        else:
            confidence = 85.0
        features = extract_features(email_text)
        response = {
            'analysis': {
                'isSpam': prediction == 'spam',
                'confidence': float(confidence),
                'features': features,
                'foundKeywords': features['foundKeywords'],
                'model_used': model_type
            }
        }
        return jsonify(response), 200
    except Exception as e:
        print(f"Error in analyze_email: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': nb_model is not None and svm_model is not None
    }), 200

@app.route('/retrain', methods=['POST'])
def retrain_models():
    try:
        train_models()
        return jsonify({'message': 'Models retrained successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("SPAM DETECTOR API SERVER")
    print("="*60)
    load_models()
    print("\n" + "="*60)
    print("🚀 Server starting on http://localhost:5000")
    print("="*60)
    print("\nAvailable endpoints:")
    print("  POST /analyze    - Analyze email text")
    print("  GET  /health     - Health check")
    print("  POST /retrain    - Retrain models")
    print("\n" + "="*60)

port = int(os.environ.get("PORT", 10000))
app.run(host='0.0.0.0', port=port)
