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
    """
    Preprocess email text: lowercase, remove punctuation, numbers, stopwords, stem
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Stem words
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def extract_features(text):
    """
    Extract additional features from email text
    """
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
    """
    Train both Naive Bayes and SVM models on the dataset
    """
    global nb_model, svm_model, vectorizer
    
    print("Loading dataset...")
    try:
        # Try to load the dataset
       # New, robust line:  
        df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'emails.csv'))
        print(f"Dataset loaded: {len(df)} emails")
        
        # Handle both 'label' and 'lable' column names
        if 'lable' in df.columns:
            df.rename(columns={'lable': 'label'}, inplace=True)
        
        emails = df['text'].tolist()
        labels = df['label'].tolist()
        
    except FileNotFoundError:
        print("Warning: data/emails.csv not found. Using sample data for demo.")
        # Fallback sample data
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
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(processed_emails)
    
    # Train Naive Bayes
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X, labels)
    print("✓ Naive Bayes model trained")
    
    # Train SVM
    svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    svm_model.fit(X, labels)
    print("✓ SVM model trained")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    with open('models/nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Models saved to 'models' directory")
def load_models():
    """
    Load pre-trained models from disk
    """
    global nb_model, svm_model, vectorizer
    
    # Define the base path for the models folder relative to the current working directory
    # Since Render's root is set to 'backend', we just look in 'models/'
    base_model_path = os.path.join(os.getcwd(), 'models')
    
    try:
        print("Loading pre-trained models...")
        
        # Construct the full file paths
        nb_path = os.path.join(base_model_path, 'nb_model.pkl')
        svm_path = os.path.join(base_model_path, 'svm_model.pkl')
        vec_path = os.path.join(base_model_path, 'vectorizer.pkl')

        # Check if the files exist before trying to open them
        if not os.path.exists(nb_path):
            # This will trigger the FileNotFoundError exception
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
        # Your models weren't found, so it attempts to train them
        train_models()
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/analyze', methods=['POST'])
def analyze_email():
    """
    API endpoint to analyze email text
    """
    try:
        data = request.get_json()
        email_text = data.get('emailText', '')
        model_type = data.get('model', 'naive_bayes')
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Preprocess the email
        processed_text = preprocess_text(email_text)
        
        # Vectorize the text
        X = vectorizer.transform([processed_text])
        
        # Select model
        model = nb_model if model_type == 'naive_bayes' else svm_model
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get prediction probability (confidence)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            # Get confidence for the predicted class
            if prediction == 'spam':
                confidence = proba[1] * 100  # Probability of spam
            else:
                confidence = proba[0] * 100  # Probability of ham
        else:
            # For models without predict_proba, use decision function
            confidence = 85.0  # Default confidence
        
        # Extract additional features
        features = extract_features(email_text)
        
        # Prepare response
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
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'models_loaded': nb_model is not None and svm_model is not None
    }), 200

@app.route('/retrain', methods=['POST'])
def retrain_models():
    """
    Endpoint to retrain models with new data
    """
    try:
        train_models()
        return jsonify({'message': 'Models retrained successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("SPAM DETECTOR API SERVER")
    print("="*60)
    
    # Load or train models on startup
    load_models()
    
    print("\n" + "="*60)
    print("🚀 Server starting on http://localhost:5000")
    print("="*60)
    print("\nAvailable endpoints:")
    print("  POST /analyze    - Analyze email text")
    print("  GET  /health     - Health check")
    print("  POST /retrain    - Retrain models")
    print("\n" + "="*60)
    
    # Run the Flask app

port = int(os.environ.get("PORT", 10000))
app.run(host='0.0.0.0', port=port)
