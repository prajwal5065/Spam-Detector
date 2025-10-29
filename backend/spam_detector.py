import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpamDetector:
    def __init__(self, classifier_type='naive_bayes'):
        """
        Initialize spam detector with chosen classifier
        classifier_type: 'naive_bayes' or 'svm'
        """
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        if classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=1.0)
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='linear', C=1.0, random_state=42)
        else:
            raise ValueError("classifier_type must be 'naive_bayes' or 'svm'")
    
    def preprocess_text(self, text):
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
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Stem words
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def extract_features(self, emails):
        """
        Extract TF-IDF features from emails
        """
        processed_emails = [self.preprocess_text(email) for email in emails]
        return processed_emails
    
    def train(self, X_train, y_train):
        """
        Train the spam detector
        """
        # Preprocess emails
        X_train_processed = self.extract_features(X_train)
        
        # Fit vectorizer and transform to features
        X_train_features = self.vectorizer.fit_transform(X_train_processed)
        
        # Train classifier
        self.classifier.fit(X_train_features, y_train)
        
        print(f"{self.classifier_type.upper()} model trained successfully!")
        return self
    
    def predict(self, X_test):
        """
        Predict spam/ham for new emails
        """
        # Preprocess emails
        X_test_processed = self.extract_features(X_test)
        
        # Transform to features
        X_test_features = self.vectorizer.transform(X_test_processed)
        
        # Predict
        predictions = self.classifier.predict(X_test_features)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label='spam')
        recall = recall_score(y_test, predictions, pos_label='spam')
        f1 = f1_score(y_test, predictions, pos_label='spam')
        
        print(f"\n{'='*50}")
        print(f"{self.classifier_type.upper()} Model Evaluation")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# Example usage with sample data
def main():
    # Sample email dataset (you should replace with actual dataset)
    emails = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hi John, can we schedule a meeting tomorrow at 3pm?",
        "URGENT: Your account will be closed. Verify your information immediately!",
        "Hey, thanks for sending the report. I'll review it by end of day.",
        "Make money fast! Work from home and earn $5000 per week!",
        "The project deadline has been extended to next Friday.",
        "Get viagra and cialis at lowest prices! No prescription needed!",
        "Please find attached the quarterly financial statements.",
        "You are the lucky winner! Claim your prize of $1,000,000 now!",
        "Let's catch up over coffee this weekend. Are you free?",
        "SPECIAL OFFER: 70% discount on all luxury watches! Limited time!",
        "The team meeting has been rescheduled to Monday at 10am.",
        "Your Amazon order has been shipped and will arrive tomorrow.",
        "FREE GIFT! Click here to receive your complimentary vacation package!",
        "Can you review the presentation slides before tomorrow's client meeting?",
    ]
    
    labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 
              'spam', 'ham', 'spam', 'ham', 'ham', 'spam', 'ham']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print("Dataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train and evaluate Naive Bayes
    print("\n" + "="*50)
    print("TRAINING NAIVE BAYES CLASSIFIER")
    print("="*50)
    nb_detector = SpamDetector(classifier_type='naive_bayes')
    nb_detector.train(X_train, y_train)
    nb_results = nb_detector.evaluate(X_test, y_test)
    
    # Train and evaluate SVM
    print("\n" + "="*50)
    print("TRAINING SVM CLASSIFIER")
    print("="*50)
    svm_detector = SpamDetector(classifier_type='svm')
    svm_detector.train(X_train, y_train)
    svm_results = svm_detector.evaluate(X_test, y_test)
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Naive Bayes Accuracy: {nb_results['accuracy']:.4f}")
    print(f"SVM Accuracy:         {svm_results['accuracy']:.4f}")
    
    # Test with new emails
    print("\n" + "="*50)
    print("TESTING WITH NEW EMAILS")
    print("="*50)
    new_emails = [
        "Congratulations! You've been selected for a free cruise!",
        "Can you send me the meeting notes from yesterday?"
    ]
    
    for email in new_emails:
        nb_pred = nb_detector.predict([email])[0]
        svm_pred = svm_detector.predict([email])[0]
        print(f"\nEmail: {email}")
        print(f"Naive Bayes prediction: {nb_pred}")
        print(f"SVM prediction: {svm_pred}")


if __name__ == "__main__":
    main()