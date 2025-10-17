"""
IMDB Sentiment Analysis - Complete NLP Pipeline
This script performs comprehensive sentiment analysis on IMDB movie reviews
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Machine Learning Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

print("=" * 80)
print("IMDB SENTIMENT ANALYSIS - COMPLETE NLP PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: DOWNLOAD NLTK RESOURCES
# ============================================================================
print("\n[1/10] Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✓ NLTK resources downloaded successfully")
except Exception as e:
    print(f"Note: {e}")

# ============================================================================
# STEP 2: DATA LOADING
# ============================================================================
print("\n[2/10] Loading dataset...")

def load_imdb_data(data_dir, subset='train'):
    """Load IMDB reviews from directory structure"""
    reviews = []
    labels = []
    
    # Load positive reviews
    pos_dir = Path(data_dir) / subset / 'pos'
    if pos_dir.exists():
        for file in pos_dir.glob('*.txt'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(1)  # Positive
            except:
                pass
    
    # Load negative reviews
    neg_dir = Path(data_dir) / subset / 'neg'
    if neg_dir.exists():
        for file in neg_dir.glob('*.txt'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(0)  # Negative
            except:
                pass
    
    return reviews, labels

# Load training and test data
train_reviews, train_labels = load_imdb_data('aclImdb', 'train')
test_reviews, test_labels = load_imdb_data('aclImdb', 'test')

print(f"✓ Training samples: {len(train_reviews)}")
print(f"✓ Test samples: {len(test_reviews)}")

# Create DataFrames
train_df = pd.DataFrame({'review': train_reviews, 'sentiment': train_labels})
test_df = pd.DataFrame({'review': test_reviews, 'sentiment': test_labels})

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[3/10] Performing Exploratory Data Analysis...")

print("\nDataset Statistics:")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"\nClass distribution (Training):")
print(train_df['sentiment'].value_counts())

# Review length statistics
train_df['review_length'] = train_df['review'].apply(len)
train_df['word_count'] = train_df['review'].apply(lambda x: len(x.split()))

print(f"\nReview Length Statistics:")
print(train_df[['review_length', 'word_count']].describe())

# ============================================================================
# STEP 4: TEXT PREPROCESSING
# ============================================================================
print("\n[4/10] Preprocessing text data...")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

def preprocess_text(text):
    """Complete preprocessing pipeline"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return ""
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Preprocess data
print("Processing training data...")
train_df['cleaned_review'] = train_df['review'].apply(preprocess_text)

print("Processing test data...")
test_df['cleaned_review'] = test_df['review'].apply(preprocess_text)

print("✓ Text preprocessing completed")

# Show example
print("\n--- Example of Preprocessing ---")
print("Original:", train_df['review'].iloc[0][:200])
print("\nCleaned:", train_df['cleaned_review'].iloc[0][:200])

# ============================================================================
# STEP 5: FEATURE EXTRACTION
# ============================================================================
print("\n[5/10] Extracting features using TF-IDF...")

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=5,
    max_df=0.8
)

X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['cleaned_review'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['cleaned_review'])

y_train = train_df['sentiment'].values
y_test = test_df['sentiment'].values

print(f"✓ TF-IDF feature matrix shape: {X_train_tfidf.shape}")
print(f"✓ Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# ============================================================================
# STEP 6: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n[6/10] Training multiple machine learning models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(random_state=42, max_iter=2000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }
    
    print(f"✓ {name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# ============================================================================
# STEP 7: MODEL EVALUATION
# ============================================================================
print("\n[7/10] Evaluating models...")

# Create results DataFrame
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()]
})

results_df = results_df.sort_values('Accuracy', ascending=False)
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))

# Best model
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# ============================================================================
# STEP 8: DETAILED EVALUATION OF BEST MODEL
# ============================================================================
print(f"\n[8/10] Detailed evaluation of {best_model_name}...")

y_pred_best = results[best_model_name]['predictions']

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)

# ============================================================================
# STEP 9: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[9/10] Analyzing important features...")

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# For Logistic Regression, get coefficients
if 'Logistic Regression' in results:
    lr_model = results['Logistic Regression']['model']
    coefficients = lr_model.coef_[0]
    
    # Top positive words (indicating positive sentiment)
    top_positive_indices = np.argsort(coefficients)[-20:]
    top_positive_words = [(feature_names[i], coefficients[i]) for i in top_positive_indices]
    
    # Top negative words (indicating negative sentiment)
    top_negative_indices = np.argsort(coefficients)[:20]
    top_negative_words = [(feature_names[i], coefficients[i]) for i in top_negative_indices]
    
    print("\nTop 20 words indicating POSITIVE sentiment:")
    for word, coef in reversed(top_positive_words):
        print(f"  {word}: {coef:.4f}")
    
    print("\nTop 20 words indicating NEGATIVE sentiment:")
    for word, coef in top_negative_words:
        print(f"  {word}: {coef:.4f}")

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n[10/10] Saving results...")

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("✓ Results saved to 'model_comparison_results.csv'")

# Save preprocessed data sample
sample_df = train_df[['review', 'cleaned_review', 'sentiment']].head(100)
sample_df.to_csv('preprocessing_sample.csv', index=False)
print("✓ Preprocessing sample saved to 'preprocessing_sample.csv'")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
print("\n" + "=" * 80)
print("PREDICTION FUNCTION READY")
print("=" * 80)

def predict_sentiment(text):
    """Predict sentiment of a new review"""
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    text_vector = tfidf_vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = best_model.predict(text_vector)[0]
    
    # Get probability if available
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(text_vector)[0]
        confidence = max(probability)
    else:
        confidence = None
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence

# Test the prediction function
print("\n--- Testing Prediction Function ---")
test_reviews_examples = [
    "This movie was absolutely amazing! Best film I've seen this year.",
    "Terrible movie, waste of time and money. Very disappointed.",
    "It was okay, nothing special but not bad either."
]

for review in test_reviews_examples:
    sentiment, confidence = predict_sentiment(review)
    conf_str = f" (confidence: {confidence:.2%})" if confidence else ""
    print(f"\nReview: {review}")
    print(f"Predicted: {sentiment}{conf_str}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nSummary:")
print(f"- Total training samples: {len(train_df)}")
print(f"- Total test samples: {len(test_df)}")
print(f"- Best model: {best_model_name}")
print(f"- Best accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"- Feature vector size: {X_train_tfidf.shape[1]}")
print("\n✓ You can now use predict_sentiment() function to classify new reviews!")