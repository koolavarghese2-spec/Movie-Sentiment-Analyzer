import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📥 Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("📥 Downloading NLTK wordnet...")
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Paths
CSV_PATH = "data/imdb_50k.csv"
MODEL_OUT = "model.pkl"
DATA_DIR = "data"

def create_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created data directory: {DATA_DIR}")

def download_sample_data():
    """Create a more comprehensive and balanced sample dataset"""
    sample_data = {
        'review': [
            # Positive reviews (30 examples)
            "This movie was absolutely fantastic! Great acting and plot.",
            "Brilliant cinematography and outstanding performances by the cast.",
            "One of the best movies I've ever seen! Highly recommended.",
            "Amazing visual effects and gripping storyline.",
            "Excellent direction and superb acting performances.",
            "A masterpiece of modern cinema!",
            "Great family movie with positive messages.",
            "Outstanding! A must-watch for all movie lovers.",
            "Beautiful cinematography and emotional depth.",
            "Fantastic movie with great character development.",
            "Incredible storytelling and powerful performances.",
            "Wonderful film that touched my heart.",
            "Superb acting and brilliant screenplay.",
            "Absolutely loved every moment of this movie.",
            "Perfect casting and excellent direction.",
            "Heartwarming story with amazing characters.",
            "The best film of the year, without a doubt.",
            "Captivating from beginning to end.",
            "Excellent movie that exceeded all expectations.",
            "Brilliant writing and outstanding performances.",
            "A cinematic triumph that deserves awards.",
            "Engaging plot with wonderful character development.",
            "Perfect in every way, a true classic.",
            "Outstanding cinematography and soundtrack.",
            "One of the most beautiful films ever made.",
            "Excellent performances by the entire cast.",
            "A remarkable achievement in filmmaking.",
            "Thoroughly enjoyable and well-crafted.",
            "Superb direction and compelling storytelling.",
            "A fantastic movie that everyone should watch.",
            
            # Negative reviews (30 examples)
            "Terrible movie, waste of time. Boring and poorly acted.",
            "I fell asleep halfway through, very disappointing.",
            "Poor storyline and weak character development.",
            "Not worth watching, complete waste of money.",
            "The plot was confusing and the acting was mediocre at best.",
            "Boring and predictable, nothing special.",
            "The worst movie I've seen this year.",
            "Poor script and terrible dialogue.",
            "I couldn't finish it, too boring.",
            "Disappointing sequel that doesn't live up to the original.",
            "Awful acting and terrible plot development.",
            "Complete waste of time and money.",
            "Horrible movie with no redeeming qualities.",
            "Boring, predictable, and poorly executed.",
            "Terrible cinematography and awful script.",
            "Bad movie with terrible acting throughout.",
            "Waste of two hours of my life.",
            "Poorly made and completely uninteresting.",
            "The acting was bad and the story was worse.",
            "Absolutely terrible in every aspect.",
            "Boring film that failed to engage me.",
            "Poor quality movie with bad production values.",
            "Not good at all, very disappointing.",
            "Bad storyline and terrible characters.",
            "Horrible execution of a weak concept.",
            "The worst film I've ever seen.",
            "Terrible from start to finish.",
            "Bad acting and poor direction.",
            "Completely awful and not worth watching.",
            "Poor movie with no entertainment value."
        ],
        'sentiment': [
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(CSV_PATH, index=False)
    print(f"📝 Created sample dataset with {len(df)} reviews at {CSV_PATH}")
    return df

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers, but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_data(path):
    """Load dataset and detect the right column names."""
    print(f"📂 Loading dataset from: {path}")
    
    if not os.path.exists(path):
        print(f"❌ Dataset not found at {path}")
        print("📝 Creating sample dataset...")
        return download_sample_data()
    
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("❌ Failed to load dataset with any encoding")
            return download_sample_data()
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        print("📝 Creating sample dataset...")
        return download_sample_data()
    
    print(f"📊 Dataset shape: {df.shape}")
    
    text_col = None
    label_col = None
    
    text_candidates = ['review', 'text', 'comment', 'content', 'sentence', 'Review', 'Text', 'message', 'feedback']
    label_candidates = ['sentiment', 'label', 'class', 'rating', 'category', 'Sentiment', 'Label', 'emotion', 'score']
    
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break
    
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
    
    if not text_col or not label_col:
        if len(df.columns) >= 2:
            text_col = df.columns[0]
            label_col = df.columns[1]
            print(f"⚠️  Using first two columns: '{text_col}' for text, '{label_col}' for labels")
        else:
            print("❌ Could not identify text and label columns.")
            print("   Creating sample dataset instead...")
            return download_sample_data()
    
    print(f"✅ Using columns: '{text_col}' for text, '{label_col}' for labels")
    
    X = df[text_col].copy()
    y = df[label_col].copy()
    
    return X, y

def handle_missing_data(X, y):
    """Handle missing values in features and target"""
    print("\n🔍 Checking for missing values...")
    print(f"   Missing values in X: {X.isnull().sum()}/{len(X)}")
    print(f"   Missing values in y: {y.isnull().sum()}/{len(y)}")
    
    original_size = len(X)
    mask = ~X.isnull() & ~y.isnull()
    
    X_clean = X[mask].copy()
    y_clean = y[mask].copy()
    
    removed_count = original_size - len(X_clean)
    if removed_count > 0:
        print(f"🗑️  Removed {removed_count} rows with missing values")
        print(f"📊 Remaining samples: {len(X_clean)}")
    
    print("🧹 Cleaning text data...")
    X_clean = X_clean.apply(clean_text)
    
    empty_mask = X_clean.str.strip() != ""
    X_final = X_clean[empty_mask]
    y_final = y_clean[empty_mask]
    
    empty_removed = len(X_clean) - len(X_final)
    if empty_removed > 0:
        print(f"🗑️  Removed {empty_removed} empty text samples")
        print(f"📊 Final dataset size: {len(X_final)} samples")
    
    return X_final, y_final

def preprocess_labels(y):
    """Convert various label formats to consistent positive/negative"""
    print("\n🏷️  Preprocessing labels...")
    print(f"   Original unique labels: {y.unique()}")
    
    y_processed = y.copy()
    y_processed = y_processed.astype(str).str.lower().str.strip()
    
    label_mapping = {
        '0': 'negative', '1': 'positive',
        '0.0': 'negative', '1.0': 'positive',
        '-1': 'negative', '2': 'positive',
        'neg': 'negative', 'negative': 'negative',
        'pos': 'positive', 'positive': 'positive',
        'no': 'negative', 'yes': 'positive',
        'false': 'negative', 'true': 'positive',
        'bad': 'negative', 'good': 'positive',
        'poor': 'negative', 'excellent': 'positive',
        'awful': 'negative', 'great': 'positive',
        'terrible': 'negative', 'awesome': 'positive',
        'horrible': 'negative', 'amazing': 'positive',
        '1': 'negative', '2': 'negative',
        '3': 'positive', '4': 'positive', '5': 'positive'
    }
    
    y_processed = y_processed.map(label_mapping).fillna(y_processed)
    
    try:
        y_numeric = pd.to_numeric(y_processed, errors='ignore')
        if y_numeric.dtype != 'object':
            y_processed = y_numeric.apply(lambda x: 'positive' if x >= 3 else 'negative')
    except:
        pass
    
    unique_labels = y_processed.unique()
    valid_labels = ['positive', 'negative']
    
    invalid_labels = [label for label in unique_labels if label not in valid_labels]
    
    if invalid_labels:
        print(f"⚠️  Warning: Found unexpected labels: {invalid_labels}")
        print("   Attempting to infer sentiment from text...")
        
        for label in invalid_labels:
            label_lower = str(label).lower()
            if any(word in label_lower for word in ['good', 'great', 'excellent', 'awesome', 'amazing', 'love', 'best']):
                y_processed = y_processed.replace(label, 'positive')
            elif any(word in label_lower for word in ['bad', 'poor', 'terrible', 'awful', 'horrible', 'hate', 'worst']):
                y_processed = y_processed.replace(label, 'negative')
    
    final_mask = y_processed.isin(valid_labels)
    y_final = y_processed[final_mask]
    
    removed_final = len(y_processed) - len(y_final)
    if removed_final > 0:
        print(f"🗑️  Removed {removed_final} samples with invalid labels")
    
    print(f"   Processed unique labels: {y_final.unique()}")
    print(f"   Label distribution:\n{y_final.value_counts()}")
    
    return y_final

def train():
    print("🚀 Starting model training...")
    
    create_data_directory()
    
    try:
        X, y = load_data(CSV_PATH)
        X_clean, y_clean = handle_missing_data(X, y)
        
        if len(X_clean) == 0:
            print("❌ No valid data remaining after cleaning!")
            return None
        
        y_processed = preprocess_labels(y_clean)
        
        common_index = X_clean.index.intersection(y_processed.index)
        X_final = X_clean.loc[common_index]
        y_final = y_processed.loc[common_index]
        
        if len(X_final) == 0:
            print("❌ No valid data after label preprocessing!")
            return None
        
        print(f"\n📊 Final dataset size: {len(X_final)} samples")
        print(f"📈 Class distribution:\n{y_final.value_counts()}")
        
        min_samples_per_class = y_final.value_counts().min()
        if min_samples_per_class < 2:
            print(f"⚠️ Not enough samples for stratified split. Using simple split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_final, 
                test_size=0.2, 
                random_state=42
            )
        else:
            print("\n📊 Splitting data with stratification...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_final, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_final
            )

        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        print(f"   Training label distribution:\n{y_train.value_counts()}")

        # Build pipeline with improved parameters
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000, 
                ngram_range=(1, 2), 
                stop_words="english",
                min_df=2,
                max_df=0.85,
                strip_accents='unicode',
                analyzer='word'
            )),
            ("clf", LogisticRegression(
                max_iter=2000, 
                solver="liblinear",
                class_weight='balanced',
                random_state=42,
                C=1.0
            ))
        ])

        print("\n🎯 Training model...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        print("\n📈 Evaluating model...")
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

        # Save model
        joblib.dump(pipeline, MODEL_OUT)
        print(f"💾 Model saved to {os.path.abspath(MODEL_OUT)}")
        
        # Test the model with critical reviews
        test_reviews = [
            "This movie was absolutely fantastic! Great acting and plot.",
            "Terrible movie, waste of time. Boring and poorly acted.",
            "It was okay, nothing special but not bad either.",
            "Brilliant cinematography and outstanding performances by the cast.",
            "I fell asleep halfway through, very disappointing."
        ]
        
        print("\n🧪 Test predictions:")
        for review in test_reviews:
            pred = pipeline.predict([review])[0]
            proba = pipeline.predict_proba([review])[0]
            conf = max(proba)
            print(f"   '{review[:50]}...'")
            print(f"   → Prediction: {pred} (confidence: {conf:.3f})")
            print()
        
        # Critical test for negative reviews
        print("\n🔍 Testing model with negative reviews:")
        negative_test_reviews = [
            "the movie is bad",
            "this is terrible",
            "awful movie", 
            "horrible acting",
            "waste of time",
            "boring and pointless",
            "worst movie ever",
            "complete garbage",
            "disappointing and bad",
            "poor quality film"
        ]

        for review in negative_test_reviews:
            pred = pipeline.predict([review])[0]
            proba = pipeline.predict_proba([review])[0]
            conf = max(proba)
            sentiment = 'positive' if 'pos' in str(pred).lower() else 'negative'
            print(f"   '{review}' → {sentiment} (confidence: {conf:.3f})")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎬 Movie Sentiment Analyzer - Model Training")
    print("=" * 50)
    
    trained_model = train()
    
    if trained_model is not None:
        print("\n🎉 Training completed successfully!")
        print("📊 Your model is ready for use in the Flask application.")
    else:
        print("\n❌ Training failed! Please check the error messages above.")
    
    print("=" * 50)