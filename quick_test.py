# quick_test.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("🚀 Creating a simple sentiment analysis model...")

# Create sample training data
data = {
    'review': [
        "This movie was absolutely fantastic! Great acting and engaging plot.",
        "Terrible movie, complete waste of time. Poor acting and boring story.",
        "Excellent cinematography and outstanding performances by the cast.",
        "I fell asleep halfway through, very disappointing and poorly made.",
        "One of the best movies I've ever seen! Highly recommended to everyone.",
        "Awful script and terrible acting made this unwatchable.",
        "Brilliant direction and superb character development throughout.",
        "Boring storyline with uninteresting characters and weak dialogue.",
        "Amazing visual effects and a gripping, emotional storyline.",
        "Poorly executed with bad pacing and unconvincing performances.",
        "A masterpiece of modern cinema with incredible depth and meaning.",
        "Predictable plot and mediocre acting throughout the entire film.",
        "Great family movie with positive messages and heartwarming scenes.",
        "The worst movie I've seen this year, absolutely terrible.",
        "Outstanding performances and a compelling narrative from start to finish.",
        "Confusing plot and terrible character development.",
        "Beautiful cinematography and powerful emotional moments.",
        "I couldn't finish watching it, too boring and poorly made.",
        "Fantastic character arcs and excellent pacing throughout the film.",
        "Disappointing sequel that doesn't live up to the original masterpiece."
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 
                  'negative', 'positive', 'negative', 'positive', 'negative',
                  'positive', 'negative', 'positive', 'negative', 'positive',
                  'negative', 'positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

print(f"📊 Training dataset: {len(df)} reviews")
print(f"📈 Class distribution:\n{df['sentiment'].value_counts()}")

# Create and train the model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LogisticRegression(random_state=42))
])

X = df['review']
y = df['sentiment']

print("🎯 Training the model...")
pipeline.fit(X, y)

# Test the model with various examples
test_reviews = [
    "This movie was fantastic with great acting!",
    "Terrible film, complete waste of money.",
    "Amazing storyline and brilliant performances!",
    "Boring and poorly executed.",
    "Outstanding cinematography and direction!"
]

print("\n🧪 Testing the model:")
for review in test_reviews:
    prediction = pipeline.predict([review])[0]
    confidence = max(pipeline.predict_proba([review])[0])
    print(f"   '{review}'")
    print(f"   → {prediction.upper()} (confidence: {confidence:.3f})")
    print()

# Save the model
joblib.dump(pipeline, 'model.pkl')
print("✅ Model successfully saved as 'model.pkl'")
print("🎉 You're ready to run the web application!")