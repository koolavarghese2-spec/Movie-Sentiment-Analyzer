# setup.py
import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Movie Sentiment Analyzer...")
    print("=" * 50)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    print("✅ Created data directory")
    
    # Install requirements
    print("📦 Installing dependencies...")
    if not run_command('pip install -r requirements.txt'):
        print("❌ Failed to install dependencies")
        return
    
    # Create a simple model
    print("🤖 Creating a simple model...")
    try:
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        # Sample data
        data = {
            'review': [
                "Great movie with amazing acting!",
                "Terrible film, waste of time.",
                "Excellent storyline and characters.",
                "Boring and poorly made.",
                "Outstanding cinematography!",
                "Awful script and bad acting."
            ],
            'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
        }
        
        df = pd.DataFrame(data)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        
        pipeline.fit(df['review'], df['sentiment'])
        joblib.dump(pipeline, 'model.pkl')
        print("✅ Model created and saved as model.pkl")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Run: python app.py")
    print("2. Open: http://127.0.0.1:5000")
    print("3. Use test credentials: username='testuser', password='testpassword123'")

if __name__ == "__main__":
    main()