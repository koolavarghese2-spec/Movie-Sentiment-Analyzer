# test_installation.py
import sys

print(f"Python version: {sys.version}")

packages = [
    'flask', 'flask_sqlalchemy', 'flask_login', 'werkzeug',
    'pandas', 'numpy', 'sklearn', 'nltk', 'joblib'
]

for package in packages:
    try:
        if package == 'sklearn':
            mod = __import__('sklearn')
            version = mod.__version__
        else:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package}: {version}")
    except ImportError as e:
        print(f"❌ {package}: {e}")