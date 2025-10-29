from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
from datetime import datetime
from sqlalchemy import func

app = Flask(__name__)
app.config['SECRET_KEY'] = 'movie-sentiment-analyzer-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_analyzer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy()
login_manager = LoginManager()

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviews = db.relationship('Review', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_title = db.Column(db.String(200), nullable=False)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'movieTitle': self.movie_title,
            'review': self.text[:100] + '...' if len(self.text) > 100 else self.text,
            'fullReview': self.text,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'sentiment': self.sentiment,
            'timestamp': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    theme = db.Column(db.String(20), default='default')
    notifications = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load model with error handling
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please run train_model.py first to create the model.")
    model = None

# Authentication Routes
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400

        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=True)
            return jsonify({
                'success': True, 
                'message': 'Login successful!',
                'user': {
                    'username': user.username,
                    'email': user.email
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Login error: {str(e)}'}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()

        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters long'}), 400

        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters long'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user, remember=True)
        
        return jsonify({
            'success': True, 
            'message': 'Registration successful!',
            'user': {
                'username': new_user.username,
                'email': new_user.email
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'}), 500

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully!'})

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    text = data.get("text", "").strip()
    movie_title = data.get("movieTitle", "").strip()

    if not text:
        return jsonify({"error": "No review text provided"}), 400

    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400

    try:
        prediction = model.predict([text])[0]
        
        if hasattr(model, 'predict_proba'):
            confidence_scores = model.predict_proba([text])[0]
            confidence = max(confidence_scores)
        else:
            confidence = 1.0
        
        prediction_str = str(prediction).lower()
        
        # Manual override for clearly negative words with low confidence
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'boring', 'waste', 'worst', 'garbage', 'disappointing', 'poor']
        text_lower = text.lower()
        
        # If text contains clearly negative words and confidence is low, override to negative
        contains_negative = any(word in text_lower for word in negative_words)
        
        if contains_negative and confidence < 0.7:
            sentiment = 'negative'
            # Adjust confidence for manual override
            confidence = max(confidence, 0.6)
            prediction_str = 'negative'  # Update prediction string for consistency
        else:
            is_positive = 'pos' in prediction_str
            sentiment = 'positive' if is_positive else 'negative'
            
        # Only save to database if user is logged in
        if current_user.is_authenticated:
            review = Review(
                movie_title=movie_title,
                text=text,
                prediction=str(prediction),
                confidence=float(confidence),
                sentiment=sentiment,
                user_id=current_user.id
            )
            db.session.add(review)
            db.session.commit()
        
        return jsonify({
            "prediction": str(prediction),
            "confidence": float(confidence),
            "sentiment": sentiment,
            "movieTitle": movie_title
        })
    except Exception as e:
        if current_user.is_authenticated:
            db.session.rollback()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/history')
@login_required
def get_history():
    try:
        reviews = Review.query.filter_by(user_id=current_user.id).order_by(Review.created_at.desc()).limit(50).all()
        history = [review.to_dict() for review in reviews]
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": f"Failed to load history: {str(e)}"}), 500

@app.route('/api/user')
@login_required
def get_user():
    return jsonify({
        'username': current_user.username,
        'email': current_user.email
    })

@app.route('/api/stats')
@login_required
def get_stats():
    try:
        total_reviews = Review.query.filter_by(user_id=current_user.id).count()
        positive_reviews = Review.query.filter_by(user_id=current_user.id, sentiment='positive').count()
        negative_reviews = Review.query.filter_by(user_id=current_user.id, sentiment='negative').count()
        
        popular_movies = db.session.query(
            Review.movie_title,
            func.count(Review.id).label('review_count')
        ).filter_by(user_id=current_user.id).group_by(Review.movie_title).order_by(func.count(Review.id).desc()).limit(5).all()
        
        popular_movies_list = [{'title': movie[0], 'count': movie[1]} for movie in popular_movies]
        
        return jsonify({
            'totalReviews': total_reviews,
            'positiveCount': positive_reviews,
            'negativeCount': negative_reviews,
            'popularMovies': popular_movies_list
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load stats: {str(e)}"}), 500

# Clear user data endpoint
@app.route('/api/clear-data', methods=['POST'])
@login_required
def clear_user_data():
    """Clear all reviews and data for the current user"""
    try:
        # Delete all reviews for the current user
        deleted_count = Review.query.filter_by(user_id=current_user.id).delete()
        
        # Reset user settings to default (optional)
        settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        if settings:
            settings.theme = 'default'
            settings.notifications = True
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully cleared {deleted_count} reviews and reset settings!'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Failed to clear data: {str(e)}'}), 500

# Delete user account endpoint
@app.route('/api/delete-account', methods=['POST'])
@login_required
def delete_account():
    """Permanently delete user account and all associated data"""
    try:
        user_id = current_user.id
        
        # Get user data before deletion for confirmation
        user_data = {
            'username': current_user.username,
            'email': current_user.email,
            'review_count': len(current_user.reviews)
        }
        
        # Delete all reviews by the user
        Review.query.filter_by(user_id=user_id).delete()
        
        # Delete user settings
        UserSettings.query.filter_by(user_id=user_id).delete()
        
        # Delete the user account
        db.session.delete(current_user)
        db.session.commit()
        
        # Logout the user after account deletion
        logout_user()
        
        return jsonify({
            'success': True,
            'message': f'Account "{user_data["username"]}" has been permanently deleted. {user_data["review_count"]} reviews were removed.',
            'deleted_user': user_data
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Failed to delete account: {str(e)}'}), 500

# Settings Routes - Available without login
@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        # For non-logged in users, return default settings
        if not current_user.is_authenticated:
            return jsonify({
                'theme': 'default',
                'notifications': True
            })
        
        settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        if not settings:
            # Create default settings
            settings = UserSettings(user_id=current_user.id)
            db.session.add(settings)
            db.session.commit()
        
        return jsonify({
            'theme': settings.theme,
            'notifications': settings.notifications
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load settings: {str(e)}"}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # For non-logged in users, just return success (settings stored in localStorage)
        if not current_user.is_authenticated:
            return jsonify({
                'success': True,
                'message': 'Settings updated successfully!',
                'settings': {
                    'theme': data.get('theme', 'default'),
                    'notifications': data.get('notifications', True)
                }
            })
        
        settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        if not settings:
            settings = UserSettings(user_id=current_user.id)
            db.session.add(settings)
        
        # Update settings
        if 'theme' in data:
            settings.theme = data['theme']
        if 'notifications' in data:
            settings.notifications = data['notifications']
        
        settings.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully!',
            'settings': {
                'theme': settings.theme,
                'notifications': settings.notifications
            }
        })
    except Exception as e:
        if current_user.is_authenticated:
            db.session.rollback()
        return jsonify({"error": f"Failed to update settings: {str(e)}"}), 500

# Beautiful Database Admin Dashboard with Delete Functionality
@app.route('/admin/database')
def view_database():
    """Admin page with simple password protection and delete functionality"""
    # Check for admin password
    admin_password = request.args.get('password')
    expected_password = 'admin123'  # Change this to whatever you want
    
    if admin_password != expected_password:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Login - Movie Sentiment Analyzer</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 0;
                }
                .login-container {
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    text-align: center;
                    max-width: 400px;
                    width: 90%;
                }
                h2 {
                    color: #333;
                    margin-bottom: 10px;
                }
                p {
                    color: #666;
                    margin-bottom: 30px;
                }
                input {
                    width: 100%;
                    padding: 15px;
                    border: 2px solid #e1e5e9;
                    border-radius: 10px;
                    font-size: 16px;
                    margin-bottom: 20px;
                    box-sizing: border-box;
                }
                button {
                    width: 100%;
                    padding: 15px;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }
                .password-hint {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                    font-size: 14px;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h2>🔒 Admin Access Required</h2>
                <p>Please enter the admin password to continue</p>
                <form method="GET">
                    <input type="password" name="password" placeholder="Enter admin password" required>
                    <button type="submit">Access Dashboard</button>
                </form>
                <div class="password-hint">
                    <strong>Default Password:</strong> admin123<br>
                    <small>You can change this in the app.py file</small>
                </div>
            </div>
        </body>
        </html>
        ''', 401

    # Get action parameters
    action = request.args.get('action')
    user_id = request.args.get('user_id')
    
    # Handle delete action
    if action == 'delete' and user_id:
        try:
            user = User.query.get(int(user_id))
            if user:
                username = user.username
                review_count = len(user.reviews)
                
                # Delete all user reviews
                Review.query.filter_by(user_id=user.id).delete()
                
                # Delete user settings
                UserSettings.query.filter_by(user_id=user.id).delete()
                
                # Delete the user
                db.session.delete(user)
                db.session.commit()
                
                # Show success message
                return f'''
                <script>
                    alert('✅ Successfully deleted user: {username}\\n🗑️ Removed {review_count} reviews');
                    window.location.href = '/admin/database?password={admin_password}';
                </script>
                '''
            else:
                return '''
                <script>
                    alert('❌ User not found');
                    window.location.href = '/admin/database?password=' + new URLSearchParams(window.location.search).get('password');
                </script>
                '''
        except Exception as e:
            db.session.rollback()
            return f'''
            <script>
                alert('❌ Error deleting user: {str(e)}');
                window.location.href = '/admin/database?password={admin_password}';
            </script>
            '''

    # Rest of admin dashboard code
    users = User.query.all()
    reviews = Review.query.all()
    settings = UserSettings.query.all()
    
    users_data = []
    for user in users:
        users_data.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'review_count': len(user.reviews)
        })
    
    reviews_data = []
    for review in reviews:
        reviews_data.append({
            'id': review.id,
            'movie_title': review.movie_title,
            'review_preview': review.text[:50] + '...' if len(review.text) > 50 else review.text,
            'sentiment': review.sentiment,
            'confidence': f"{review.confidence:.2%}",
            'confidence_value': review.confidence,
            'user_id': review.user_id,
            'username': review.user.username if review.user else 'Unknown',
            'created_at': review.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    settings_data = []
    for setting in settings:
        user = User.query.get(setting.user_id)
        settings_data.append({
            'id': setting.id,
            'user_id': setting.user_id,
            'username': user.username if user else 'Unknown',
            'theme': setting.theme,
            'notifications': setting.notifications,
            'updated_at': setting.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Build HTML in parts
    html_parts = []
    
    # HTML Header
    html_parts.append('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Database Admin - Movie Sentiment Analyzer</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }

            .admin-container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }

            .admin-header {
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .admin-header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }

            .admin-header p {
                opacity: 0.9;
                font-size: 1.1rem;
            }

            .stats-overview {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
            }

            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }

            .stat-number {
                font-size: 2.5rem;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
            }

            .stat-label {
                color: #666;
                font-size: 1rem;
                font-weight: 500;
            }

            .tabs {
                display: flex;
                background: #2c3e50;
                padding: 0;
            }

            .tab {
                flex: 1;
                padding: 20px;
                text-align: center;
                background: none;
                border: none;
                color: white;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                border-bottom: 3px solid transparent;
            }

            .tab:hover {
                background: rgba(255,255,255,0.1);
            }

            .tab.active {
                background: rgba(255,255,255,0.1);
                border-bottom-color: #667eea;
            }

            .tab-content {
                display: none;
                padding: 30px;
                max-height: 600px;
                overflow-y: auto;
            }

            .tab-content.active {
                display: block;
            }

            .data-table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }

            .data-table th {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }

            .data-table td {
                padding: 15px;
                border-bottom: 1px solid #e1e5e9;
                font-size: 14px;
            }

            .data-table tr:hover {
                background: #f8f9fa;
            }

            .data-table tr:last-child td {
                border-bottom: none;
            }

            .badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
            }

            .badge.positive {
                background: #d4edda;
                color: #155724;
            }

            .badge.negative {
                background: #f8d7da;
                color: #721c24;
            }

            .badge.user {
                background: #e3f2fd;
                color: #1565c0;
            }

            .badge.settings {
                background: #fff3cd;
                color: #856404;
            }

            .user-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 16px;
                margin-right: 10px;
            }

            .user-info {
                display: flex;
                align-items: center;
            }

            .search-box {
                padding: 15px;
                background: #f8f9fa;
                border-bottom: 1px solid #e1e5e9;
            }

            .search-input {
                width: 100%;
                padding: 12px 20px;
                border: 2px solid #e1e5e9;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s ease;
            }

            .search-input:focus {
                border-color: #667eea;
            }

            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #666;
            }

            .empty-state i {
                font-size: 3rem;
                margin-bottom: 20px;
                opacity: 0.5;
            }

            @media (max-width: 768px) {
                .admin-container {
                    margin: 10px;
                    border-radius: 10px;
                }

                .admin-header h1 {
                    font-size: 2rem;
                }

                .stats-overview {
                    grid-template-columns: 1fr;
                    padding: 20px;
                }

                .tabs {
                    flex-direction: column;
                }

                .data-table {
                    font-size: 12px;
                }

                .data-table th,
                .data-table td {
                    padding: 10px 8px;
                }
            }

            .timestamp {
                font-size: 12px;
                color: #666;
                font-family: 'Courier New', monospace;
            }

            .confidence-bar {
                background: #e9ecef;
                border-radius: 10px;
                height: 8px;
                margin-top: 5px;
                overflow: hidden;
                width: 100px;
            }

            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                border-radius: 10px;
            }

            .action-buttons {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            .delete-btn {
                background: #dc3545;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.3s ease;
            }

            .delete-btn:hover {
                background: #c82333;
                transform: scale(1.05);
            }

            .admin-actions {
                background: #f8f9fa;
                padding: 20px;
                border-bottom: 1px solid #e1e5e9;
                display: flex;
                gap: 15px;
                align-items: center;
            }

            .admin-warning {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                color: #856404;
            }
        </style>
    </head>
    <body>
        <div class="admin-container">
            <div class="admin-header">
                <h1>🎬 Database Admin Dashboard</h1>
                <p>Movie Sentiment Analyzer - System Overview</p>
            </div>

            <div class="admin-actions">
                <strong>🔧 Admin Actions:</strong>
                <button onclick="refreshData()" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    🔄 Refresh Data
                </button>
                <button onclick="exportData()" style="padding: 8px 16px; background: #17a2b8; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    📊 Export Data
                </button>
            </div>

            <div class="stats-overview">
                <div class="stat-card">
                    <div class="stat-number">''' + str(len(users_data)) + '''</div>
                    <div class="stat-label">Total Users</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">''' + str(len(reviews_data)) + '''</div>
                    <div class="stat-label">Total Reviews</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">''' + str(len(settings_data)) + '''</div>
                    <div class="stat-label">User Settings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">''' + str(sum(user['review_count'] for user in users_data)) + '''</div>
                    <div class="stat-label">All User Reviews</div>
                </div>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('users')">👥 Users</button>
                <button class="tab" onclick="switchTab('reviews')">📝 Reviews</button>
                <button class="tab" onclick="switchTab('settings')">⚙️ Settings</button>
            </div>

            <div class="search-box">
                <input type="text" id="searchInput" class="search-input" placeholder="Search across all data..." onkeyup="filterTable()">
            </div>
    ''')
    
    # Users Tab with Delete Buttons (WARNING REMOVED)
    html_parts.append('''
            <!-- Users Tab -->
            <div id="users-tab" class="tab-content active">
    ''')
    
    if users_data:
        html_parts.append('''
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>User</th>
                            <th>Email</th>
                            <th>Reviews</th>
                            <th>Joined</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        ''')
        
        for user in users_data:
            html_parts.append(f'''
                        <tr>
                            <td>
                                <div class="user-info">
                                    <div class="user-avatar">{user['username'][0].upper()}</div>
                                    <div>
                                        <strong>{user['username']}</strong>
                                        <div style="font-size: 12px; color: #666;">ID: #{user['id']}</div>
                                    </div>
                                </div>
                            </td>
                            <td>{user['email']}</td>
                            <td>
                                <span class="badge user">{user['review_count']} reviews</span>
                            </td>
                            <td class="timestamp">{user['created_at']}</td>
                            <td>
                                <div class="action-buttons">
                                    <button class="delete-btn" onclick="deleteUser({user['id']}, '{user['username']}', {user['review_count']})">
                                        🗑️ Delete
                                    </button>
                                </div>
                            </td>
                        </tr>
            ''')
        
        html_parts.append('''
                    </tbody>
                </table>
        ''')
    else:
        html_parts.append('''
                <div class="empty-state">
                    <div>👥</div>
                    <h3>No Users Found</h3>
                    <p>There are no users in the database yet.</p>
                </div>
        ''')
    
    html_parts.append('''
            </div>
    ''')
    
    # Reviews Tab
    html_parts.append('''
            <!-- Reviews Tab -->
            <div id="reviews-tab" class="tab-content">
    ''')
    
    if reviews_data:
        html_parts.append('''
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Movie</th>
                            <th>Review Preview</th>
                            <th>Sentiment</th>
                            <th>Confidence</th>
                            <th>User</th>
                            <th>Date</th>
                        </tr>
                    </thead>
                    <tbody>
        ''')
        
        for review in reviews_data:
            sentiment_class = 'positive' if review['sentiment'] == 'positive' else 'negative'
            html_parts.append(f'''
                        <tr>
                            <td><strong>{review['movie_title']}</strong></td>
                            <td>"{review['review_preview']}"</td>
                            <td>
                                <span class="badge {sentiment_class}">
                                    {review['sentiment']}
                                </span>
                            </td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span>{review['confidence']}</span>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: {review['confidence_value'] * 100}%"></div>
                                    </div>
                                </div>
                            </td>
                            <td>
                                <span class="badge user">{review['username']}</span>
                            </td>
                            <td class="timestamp">{review['created_at']}</td>
                        </tr>
            ''')
        
        html_parts.append('''
                    </tbody>
                </table>
        ''')
    else:
        html_parts.append('''
                <div class="empty-state">
                    <div>📝</div>
                    <h3>No Reviews Found</h3>
                    <p>There are no reviews in the database yet.</p>
                </div>
        ''')
    
    html_parts.append('''
            </div>
    ''')
    
    # Settings Tab
    html_parts.append('''
            <!-- Settings Tab -->
            <div id="settings-tab" class="tab-content">
    ''')
    
    if settings_data:
        html_parts.append('''
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>User</th>
                            <th>Theme</th>
                            <th>Notifications</th>
                            <th>Last Updated</th>
                            <th>Setting ID</th>
                        </tr>
                    </thead>
                    <tbody>
        ''')
        
        for setting in settings_data:
            notification_class = 'positive' if setting['notifications'] else 'negative'
            notification_text = '🔔 On' if setting['notifications'] else '🔕 Off'
            html_parts.append(f'''
                        <tr>
                            <td>
                                <div class="user-info">
                                    <div class="user-avatar">{setting['username'][0].upper()}</div>
                                    <div>
                                        <strong>{setting['username']}</strong>
                                        <div style="font-size: 12px; color: #666;">ID: {setting['user_id']}</div>
                                    </div>
                                </div>
                            </td>
                            <td>
                                <span class="badge settings">{setting['theme']}</span>
                            </td>
                            <td>
                                <span class="badge {notification_class}">
                                    {notification_text}
                                </span>
                            </td>
                            <td class="timestamp">{setting['updated_at']}</td>
                            <td>#{setting['id']}</td>
                        </tr>
            ''')
        
        html_parts.append('''
                    </tbody>
                </table>
        ''')
    else:
        html_parts.append('''
                <div class="empty-state">
                    <div>⚙️</div>
                    <h3>No Settings Found</h3>
                    <p>There are no user settings in the database yet.</p>
                </div>
        ''')
    
    html_parts.append('''
            </div>
        </div>

        <script>
            function switchTab(tabName) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabName + '-tab').classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
            }

            function filterTable() {
                const input = document.getElementById('searchInput');
                const filter = input.value.toLowerCase();
                const tables = document.querySelectorAll('.data-table tbody');
                
                tables.forEach(table => {
                    const rows = table.getElementsByTagName('tr');
                    for (let i = 0; i < rows.length; i++) {
                        const text = rows[i].textContent.toLowerCase();
                        if (text.includes(filter)) {
                            rows[i].style.display = '';
                        } else {
                            rows[i].style.display = 'none';
                        }
                    }
                });
            }

            function deleteUser(userId, username, reviewCount) {
                const confirmation = confirm(`🚨 DANGER ZONE 🚨\\n\\nAre you sure you want to delete user "${username}"?\\n\\nThis will permanently:\\n• Delete the user account\\n• Remove ${reviewCount} reviews\\n• Erase all user settings\\n\\nThis action cannot be undone!`);
                
                if (confirmation) {
                    const password = new URLSearchParams(window.location.search).get('password');
                    window.location.href = `/admin/database?password=${password}&action=delete&user_id=${userId}`;
                }
            }

            function refreshData() {
                window.location.reload();
            }

            function exportData() {
                alert('📊 Export feature would download a CSV file with all data\\n(This is a placeholder - implement actual export if needed)');
            }

            // Auto-refresh every 30 seconds
            setInterval(() => {
                window.location.reload();
            }, 30000);

            // Add some interactive features
            document.addEventListener('DOMContentLoaded', function() {
                console.log('🎬 Admin Dashboard Loaded Successfully!');
                console.log('📊 Total Users: ''' + str(len(users_data)) + '''');
                console.log('📝 Total Reviews: ''' + str(len(reviews_data)) + '''');
                console.log('⚙️ Total Settings: ''' + str(len(settings_data)) + '''');
            });
        </script>
    </body>
    </html>
    ''')
    
    return ''.join(html_parts)

@app.route('/api/debug/users')
def debug_users():
    """Debug endpoint to see all users"""
    users = User.query.all()
    users_data = []
    for user in users:
        users_data.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'password_hash_length': len(user.password_hash) if user.password_hash else 0
        })
    return jsonify(users_data)

@app.route('/<path:path>')
def serve_static(path):
    return send_file('index.html')

# Initialize database
def init_db():
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created!")
            
            if User.query.count() == 0:
                test_user = User(username='testuser', email='test@example.com')
                test_user.set_password('testpassword123')
                db.session.add(test_user)
                db.session.commit()
                print("✅ Test user created: username='testuser', password='testpassword123'")
            
            users = User.query.all()
            print(f"📊 Total users in database: {len(users)}")
            for user in users:
                print(f"   - {user.username} ({user.email})")
                
        except Exception as e:
            print(f"❌ Database initialization error: {e}")

if __name__ == "__main__":
    init_db()
    print("🚀 Starting Flask server...")
    print("📱 Open http://127.0.0.1:5000 in your browser")
    print("🔐 Test credentials: username='testuser', password='testpassword123'")
    print("📊 Admin Dashboard: http://127.0.0.1:5000/admin/database?password=admin123")
    app.run(debug=True, host='127.0.0.1', port=5000)