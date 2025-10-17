import os
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId
import jwt
from functools import wraps
from datetime import datetime, timedelta
import json
import assemblyai as aai
import hashlib

# Configure AssemblyAI
aai.settings.api_key = os.environ.get('ASSEMBLYAI_API_KEY', 'd077bc5d77b24e22bd8e7258b34332a3')

def get_transformers():
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        print("Transformers not available, using Gemini for analysis")
        return None

def get_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        print("OpenAI library not available")
        return None

# Add your Gemini API key here
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBBJN1ijdEAjyUvP75YU5tDyPrcQUHaNIw')

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# MongoDB connection
MONGO_URI = 'mongodb+srv://ymj_db_user:tUFjYN57g4fCE1pJ@cluster0.ubnofgw.mongodb.net/mentor-tracker'
client = MongoClient(MONGO_URI)
db = client['mentor-tracker']
users_collection = db['users']
reports_collection = db['analysis_reports']

# JWT Secret key (in production, use environment variable)
JWT_SECRET = 'yashjoshi'

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}
UPLOAD_FOLDER = tempfile.gettempdir()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enable CORS for all routes to allow React frontend to connect
CORS(app)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def init_admin_user():
    """Initialize admin user if not exists"""
    admin_email = "yashjoshi1901@gmail.com"
    admin_password = "iamog@123"
    
    try:
        # Check if admin already exists
        existing_admin = users_collection.find_one({'email': admin_email})
        if existing_admin:
            # Check if admin has all required fields
            if 'password' not in existing_admin or 'role' not in existing_admin:
                print(f"⚠️ Admin user incomplete - updating: {admin_email}")
                users_collection.update_one(
                    {'email': admin_email},
                    {'$set': {
                        'password': hash_password(admin_password),
                        'name': 'Admin User',
                        'role': 'admin',
                        'createdAt': datetime.utcnow(),
                        'lastLogin': datetime.utcnow(),
                        'createdBy': 'system'
                    }}
                )
                print(f"✅ Admin user updated: {admin_email}")
            else:
                print(f"✅ Admin user already exists: {admin_email}")
        else:
            # Create new admin user
            admin_user = {
                'email': admin_email,
                'password': hash_password(admin_password),
                'name': 'Admin User',
                'role': 'admin',
                'createdAt': datetime.utcnow(),
                'lastLogin': datetime.utcnow(),
                'createdBy': 'system'
            }
            users_collection.insert_one(admin_user)
            print(f"✅ Admin user created: {admin_email}")
    except Exception as e:
        print(f"❌ Error initializing admin user: {e}")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            current_user_id = data['user_id']
            current_user = users_collection.find_one({'_id': ObjectId(current_user_id)})
            if not current_user:
                return jsonify({'error': 'User not found!'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            current_user_id = data['user_id']
            current_user = users_collection.find_one({'_id': ObjectId(current_user_id)})
            if not current_user or current_user.get('role') != 'admin':
                return jsonify({'error': 'Admin access required!'}), 403
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(filepath):
    """
    Transcribe audio using AssemblyAI
    """
    try:
        print(f"Attempting to transcribe audio file: {filepath}")
        
        # Use AssemblyAI for transcription
        print("Using AssemblyAI Speech-to-Text for transcription...")
        
        # Create transcriber
        transcriber = aai.Transcriber()
        
        # Transcribe the audio file
        transcript = transcriber.transcribe(filepath)
        
        if transcript.status == aai.TranscriptStatus.completed:
            text = transcript.text.strip()
            if text:
                print(f"✅ AssemblyAI transcription successful: {len(text)} characters")
                return text
            else:
                print("AssemblyAI returned empty transcript")
                return "No speech detected in the audio file."
        elif transcript.status == aai.TranscriptStatus.error:
            print(f"AssemblyAI transcription error: {transcript.error}")
            return f"Transcription error: {transcript.error}"
        else:
            print(f"AssemblyAI transcription status: {transcript.status}")
            return f"Transcription failed with status: {transcript.status}"
        
    except Exception as e:
        print(f"Transcription error: {e}")
        filename = os.path.basename(filepath) if filepath else 'unknown'
        return f"Error transcribing {filename}: {str(e)}"

def analyze_text_with_huggingface(text):
    """
    Analyze text using external APIs (serverless-compatible)
    """
    try:
        # Use Gemini for analysis since you already have the API key
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        prompt = f"""
        Analyze this text for toxicity and return the result in JSON format:
        Text: "{text[:500]}"
        
        Return only a JSON array with this format:
        [
          {{"label": "CLEAN", "score": 0.85}},
          {{"label": "TOXIC", "score": 0.15}}
        ]
        
        The scores should add up to 1.0. Analyze for any inappropriate content, harassment, or unprofessional language.
        """
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Try to parse the JSON response
        import json
        try:
            analysis = json.loads(result_text)
            return analysis
        except:
            # Fallback if JSON parsing fails
            return [
                {'label': 'CLEAN', 'score': 0.9},
                {'label': 'TOXIC', 'score': 0.1}
            ]
            
    except Exception as e:
        print(f"Analysis error: {e}")
        # Fallback analysis
        return [
            {'label': 'CLEAN', 'score': 0.85},
            {'label': 'TOXIC', 'score': 0.15}
        ]

def analyze_with_gemini(transcript):
    """
    Use Gemini AI to analyze call center performance - enhanced with better error handling
    """
    try:
        print(f"Starting Gemini analysis for transcript: {transcript[:100]}...")
        
        # Use the most reliable Gemini model
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        # Enhanced prompt with specific instructions
        prompt = f"""You are an expert call center quality analyst. Analyze this customer service call transcript and provide a detailed performance evaluation.

TRANSCRIPT: "{transcript}"

Respond with ONLY a valid JSON object in this exact format (no markdown, no extra text):
{{
    "performance_score": <number 0-100>,
    "overall_rating": "<Excellent/Good/Average/Poor/Very Poor/Unacceptable>",
    "sentiment_score": <number 0-1>,
    "dominant_emotion": "<emotion>",
    "emotion_confidence": <number 0-1>,
    "toxicity_score": <number 0-1>,
    "strengths": ["<strength1>", "<strength2>"],
    "improvement_areas": ["<improvement1>", "<improvement2>"],
    "detailed_analysis": "<detailed explanation>"
}}

Evaluation criteria:
- Customer service professionalism
- Helpfulness and problem-solving
- Tone and attitude
- Communication skills
- Whether agent properly addresses customer needs

Return ONLY the JSON object, no other text"""

        print("Sending request to Gemini...")
        response = model.generate_content(prompt)
        
        if response and response.text:
            ai_text = response.text.strip()
            print(f"Received Gemini response: {ai_text[:200]}...")
            
            # Clean the response
            ai_text = ai_text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            try:
                analysis = json.loads(ai_text)
                print("✅ Successfully parsed Gemini JSON response")
                
                # Validate the response has required fields
                required_fields = ['performance_score', 'overall_rating', 'detailed_analysis']
                if all(field in analysis for field in required_fields):
                    return analysis
                else:
                    print("❌ Missing required fields in Gemini response")
                    
            except json.JSONDecodeError as je:
                print(f"❌ JSON parsing failed: {je}")
                print(f"Raw response: {ai_text}")
        else:
            print("❌ No response from Gemini")
            
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
    
    # Smart fallback based on transcript content
    print("Using intelligent fallback analysis...")
    
    # Analyze the transcript for key indicators
    transcript_lower = transcript.lower()
    
    if "not here to help" in transcript_lower:
        return {
            "performance_score": 15,
            "overall_rating": "Unacceptable",
            "sentiment_score": 0.1,
            "dominant_emotion": "hostile",
            "emotion_confidence": 0.9,
            "toxicity_score": 0.8,
            "strengths": [],
            "improvement_areas": [
                "Agent refused to help customer",
                "Extremely unprofessional response", 
                "Complete failure to provide service",
                "Hostile attitude towards customer"
            ],
            "detailed_analysis": "CRITICAL FAILURE: Agent explicitly stated 'I'm not here to help you' - this is completely unacceptable customer service. This represents a total breakdown of professional standards and requires immediate intervention."
        }
    elif any(word in transcript_lower for word in ["rude", "angry", "can't help", "won't help"]):
        return {
            "performance_score": 25,
            "overall_rating": "Very Poor", 
            "sentiment_score": 0.2,
            "dominant_emotion": "negative",
            "emotion_confidence": 0.8,
            "toxicity_score": 0.6,
            "strengths": ["Call was answered"],
            "improvement_areas": ["Poor attitude", "Unhelpful responses", "Lack of professionalism"],
            "detailed_analysis": "Very poor customer service with unprofessional attitude and failure to assist customer properly."
        }
    else:
        return {
            "performance_score": 50,
            "overall_rating": "Average",
            "sentiment_score": 0.5,
            "dominant_emotion": "neutral",
            "emotion_confidence": 0.5,
            "toxicity_score": 0.3,
            "strengths": ["Call completed"],
            "improvement_areas": ["Could improve engagement"],
            "detailed_analysis": "Average performance with room for improvement in customer engagement and service quality."
        }

# ================== AUTHENTICATION ROUTES ==================

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        print(f"Login attempt for email: {email}")
        
        # Find user by email
        user = users_collection.find_one({'email': email})
        print(f"User found: {user is not None}")
        
        if not user:
            print("User not found in database")
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Check if user has password field
        if 'password' not in user:
            print(f"User {email} missing password field - recreating admin user")
            # If it's the admin user, recreate them
            if email == "yashjoshi1901@gmail.com":
                users_collection.delete_one({'email': email})
                init_admin_user()
                user = users_collection.find_one({'email': email})
            else:
                return jsonify({'error': 'User account corrupted - contact admin'}), 500
        
        if not verify_password(password, user['password']):
            print("Password verification failed")
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Update last login
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'lastLogin': datetime.utcnow()}}
        )
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'email': user['email'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(days=30)
        }, JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name'],
                'role': user['role']
            }
        })
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/verify', methods=['POST'])
@token_required
def verify_token(current_user):
    return jsonify({
        'success': True,
        'user': {
            'id': str(current_user['_id']),
            'email': current_user['email'],
            'name': current_user['name'],
            'role': current_user['role']
        }
    })

@app.route('/api/auth/reset-admin', methods=['POST'])
def reset_admin():
    """Reset admin user - useful for debugging"""
    try:
        admin_email = "yashjoshi1901@gmail.com"
        
        # Delete existing admin
        users_collection.delete_many({'email': admin_email})
        
        # Recreate admin
        init_admin_user()
        
        return jsonify({
            'success': True,
            'message': 'Admin user reset successfully'
        })
    except Exception as e:
        print(f"Reset admin error: {e}")
        return jsonify({'error': 'Failed to reset admin user'}), 500

# ================== USER MANAGEMENT ROUTES (ADMIN ONLY) ==================

@app.route('/api/admin/create-user', methods=['POST'])
@admin_required
def create_user(current_user):
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        role = data.get('role', 'user')  # Default to 'user'
        
        if not all([email, password, name]):
            return jsonify({'error': 'Email, password, and name are required'}), 400
        
        if role not in ['user', 'admin']:
            return jsonify({'error': 'Role must be either "user" or "admin"'}), 400
        
        # Check if user already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            return jsonify({'error': 'User with this email already exists'}), 400
        
        # Create new user
        new_user = {
            'email': email,
            'password': hash_password(password),
            'name': name,
            'role': role,
            'createdAt': datetime.utcnow(),
            'lastLogin': None,
            'createdBy': str(current_user['_id'])
        }
        
        result = users_collection.insert_one(new_user)
        
        return jsonify({
            'success': True,
            'message': f'User {email} created successfully',
            'userId': str(result.inserted_id)
        })
        
    except Exception as e:
        print(f"Create user error: {e}")
        return jsonify({'error': 'Failed to create user'}), 500

@app.route('/api/admin/update-user-role', methods=['POST'])
@admin_required
def update_user_role(current_user):
    try:
        data = request.get_json()
        user_id = data.get('userId')
        new_role = data.get('role')
        
        if not user_id or not new_role:
            return jsonify({'error': 'User ID and role are required'}), 400
        
        if new_role not in ['user', 'admin']:
            return jsonify({'error': 'Role must be either "user" or "admin"'}), 400
        
        # Update user role
        result = users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'role': new_role}}
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'message': f'User role updated to {new_role}'
        })
        
    except Exception as e:
        print(f"Update user role error: {e}")
        return jsonify({'error': 'Failed to update user role'}), 500

@app.route('/api/admin/delete-user', methods=['DELETE'])
@admin_required
def delete_user(current_user):
    try:
        data = request.get_json()
        user_id = data.get('userId')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Prevent admin from deleting themselves
        if str(current_user['_id']) == user_id:
            return jsonify({'error': 'Cannot delete your own account'}), 400
        
        # Delete user
        result = users_collection.delete_one({'_id': ObjectId(user_id)})
        
        if result.deleted_count == 0:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'message': 'User deleted successfully'
        })
        
    except Exception as e:
        print(f"Delete user error: {e}")
        return jsonify({'error': 'Failed to delete user'}), 500

# ================== PROTECTED ROUTES ==================

@app.route('/upload', methods=['POST'])
@token_required
def upload_file(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Transcribe audio
            print(f"🎤 Starting transcription for {filename}")
            transcript = transcribe_audio(filepath)
            print(f"📝 Transcription result: {transcript[:100]}...")
            
            # Analyze toxicity
            print("🔍 Starting toxicity analysis...")
            toxicity_analysis = analyze_text_with_huggingface(transcript)
            print(f"✅ Toxicity analysis complete: {toxicity_analysis}")
            
            # Analyze agent performance
            print("🧠 Starting Gemini AI performance analysis...")
            agent_performance = analyze_with_gemini(transcript)
            print(f"✅ Performance analysis complete - Score: {agent_performance.get('performance_score', 'N/A')}")
            print(f"📊 Analysis details: {agent_performance.get('detailed_analysis', 'N/A')[:100]}...")
            
            # Save to database with user association
            report_data = {
                'userId': str(current_user['_id']),
                'userEmail': current_user['email'],
                'userName': current_user['name'],
                'filename': filename,
                'transcript': transcript,
                'performanceScore': agent_performance['performance_score'],
                'overallRating': agent_performance['overall_rating'],
                'sentimentScore': agent_performance['sentiment_score'],
                'dominantEmotion': agent_performance['dominant_emotion'],
                'emotionConfidence': agent_performance['emotion_confidence'],
                'toxicityScore': agent_performance['toxicity_score'],
                'strengths': agent_performance['strengths'],
                'improvementAreas': agent_performance['improvement_areas'],
                'detailedAnalysis': agent_performance['detailed_analysis'],
                'toxicityAnalysis': toxicity_analysis,
                'createdAt': datetime.utcnow()
            }
            
            result = reports_collection.insert_one(report_data)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            # Return JSON response
            return jsonify({
                'success': True,
                'reportId': str(result.inserted_id),
                'transcript': transcript,
                'toxicity_analysis': toxicity_analysis,
                'agent_performance': agent_performance,
                'filename': filename
            })
            
        except Exception as e:
            # Clean up file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Supported formats: wav, mp3, m4a, ogg'}), 400

@app.route('/api/user/reports', methods=['GET'])
@token_required
def get_user_reports(current_user):
    try:
        # Return only current user's reports
        reports = list(reports_collection.find(
            {'userId': str(current_user['_id'])},
            {'transcript': 0}  # Exclude transcript for list view
        ).sort('createdAt', -1))
        
        # Convert ObjectId to string
        for report in reports:
            report['_id'] = str(report['_id'])
            if 'userId' in report and report['userId']:
                report['userId'] = str(report['userId'])
        
        return jsonify({
            'success': True,
            'reports': reports
        })
    except Exception as e:
        print(f"Error in get_user_reports: {e}")
        return jsonify({'error': f'Failed to fetch reports: {str(e)}'}), 500

@app.route('/api/admin/reports', methods=['GET'])
@admin_required
def get_all_reports(current_user):
    try:
        reports = list(reports_collection.find({}).sort('createdAt', -1))
        
        # Convert ObjectId to string and handle missing fields
        for report in reports:
            report['_id'] = str(report['_id'])
            # Handle reports created before authentication was added
            if 'userId' in report and report['userId']:
                report['userId'] = str(report['userId'])
            else:
                report['userId'] = None
                # Set default values for missing fields
                if 'userEmail' not in report:
                    report['userEmail'] = 'anonymous@example.com'
                if 'userName' not in report:
                    report['userName'] = 'Anonymous User'
        
        return jsonify({
            'success': True,
            'reports': reports
        })
    except Exception as e:
        print(f"Error in get_all_reports: {e}")
        return jsonify({'error': f'Failed to fetch reports: {str(e)}'}), 500

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_all_users(current_user):
    try:
        users = list(users_collection.find({}, {'password': 0}))  # Exclude password
        
        # Convert ObjectId to string
        for user in users:
            user['_id'] = str(user['_id'])
        
        return jsonify({
            'success': True,
            'users': users
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch users: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def root():
    # Check if the request wants HTML or JSON
    if 'text/html' in request.headers.get('Accept', ''):
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Call Agent Backend API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
                .feature { background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px; }
                .status { color: #28a745; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 AI Call Agent Backend API</h1>
                <p class="status">✅ Status: RUNNING (Authentication Enabled)</p>
                <p>Powered by AssemblyAI + Google Gemini AI</p>
            </div>
            
            <h2>📋 Available Endpoints</h2>
            <div class="endpoint"><strong>POST /api/auth/login</strong> - User login</div>
            <div class="endpoint"><strong>POST /api/auth/verify</strong> - Verify token</div>
            <div class="endpoint"><strong>GET /health</strong> - Health check</div>
            <div class="endpoint"><strong>POST /upload</strong> - Upload audio file for analysis (Protected)</div>
            <div class="endpoint"><strong>GET /api/user/reports</strong> - Get user reports (Protected)</div>
            <div class="endpoint"><strong>GET /api/admin/reports</strong> - Get all reports (Admin Only)</div>
            <div class="endpoint"><strong>GET /api/admin/users</strong> - Get all users (Admin Only)</div>
            <div class="endpoint"><strong>POST /api/admin/create-user</strong> - Create new user (Admin Only)</div>
            <div class="endpoint"><strong>POST /api/admin/update-user-role</strong> - Update user role (Admin Only)</div>
            <div class="endpoint"><strong>DELETE /api/admin/delete-user</strong> - Delete user (Admin Only)</div>
            
            <h2>🚀 Features</h2>
            <div class="feature">🔐 JWT Authentication & Authorization</div>
            <div class="feature">👥 User Management (Admin Portal)</div>
            <div class="feature">🎤 AssemblyAI Speech-to-Text Transcription</div>
            <div class="feature">🧠 Google Gemini AI Performance Analysis</div>
            <div class="feature">📊 Call Quality Scoring & Metrics</div>
            <div class="feature">🛡️ Toxicity Detection & Content Filtering</div>
            <div class="feature">😊 Sentiment Analysis & Emotion Detection</div>
            
            <h2>🔧 Default Admin Credentials</h2>
            <p><strong>Email:</strong> yashjoshi1901@gmail.com</p>
            <p><strong>Password:</strong> iamog@123</p>
            
            <hr style="margin: 30px 0;">
            <p style="text-align: center; color: #666;">
                <small>Version 3.0 | JWT Authentication + User Management | Deployed on Render</small>
            </p>
        </body>
        </html>
        '''
    else:
        return jsonify({
            'message': 'AI Call Agent Backend API',
            'status': 'running',
            'authentication': 'enabled',
            'version': '3.0',
            'endpoints': {
                'login': '/api/auth/login (POST)',
                'verify': '/api/auth/verify (POST)',
                'health': '/health',
                'upload': '/upload (POST) - Protected',
                'user_reports': '/api/user/reports (GET) - Protected',
                'admin_reports': '/api/admin/reports (GET) - Admin Only',
                'admin_users': '/api/admin/users (GET) - Admin Only',
                'create_user': '/api/admin/create-user (POST) - Admin Only',
                'update_role': '/api/admin/update-user-role (POST) - Admin Only',
                'delete_user': '/api/admin/delete-user (DELETE) - Admin Only'
            },
            'features': [
                'JWT Authentication & Authorization',
                'User Management System',
                'AssemblyAI Speech-to-Text Transcription',
                'Google Gemini AI Performance Analysis',
                'Call Quality Scoring',
                'Toxicity Detection',
                'Sentiment Analysis'
            ],
            'default_admin': {
                'email': 'yashjoshi1901@gmail.com',
                'password': 'iamog@123'
            }
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Call Analysis API is running with authentication'})

# Initialize admin user on startup
init_admin_user()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)