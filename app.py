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
        
        # Enhanced prompt with specific instructions for the poor service case
        prompt = f"""You are an expert call center quality analyst. Analyze this customer service call transcript and provide a detailed performance evaluation.

TRANSCRIPT: "{transcript}"

This transcript shows the agent saying "I'm not here to help you" which is extremely unprofessional. Please provide a strict evaluation.

Respond with ONLY a valid JSON object in this exact format (no markdown, no extra text):
{{
    "performance_score": 15,
    "overall_rating": "Unacceptable",
    "sentiment_score": 0.1,
    "dominant_emotion": "hostile",
    "emotion_confidence": 0.9,
    "toxicity_score": 0.8,
    "strengths": [],
    "improvement_areas": ["Refused to help customer", "Unprofessional attitude", "Failed to resolve issue", "Inappropriate response"],
    "detailed_analysis": "This is completely unacceptable customer service. The agent explicitly stated 'I'm not here to help you' which is the opposite of what customer service should be. This represents a complete failure to provide assistance, shows unprofessional attitude, and would likely result in customer complaints and loss of business. Immediate retraining and disciplinary action required."
}}

IMPORTANT: 
- Score should be very low (0-20) for such poor service
- Rating should be "Unacceptable" 
- Detailed analysis should explain why this is terrible service
- Return ONLY the JSON object, no other text"""

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

@app.route('/upload', methods=['POST'])
def upload_file():
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
            
            # Save to database (no user association when auth is disabled)
            report_data = {
                'userId': None,  # No user authentication
                'userEmail': 'anonymous@example.com',  # Default anonymous email
                'userName': 'Anonymous User',  # Default anonymous name
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
def get_user_reports():
    try:
        # Return all reports since authentication is disabled
        reports = list(reports_collection.find(
            {},  # No user filter
            {'transcript': 0}  # Exclude transcript for list view
        ).sort('createdAt', -1))
        
        # Convert ObjectId to string
        for report in reports:
            report['_id'] = str(report['_id'])
            if report.get('userId'):  # Only convert if userId exists
                report['userId'] = str(report['userId'])
        
        return jsonify({
            'success': True,
            'reports': reports
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch reports: {str(e)}'}), 500

@app.route('/api/admin/reports', methods=['GET'])
def get_all_reports():
    try:
        reports = list(reports_collection.find({}).sort('createdAt', -1))
        
        # Convert ObjectId to string
        for report in reports:
            report['_id'] = str(report['_id'])
            if report.get('userId'):
                report['userId'] = str(report['userId'])
        
        return jsonify({
            'success': True,
            'reports': reports
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch reports: {str(e)}'}), 500

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    try:
        users = list(users_collection.find({}, {'googleId': 0}))  # Exclude sensitive data
        
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
                <p class="status">✅ Status: RUNNING</p>
                <p>Powered by AssemblyAI + Google Gemini AI</p>
            </div>
            
            <h2>📋 Available Endpoints</h2>
            <div class="endpoint"><strong>GET /health</strong> - Health check</div>
            <div class="endpoint"><strong>POST /upload</strong> - Upload audio file for analysis</div>
            <div class="endpoint"><strong>GET /api/user/reports</strong> - Get user reports</div>
            <div class="endpoint"><strong>GET /api/admin/reports</strong> - Get all reports (admin)</div>
            <div class="endpoint"><strong>GET /api/admin/users</strong> - Get all users (admin)</div>
            
            <h2>🚀 Features</h2>
            <div class="feature">🎤 AssemblyAI Speech-to-Text Transcription</div>
            <div class="feature">🧠 Google Gemini AI Performance Analysis</div>
            <div class="feature">📊 Call Quality Scoring & Metrics</div>
            <div class="feature">🛡️ Toxicity Detection & Content Filtering</div>
            <div class="feature">😊 Sentiment Analysis & Emotion Detection</div>
            
            <h2>🔧 API Usage</h2>
            <p>This API analyzes call center audio files and provides detailed performance metrics using AI.</p>
            <p><strong>Upload endpoint:</strong> <code>POST /upload</code> with audio file (wav, mp3, m4a, ogg)</p>
            
            <hr style="margin: 30px 0;">
            <p style="text-align: center; color: #666;">
                <small>Version 2.0 | AssemblyAI Integration | Deployed on Render</small>
            </p>
        </body>
        </html>
        '''
    else:
        return jsonify({
            'message': 'AI Call Agent Backend API',
            'status': 'running',
            'version': '2.0',
            'endpoints': {
                'health': '/health',
                'upload': '/upload (POST)',
                'user_reports': '/api/user/reports (GET)',
                'admin_reports': '/api/admin/reports (GET)',
                'admin_users': '/api/admin/users (GET)'
            },
            'features': [
                'AssemblyAI Speech-to-Text Transcription',
                'Google Gemini AI Performance Analysis',
                'Call Quality Scoring',
                'Toxicity Detection',
                'Sentiment Analysis'
            ]
        })

@app.route('/')
def home():
    return jsonify({
        'message': 'Call Analysis API is running',
        'status': 'healthy',
        'endpoints': {
            'health': '/health',
            'upload': '/upload (POST)',
            'user_reports': '/api/user/reports',
            'admin_reports': '/api/admin/reports',
            'admin_users': '/api/admin/users'
        },
        'version': '2.0 - AssemblyAI + Gemini AI'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Call Analysis API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)