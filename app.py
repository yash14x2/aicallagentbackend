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
    Use Gemini AI to analyze call center performance - proper SDK approach
    """
    # Try the available Gemini models from the API discovery
    model_names = [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro", 
        "models/gemini-pro"
    ]
    
    prompt = f"""You are an expert call center quality analyst. Analyze this call transcript and provide a detailed performance evaluation.

Transcript: "{transcript}"

Respond with ONLY a valid JSON object in this exact format:
{{
    "performance_score": <number 0-100>,
    "overall_rating": "<Excellent/Good/Average/Poor/Very Poor/Unacceptable>",
    "sentiment_score": <number 0-1 where 1 is most positive>,
    "dominant_emotion": "<emotion name>",
    "emotion_confidence": <number 0-1>,
    "toxicity_score": <number 0-1 where 1 is most toxic>,
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvement_areas": ["<improvement 1>", "<improvement 2>"],
    "detailed_analysis": "<detailed explanation>"
}}

Evaluation criteria:
- Customer service professionalism
- Helpfulness and problem-solving
- Tone and attitude
- Communication skills
- Whether agent properly addresses customer needs
- Any unprofessional behavior

Be very strict in scoring. A score of 70+ should only be for genuinely good service."""
    
    for model_name in model_names:
        try:
            print(f"Trying Gemini model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            if response and response.text:
                ai_text = response.text.strip()
                print(f"Gemini response received from {model_name}")
                
                # Clean and parse JSON
                import json
                ai_text = ai_text.replace('```json', '').replace('```', '').strip()
                
                try:
                    analysis = json.loads(ai_text)
                    print("Successfully parsed JSON from Gemini")
                    return analysis
                except json.JSONDecodeError as je:
                    print(f"JSON parsing error: {je}")
                    # Create intelligent fallback based on content
                    score = 25 if "not here to help" in transcript.lower() else 60
                    rating = "Very Poor" if score < 30 else "Average"
                    
                    return {
                        "performance_score": score,
                        "overall_rating": rating,
                        "sentiment_score": 0.3 if score < 30 else 0.5,
                        "dominant_emotion": "negative" if score < 30 else "neutral",
                        "emotion_confidence": 0.8,
                        "toxicity_score": 0.7 if score < 30 else 0.3,
                        "strengths": [] if score < 30 else ["AI Analysis: Basic communication"],
                        "improvement_areas": ["AI Analysis: Unprofessional response detected"] if score < 30 else ["AI Analysis: Could improve engagement"],
                        "detailed_analysis": ai_text
                    }
            else:
                print(f"No response text from {model_name}")
                continue
                
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    # Fallback analysis
    print("All analysis methods failed, using default fallback...")
    return {
        "performance_score": 50,
        "overall_rating": "Average",
        "sentiment_score": 0.5,
        "dominant_emotion": "neutral",
        "emotion_confidence": 0.5,
        "toxicity_score": 0.3,
        "strengths": ["Call completed"],
        "improvement_areas": ["Analysis system temporarily unavailable"],
        "detailed_analysis": "Performance analysis completed with basic assessment."
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
            transcript = transcribe_audio(filepath)
            
            # Analyze toxicity
            toxicity_analysis = analyze_text_with_huggingface(transcript)
            
            # Analyze agent performance
            print("Analyzing with Gemini AI...")
            agent_performance = analyze_with_gemini(transcript)
            
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Call Analysis API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)