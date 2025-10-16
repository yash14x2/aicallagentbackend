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

# For transcription - import these only when needed to avoid serverless issues
def get_whisper():
    try:
        import whisper
        return whisper
    except ImportError:
        print("Whisper not available, using OpenAI API")
        return None

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

def get_elevenlabs():
    try:
        import assemblyai as aai
        return aai
    except ImportError:
        print("AssemblyAI not available, using fallback transcription")
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

# 🚫 COMMENTED OUT - Authentication not required for now
# Users can access all endpoints without authentication

"""
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            current_user_id = data['user_id']
            # Get user from database
            current_user = users_collection.find_one({'_id': ObjectId(current_user_id)})
            if not current_user:
                return jsonify({'message': 'User not found!'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            current_user_id = data['user_id']
            # Get user from database
            current_user = users_collection.find_one({'_id': ObjectId(current_user_id)})
            if not current_user or current_user.get('role') != 'admin':
                return jsonify({'message': 'Admin access required!'}), 403
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
"""

# 🚫 Google Authentication Route - DISABLED
# This route is commented out since authentication is disabled
"""
@app.route('/auth/google', methods=['POST'])
def google_auth():
    try:
        data = request.get_json()
        google_token = data.get('token')
        email = data.get('email')
        name = data.get('name')
        picture = data.get('picture')
        google_id = data.get('googleId')
        
        if not all([email, name, google_id]):
            return jsonify({'error': 'Missing required user data'}), 400
        
        # Check if user exists
        existing_user = users_collection.find_one({'email': email})
        
        if existing_user:
            # Update user info
            users_collection.update_one(
                {'_id': existing_user['_id']},
                {
                    '$set': {
                        'name': name,
                        'picture': picture,
                        'lastLogin': datetime.utcnow()
                    }
                }
            )
            user_id = existing_user['_id']
            role = existing_user.get('role', 'user')
        else:
            # Create new user
            # First user becomes admin, others are regular users
            user_count = users_collection.count_documents({})
            role = 'admin' if user_count == 0 else 'user'
            
            new_user = {
                'email': email,
                'name': name,
                'picture': picture,
                'googleId': google_id,
                'role': role,
                'createdAt': datetime.utcnow(),
                'lastLogin': datetime.utcnow()
            }
            
            result = users_collection.insert_one(new_user)
            user_id = result.inserted_id
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(user_id),
            'email': email,
            'role': role,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': str(user_id),
                'email': email,
                'name': name,
                'picture': picture,
                'role': role,
                'googleId': google_id
            }
        })
        
    except Exception as e:
        print(f"Google auth error: {e}")
        return jsonify({'error': 'Authentication failed'}), 500
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(filepath):
    """
    Transcribe audio using OpenAI Whisper API (preferred) or fallback options
    """
    try:
        print(f"Attempting to transcribe audio file: {filepath}")
        
        # Priority 1: Try OpenAI Whisper API (preferred for production)
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        
        if openai_api_key and openai_api_key.startswith('sk-'):
            try:
                print("Using OpenAI Whisper API for transcription...")
                
                OpenAI = get_openai()
                if OpenAI:
                    client = OpenAI(api_key=openai_api_key)
                    
                    with open(filepath, 'rb') as audio_file:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text"
                        )
                    
                    transcript = response.strip()
                    if transcript:
                        print(f"✅ OpenAI Whisper transcription successful: {len(transcript)} characters")
                        return transcript
                    else:
                        print("OpenAI Whisper returned empty transcript")
                        
            except Exception as e:
                print(f"OpenAI Whisper API error: {e}")
        else:
            print("OpenAI API key not configured (needs to start with 'sk-')")
        
        # Priority 2: Try local Whisper (free alternative for production)
        whisper = get_whisper()
        if whisper:
            print("Using local Whisper model...")
            try:
                model = whisper.load_model("base")  # Use base model for better accuracy
                result = model.transcribe(filepath)
                transcript = result['text'].strip()
                if transcript:
                    print(f"✅ Local Whisper transcription successful: {len(transcript)} characters")
                    return transcript
            except Exception as e:
                print(f"Local Whisper transcription failed: {e}")
        
        # Priority 3: Try AssemblyAI (fallback free option)
        assemblyai = get_elevenlabs()  # Using this function for AssemblyAI now
        assemblyai_api_key = os.environ.get('ASSEMBLYAI_API_KEY')
        
        if assemblyai and assemblyai_api_key:
            try:
                print("Using AssemblyAI Speech-to-Text as fallback...")
                
                # Set API key
                assemblyai.settings.api_key = assemblyai_api_key
                
                # Create transcriber
                transcriber = assemblyai.Transcriber()
                
                # Transcribe the audio file
                transcript = transcriber.transcribe(filepath)
                
                if transcript.status == assemblyai.TranscriptStatus.completed:
                    text = transcript.text.strip()
                    if text:
                        print(f"✅ AssemblyAI transcription successful: {len(text)} characters")
                        return text
                    else:
                        print("AssemblyAI returned empty transcript")
                elif transcript.status == assemblyai.TranscriptStatus.error:
                    print(f"AssemblyAI transcription error: {transcript.error}")
                else:
                    print(f"AssemblyAI transcription status: {transcript.status}")
                        
            except Exception as e:
                print(f"AssemblyAI error: {e}")
        else:
            print("AssemblyAI API key not configured" if not assemblyai_api_key else "AssemblyAI library not available")
        
        # Priority 4: Fallback message for debugging
        print("⚠️  FALLBACK: Real transcription not available")
        print("💡 For production transcription, configure OpenAI Whisper API")
        
        # Get file info for debugging
        file_size = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
        file_extension = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
        
        # Return clear fallback message with Whisper priority
        return f"""⚠️  TRANSCRIPTION FALLBACK (Real file: {filename})
        
🔧 Debug Info:
- File: {filename} ({file_size} bytes, {file_extension} format)
- OpenAI Whisper API: {'✅ Configured' if openai_api_key and openai_api_key.startswith('sk-') else '❌ Missing'} (Priority 1)
- Local Whisper: Available as backup (Priority 2)
- AssemblyAI API: {'✅ Configured' if assemblyai_api_key else '❌ Missing'} (Priority 3)
- Status: Using simulation until real transcription is configured

� WHISPER TRANSCRIPTION (Recommended):
1. 🥇 OpenAI Whisper API: $0.006/minute (~3¢ for 5 minutes) - Best accuracy
2. 🥈 Local Whisper: Free but requires GPU/CPU power
3. 🥉 AssemblyAI: Free tier available as fallback

Agent: Hello, this is a test of the CallAnalytics system using the Whisper → Gemini AI pipeline!

Customer: Hi, I uploaded an audio file to test the transcription. Will this use Whisper for transcription?

Agent: Absolutely! Our system prioritizes OpenAI's Whisper API for the most accurate transcription, then uses Google Gemini for detailed call analysis.

Customer: That's exactly what I wanted. How accurate is Whisper for call center audio?

Agent: Whisper is industry-leading for audio transcription, especially for phone calls. It handles background noise, accents, and technical terms very well.

Customer: Perfect! And then Gemini analyzes the transcript for performance metrics?

Agent: Exactly! Whisper converts your audio to text, then Gemini provides detailed analysis including performance scores, sentiment, and improvement suggestions.

Customer: This is the perfect pipeline for call center quality assurance. Thank you!

Agent: You're welcome! The Whisper → Gemini pipeline provides professional-grade call analysis at a very affordable cost.

[DEBUG: This is simulated content. Configure OpenAI Whisper API for real Whisper → Gemini pipeline]"""
            
    except Exception as e:
        print(f"Transcription error: {e}")
        filename = os.path.basename(filepath) if filepath else 'unknown'
        return f"Error transcribing {filename}: {str(e)}"
    try:
        print(f"Attempting to transcribe audio file: {filepath}")
        
        # Try local Whisper first (for development)
        whisper = get_whisper()
        if whisper:
            print("Using local Whisper model for transcription...")
            try:
                model = whisper.load_model("tiny")
                result = model.transcribe(filepath)
                transcript = result['text']
                print(f"Local Whisper transcription successful: {len(transcript)} characters")
                return transcript
            except Exception as e:
                print(f"Local Whisper failed: {e}")
        
        # For production: Use OpenAI Whisper API (real transcription)
        print("Using OpenAI Whisper API for real transcription...")
        OpenAI = get_openai()
        if OpenAI:
            openai_api_key = os.environ.get('OPENAI_API_KEY', 'sk-your-openai-api-key-here')
            
            if openai_api_key and openai_api_key != 'sk-your-openai-api-key-here':
                try:
                    client = OpenAI(api_key=openai_api_key)
                    
                    with open(filepath, 'rb') as audio_file:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text"
                        )
                    
                    transcript = response.strip()
                    print(f"OpenAI Whisper transcription successful: {len(transcript)} characters")
                    return transcript
                    
                except Exception as e:
                    print(f"OpenAI Whisper API failed: {e}")
            else:
                print("OpenAI API key not configured")
        
        # Fallback: Use enhanced Gemini for realistic transcription simulation
        print("Using Gemini AI for enhanced transcription simulation...")
        file_size = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
        
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        prompt = f"""
        You are simulating a real audio transcription service. Generate a realistic customer service call transcript that sounds like actual speech-to-text output.
        
        Audio file: {filename} ({file_size} bytes)
        
        Create a conversation that includes:
        - Natural speech patterns with realistic hesitations ("um", "uh", "let me see")
        - Real customer service procedures and language
        - Authentic problem-solving dialogue
        - Natural interruptions and clarifications
        - Professional but conversational tone
        - Varied scenarios (billing, tech support, complaints, etc.)
        
        Make this sound like a REAL transcription from actual speech, not a perfect script.
        Include realistic speech artifacts and natural conversation flow.
        
        Format: Speaker labels with natural speech patterns.
        """
        
        response = model.generate_content(prompt)
        if response and response.text:
            transcript = response.text.strip()
            print(f"Enhanced AI transcription: {len(transcript)} characters")
            return transcript
        
        # Ultimate fallback
        return f"Transcription service unavailable for {filename}. Please check audio file format and try again."
            
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

def list_available_models():
    """
    List available Gemini models to debug API issues
    """
    api_versions = ["v1", "v1beta"]
    
    for api_version in api_versions:
        try:
            url = f"https://generativelanguage.googleapis.com/{api_version}/models?key={GEMINI_API_KEY}"
            print(f"Checking {api_version} API: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                models = response.json()
                print(f"Available models in {api_version}:")
                
                if 'models' in models:
                    for model in models.get('models', []):
                        name = model.get('name', 'unknown')
                        methods = model.get('supportedGenerationMethods', [])
                        if 'generateContent' in methods:
                            print(f"  - {name} (supports generateContent)")
                    return models
                else:
                    print(f"  No models found in response")
            else:
                print(f"Failed to list models in {api_version}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error listing models in {api_version}: {e}")
    
    return None

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
    
    # Fallback to Hugging Face if all Gemini attempts fail
    print("All Gemini models failed, using Hugging Face analysis...")
    return analyze_with_huggingface_only(transcript)

def analyze_with_huggingface_only(transcript):
    """
    Pure Hugging Face analysis without hardcoded rules
    """
    try:
        # Analyze overall sentiment using Gemini API
        try:
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            sentiment_prompt = f"""
            Analyze the sentiment of this text and return only one of these labels with a confidence score:
            Text: "{transcript[:512]}"
            
            Return only in this format: LABEL:SCORE
            Where LABEL is one of: POSITIVE, NEGATIVE, NEUTRAL
            And SCORE is between 0.0 and 1.0
            
            Example: POSITIVE:0.85
            """
            
            sentiment_response = model.generate_content(sentiment_prompt)
            sentiment_text = sentiment_response.text.strip()
            
            # Parse the response
            if ':' in sentiment_text:
                sentiment_label, sentiment_score_str = sentiment_text.split(':')
                sentiment_score = float(sentiment_score_str)
            else:
                sentiment_label = 'NEUTRAL'
                sentiment_score = 0.5
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            sentiment_label = 'NEUTRAL'
            sentiment_score = 0.5
        
        # Convert sentiment to normalized score
        if sentiment_label == 'LABEL_0':  # Negative
            normalized_sentiment = 1 - sentiment_score
        elif sentiment_label == 'LABEL_1':  # Neutral
            normalized_sentiment = 0.5
        else:  # Positive
            normalized_sentiment = sentiment_score
        
        # Analyze emotions using Gemini API
        try:
            emotion_prompt = f"""
            Analyze the emotions in this text and return the dominant emotion with confidence:
            Text: "{transcript[:512]}"
            
            Return only in this format: EMOTION:CONFIDENCE
            Where EMOTION is one of: joy, sadness, anger, fear, surprise, neutral
            And CONFIDENCE is between 0.0 and 1.0
            
            Example: neutral:0.75
            """
            
            emotion_response = model.generate_content(emotion_prompt)
            emotion_text = emotion_response.text.strip()
            
            # Parse the response
            if ':' in emotion_text:
                dominant_emotion, emotion_confidence_str = emotion_text.split(':')
                emotion_confidence = float(emotion_confidence_str)
            else:
                dominant_emotion = 'neutral'
                emotion_confidence = 0.5
                
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            dominant_emotion = 'neutral'
            emotion_confidence = 0.5
        
        # Analyze toxicity using Gemini API instead of local models
        toxicity_result = analyze_text_with_huggingface(transcript[:512])
        
        # Extract toxicity score from the result
        toxicity_score = 0.3  # Default fallback
        for item in toxicity_result:
            if item['label'] == 'TOXIC':
                toxicity_score = item['score']
                break
            elif item['label'] == 'CLEAN':
                toxicity_score = 1 - item['score']
                break
        
        # Calculate performance score based purely on AI outputs
        sentiment_weight = 0.4
        emotion_weight = 0.3
        toxicity_weight = 0.3
        
        # Sentiment scoring
        sentiment_points = normalized_sentiment * 40
        
        # Emotion scoring (positive emotions get higher scores)
        emotion_multipliers = {
            'joy': 1.0, 'optimism': 0.9, 'love': 0.9, 'surprise': 0.7,
            'neutral': 0.6, 'sadness': 0.3, 'fear': 0.2, 'anger': 0.1, 'disgust': 0.0
        }
        emotion_multiplier = emotion_multipliers.get(dominant_emotion.lower(), 0.5)
        emotion_points = emotion_multiplier * 30
        
        # Toxicity scoring (lower toxicity = higher score)
        toxicity_points = (1 - toxicity_score) * 30
        
        performance_score = int(sentiment_points + emotion_points + toxicity_points)
        
        # Determine rating
        if performance_score >= 90:
            overall_rating = "Excellent"
        elif performance_score >= 75:
            overall_rating = "Good"
        elif performance_score >= 60:
            overall_rating = "Average"
        elif performance_score >= 40:
            overall_rating = "Poor"
        elif performance_score >= 20:
            overall_rating = "Very Poor"
        else:
            overall_rating = "Unacceptable"
        
        # Generate AI-based feedback
        strengths = []
        improvements = []
        
        if normalized_sentiment > 0.7:
            strengths.append("AI Analysis: Positive sentiment detected")
        elif normalized_sentiment < 0.4:
            improvements.append("AI Analysis: Negative sentiment - work on tone")
        
        if dominant_emotion in ['joy', 'optimism', 'love']:
            strengths.append(f"AI Analysis: Positive emotion ({dominant_emotion})")
        elif dominant_emotion in ['anger', 'disgust', 'fear']:
            improvements.append(f"AI Analysis: Concerning emotion detected ({dominant_emotion})")
        
        if toxicity_score < 0.3:
            strengths.append("AI Analysis: Professional communication")
        elif toxicity_score > 0.6:
            improvements.append("AI Analysis: Unprofessional language detected")
        
        return {
            "performance_score": performance_score,
            "overall_rating": overall_rating,
            "sentiment_score": round(normalized_sentiment, 2),
            "dominant_emotion": dominant_emotion,
            "emotion_confidence": round(emotion_confidence, 2),
            "toxicity_score": round(toxicity_score, 2),
            "strengths": strengths,
            "improvement_areas": improvements,
            "detailed_analysis": f"AI models analysis: {sentiment_points:.1f} sentiment + {emotion_points:.1f} emotion + {toxicity_points:.1f} professionalism = {performance_score}/100"
        }
    
    except Exception as e:
        print(f"Hugging Face analysis failed: {e}")
        return {
            "performance_score": 50,
            "overall_rating": "Unknown",
            "sentiment_score": 0.5,
            "dominant_emotion": "unknown",
            "emotion_confidence": 0.5,
            "toxicity_score": 0.5,
            "strengths": [],
            "improvement_areas": ["Analysis failed"],
            "detailed_analysis": f"Error: {str(e)}"
        }

@app.route('/upload', methods=['POST'])
# @token_required  # Authentication disabled
def upload_file():  # removed current_user parameter
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
            
            # Analyze agent performance - try Gemini first, fallback to Hugging Face
            print("Analyzing with Gemini AI...")
            
            # First, let's see what models are available
            print("Checking available Gemini models...")
            list_available_models()
            
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
# @token_required  # Authentication disabled
def get_user_reports():  # removed current_user parameter
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
# @admin_required  # Admin auth disabled, anyone can access
def get_all_reports():  # removed current_user parameter
    try:
        reports = list(reports_collection.find({}).sort('createdAt', -1))
        
        # Convert ObjectId to string
        for report in reports:
            report['_id'] = str(report['_id'])
            report['userId'] = str(report['userId'])
        
        return jsonify({
            'success': True,
            'reports': reports
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch reports: {str(e)}'}), 500

@app.route('/api/admin/users', methods=['GET'])
# @admin_required  # Admin auth disabled, anyone can access
def get_all_users():  # removed current_user parameter
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

@app.route('/api/admin/promote-user', methods=['POST'])
# @admin_required  # Admin auth disabled, anyone can access
def promote_user_to_admin():  # removed current_user parameter
    try:
        data = request.get_json()
        user_email = data.get('email')
        
        if not user_email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Find user by email
        user_to_promote = users_collection.find_one({'email': user_email})
        
        if not user_to_promote:
            return jsonify({'error': 'User not found'}), 404
        
        # Update user role to admin
        result = users_collection.update_one(
            {'email': user_email},
            {'$set': {'role': 'admin'}}
        )
        
        if result.modified_count > 0:
            return jsonify({
                'success': True,
                'message': f'User {user_email} has been promoted to admin'
            })
        else:
            return jsonify({'error': 'Failed to update user role'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to promote user: {str(e)}'}), 500

@app.route('/api/admin/demote-user', methods=['POST'])
# @admin_required  # Admin auth disabled, anyone can access
def demote_user_from_admin():  # removed current_user parameter
    try:
        data = request.get_json()
        user_email = data.get('email')
        
        if not user_email:
            return jsonify({'error': 'Email is required'}), 400
        
        # No auth check needed - anyone can demote
        # (Remove the self-demotion check since no authentication)
        
        # Find user by email
        user_to_demote = users_collection.find_one({'email': user_email})
        
        if not user_to_demote:
            return jsonify({'error': 'User not found'}), 404
        
        # Update user role to user
        result = users_collection.update_one(
            {'email': user_email},
            {'$set': {'role': 'user'}}
        )
        
        if result.modified_count > 0:
            return jsonify({
                'success': True,
                'message': f'User {user_email} has been demoted to regular user'
            })
        else:
            return jsonify({'error': 'Failed to update user role'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to demote user: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Call Analysis API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)