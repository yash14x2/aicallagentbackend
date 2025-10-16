from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Call Analysis API is running'})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # For now, return a mock response
        mock_result = {
            'success': True,
            'filename': file.filename,
            'transcript': 'This is a mock transcript for testing purposes.',
            'toxicity_analysis': [
                {'label': 'CLEAN', 'score': 0.95},
                {'label': 'TOXIC', 'score': 0.05}
            ],
            'agent_performance': {
                'performance_score': 85,
                'overall_rating': 'Good',
                'strengths': ['Professional tone', 'Clear communication'],
                'improvement_areas': ['Active listening'],
                'sentiment_score': 0.7,
                'dominant_emotion': 'neutral',
                'emotion_confidence': 0.8,
                'toxicity_score': 0.05,
                'detailed_analysis': 'Mock analysis for testing purposes.'
            }
        }
        
        return jsonify(mock_result)
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/user/reports', methods=['GET'])
def get_user_reports():
    # Return mock reports
    mock_reports = {
        'success': True,
        'reports': [
            {
                '_id': '12345',
                'filename': 'test_call.wav',
                'performanceScore': 85,
                'overallRating': 'Good',
                'createdAt': '2025-10-15T09:00:00Z'
            }
        ]
    }
    return jsonify(mock_reports)

@app.route('/api/admin/reports', methods=['GET'])
def get_all_reports():
    # Return mock reports for admin
    mock_reports = {
        'success': True,
        'reports': [
            {
                '_id': '12345',
                'userEmail': 'test@example.com',
                'userName': 'Test User',
                'filename': 'test_call.wav',
                'performanceScore': 85,
                'overallRating': 'Good',
                'createdAt': '2025-10-15T09:00:00Z'
            }
        ]
    }
    return jsonify(mock_reports)

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    # Return mock users
    mock_users = {
        'success': True,
        'users': [
            {
                '_id': '12345',
                'name': 'Test User',
                'email': 'test@example.com',
                'role': 'user',
                'createdAt': '2025-10-15T09:00:00Z'
            }
        ]
    }
    return jsonify(mock_users)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)