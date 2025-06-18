#!/usr/bin/env python
"""
Multimodal AI Media Analysis System - Web Application
Supports file upload and real-time analysis
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import uuid
from multimodal_analyzer import MultiModalAnalyzer, MediaAnalysisResult

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Supported file formats
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac'}
ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global variables for analysis status
analysis_status = {}
analyzer = None

def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    """Get file type"""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_VIDEO_EXTENSIONS:
        return 'video'
    elif ext in ALLOWED_AUDIO_EXTENSIONS:
        return 'audio'
    return 'unknown'

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{file_id}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        # Save file
        file.save(filepath)
        
        # Initialize analysis status
        analysis_status[file_id] = {
            'status': 'uploaded',
            'progress': 0,
            'filename': filename,
            'filepath': filepath,
            'file_type': get_file_type(filename),
            'start_time': datetime.now().isoformat()
        }
        
        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'file_type': analysis_status[file_id]['file_type']
        })
    
    return jsonify({'error': 'Unsupported file format'}), 400

@app.route('/analyze/<file_id>', methods=['POST'])
def analyze(file_id):
    """Start file analysis"""
    global analyzer
    
    if file_id not in analysis_status:
        return jsonify({'error': 'File not found'}), 404
    
    if analysis_status[file_id]['status'] == 'analyzing':
        return jsonify({'error': 'Analysis already in progress'}), 400
    
    # Initialize analyzer if not already initialized
    if analyzer is None:
        try:
            analyzer = MultiModalAnalyzer(device="auto")
        except Exception as e:
            return jsonify({'error': f'Failed to initialize analyzer: {str(e)}'}), 500
    
    # Run analysis in background thread
    thread = threading.Thread(target=run_analysis, args=(file_id,))
    thread.start()
    
    return jsonify({'message': 'Analysis started', 'file_id': file_id})

def run_analysis(file_id):
    """Run analysis in background"""
    global analyzer
    
    try:
        # Update status
        analysis_status[file_id]['status'] = 'analyzing'
        analysis_status[file_id]['progress'] = 10
        
        filepath = analysis_status[file_id]['filepath']
        file_type = analysis_status[file_id]['file_type']
        
        # For audio files, only perform speech recognition
        if file_type == 'audio':
            analysis_status[file_id]['progress'] = 30
            
            # Create simplified result object
            transcripts = analyzer._transcribe_audio(Path(filepath))
            
            results = MediaAnalysisResult(
                video_path=filepath,
                scenes=[],  # No scenes for audio files
                objects=[],  # No objects for audio files
                transcripts=transcripts,
                embeddings=analyzer._generate_embeddings([], [], transcripts),
                processing_time=0
            )
            
            analysis_status[file_id]['progress'] = 90
        else:
            # Full analysis for video files
            # Simulate progress updates
            analysis_status[file_id]['progress'] = 20
            
            results = analyzer.analyze_video(filepath, sample_rate=30)
            
            analysis_status[file_id]['progress'] = 90
        
        # Save results
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{file_id}.json")
        save_results(results, result_file)
        
        # Update status
        analysis_status[file_id]['status'] = 'completed'
        analysis_status[file_id]['progress'] = 100
        analysis_status[file_id]['result_file'] = result_file
        analysis_status[file_id]['results'] = results
        
    except Exception as e:
        analysis_status[file_id]['status'] = 'error'
        analysis_status[file_id]['error'] = str(e)
        print(f"Analysis error: {e}")

def save_results(results, filepath):
    """Save analysis results to JSON file"""
    data = {
        'video_path': results.video_path,
        'scenes': results.scenes,
        'objects': results.objects,
        'transcripts': results.transcripts,
        'processing_time': results.processing_time,
        'embeddings_shape': results.embeddings.shape if len(results.embeddings) > 0 else [0, 0]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route('/status/<file_id>')
def get_status(file_id):
    """Get analysis status"""
    if file_id not in analysis_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = analysis_status[file_id].copy()
    
    # If analysis is complete, add simplified result summary
    if status['status'] == 'completed' and 'results' in status:
        results = status['results']
        
        # Statistics
        unique_objects = {}
        for obj in results.objects:
            if obj['class'] not in unique_objects:
                unique_objects[obj['class']] = 0
            unique_objects[obj['class']] += 1
        
        status['summary'] = {
            'scenes_count': len(results.scenes),
            'objects_count': len(unique_objects),
            'unique_objects': list(unique_objects.keys())[:10],  # Top 10
            'transcript_segments': len(results.transcripts.get('segments', [])),
            'language': results.transcripts.get('language', 'unknown'),
            'processing_time': results.processing_time
        }
        
        # Don't send full results object
        status.pop('results', None)
    
    return jsonify(status)

@app.route('/results/<file_id>')
def get_results(file_id):
    """Get detailed analysis results"""
    if file_id not in analysis_status:
        return jsonify({'error': 'File not found'}), 404
    
    if analysis_status[file_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    # Read results file
    result_file = analysis_status[file_id]['result_file']
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return jsonify(results)

@app.route('/search/<file_id>', methods=['POST'])
def search(file_id):
    """Execute natural language query"""
    global analyzer
    
    if file_id not in analysis_status:
        return jsonify({'error': 'File not found'}), 404
    
    if analysis_status[file_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        results = analysis_status[file_id]['results']
        matches = analyzer.search(query, results, top_k=5)
        
        # Format match results
        formatted_matches = []
        for match in matches:
            formatted_match = {
                'type': match['type'],
                'score': float(match['score']),
                'content': {}
            }
            
            if match['type'] == 'scene':
                formatted_match['content'] = {
                    'scene_id': match['content'].get('scene_id'),
                    'start_time': match['content'].get('start_time'),
                    'end_time': match['content'].get('end_time'),
                    'description': match['content'].get('description')
                }
            elif match['type'] == 'object':
                formatted_match['content'] = {
                    'class': match['content'].get('class')
                }
            elif match['type'] == 'transcript':
                formatted_match['content'] = {
                    'text': match['content'].get('text'),
                    'start': match['content'].get('start'),
                    'end': match['content'].get('end')
                }
            
            formatted_matches.append(formatted_match)
        
        return jsonify({'matches': formatted_matches})
        
    except Exception as e:
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/download/<file_id>')
def download_results(file_id):
    """Download analysis results"""
    if file_id not in analysis_status:
        return jsonify({'error': 'File not found'}), 404
    
    if analysis_status[file_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    result_file = analysis_status[file_id]['result_file']
    filename = f"{analysis_status[file_id]['filename']}_analysis.json"
    
    return send_file(result_file, as_attachment=True, download_name=filename)
