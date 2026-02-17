"""
Flask Web Application for Idea Intake AI Pipeline

This web app allows users to:
1. Upload Excel files with idea intake data
2. Process the data through the AI pipeline
3. View recommendations on the web interface
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Import pipeline functions
from routes.pipeline_routes import run_pipeline

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('results')
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process through pipeline."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload .xlsx or .xls file'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(filepath)
        
        # Get similarity threshold from request (optional)
        similarity_threshold = float(request.form.get('similarity_threshold', 0.78))
        
        # Run pipeline
        output_filename = f"{timestamp}_analysis.json"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        
        result = run_pipeline(
            input_file=str(filepath),
            output_file=str(output_path),
            similarity_threshold=similarity_threshold,
            save_intermediate=False,
            verbose=False
        )
        
        # Clean up uploaded file
        filepath.unlink()
        
        # Return results
        return jsonify({
            'success': True,
            'result': result,
            'output_file': output_filename
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download result JSON file."""
    try:
        filepath = app.config['RESULTS_FOLDER'] / filename
        if filepath.exists():
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    print("="*70)
    print("IDEA INTAKE AI PIPELINE - WEB APPLICATION")
    print("="*70)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5001")
    print("\nPress CTRL+C to stop the server\n")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)

# Made with Bob
