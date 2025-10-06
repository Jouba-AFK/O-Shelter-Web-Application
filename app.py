from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
JSON_FILE = 'data.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data():
    if not os.path.exists(JSON_FILE):
        return []
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if not request.data:
        return jsonify({'error': 'No image data provided'}), 400
    image_data = request.data
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_esp32_capture.jpg"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        with open(filepath, 'wb') as f:
            f.write(image_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    data = load_data()
    data.append({'filename': filename, 'timestamp': datetime.now().isoformat()})
    save_data(data)

    return jsonify({'message': 'Upload successful', 'filename': filename}), 200

@app.route('/images_list')
def images_list():
    data = load_data()
    filenames = [item['filename'] for item in sorted(data, key=lambda x: x['timestamp'], reverse=True)]
    return jsonify(filenames)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
if __name__ == '__main__':
    app.run(debug=True)