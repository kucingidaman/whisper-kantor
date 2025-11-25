from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import whisper
import tempfile
import os
import datetime
import threading
import webbrowser
import time
import torch

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Global variables
current_model = None
current_model_name = "base"
model_lock = threading.Lock()

# Model recommendations based on system specs
def get_recommended_model():
    try:
        has_gpu = torch.cuda.is_available()
        
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
        except:
            available_ram_gb = 4
        
        if has_gpu:
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 8:
                    return "large-v2", "GPU tersedia dengan VRAM tinggi"
                elif vram_gb >= 4:
                    return "medium", "GPU tersedia dengan VRAM sedang"
                else:
                    return "small", "GPU tersedia dengan VRAM terbatas"
            except:
                return "small", "GPU tersedia"
        else:
            if available_ram_gb >= 8:
                return "medium", "RAM cukup untuk model sedang"
            elif available_ram_gb >= 4:
                return "small", "RAM cukup untuk model kecil"
            else:
                return "tiny", "RAM terbatas"
    except:
        return "base", "Konfigurasi standar"

# Set custom model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['XDG_CACHE_HOME'] = MODEL_DIR

print(f"Model directory: {MODEL_DIR}")
print("Models akan disimpan di folder: models/")

# Load initial model
print("Checking system specifications...")
recommended_model, reason = get_recommended_model()
print(f"Recommended model: {recommended_model} ({reason})")
print(f"Loading default model: base")

try:
    current_model = whisper.load_model("base", download_root=MODEL_DIR)
    current_model_name = "base"
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Model akan diunduh otomatis saat pertama kali digunakan")

@app.route('/api/models', methods=['GET'])
def get_models():
    models_info = {
        "tiny": {"size": "39 MB", "speed": "~32x", "quality": "Rendah", "ram": "~1 GB"},
        "base": {"size": "74 MB", "speed": "~16x", "quality": "Baik", "ram": "~1 GB"},
        "small": {"size": "244 MB", "speed": "~6x", "quality": "Bagus", "ram": "~2 GB"},
        "medium": {"size": "769 MB", "speed": "~2x", "quality": "Sangat Bagus", "ram": "~5 GB"},
        "large": {"size": "1550 MB", "speed": "~1x", "quality": "Terbaik", "ram": "~10 GB"},
        "large-v2": {"size": "1550 MB", "speed": "~1x", "quality": "Terbaik (v2)", "ram": "~10 GB"}
    }
    
    # Check which models are already downloaded
    downloaded_models = []
    for model_name in models_info.keys():
        model_file = os.path.join(MODEL_DIR, f"{model_name}.pt")
        if os.path.exists(model_file):
            downloaded_models.append(model_name)
    
    recommended, reason = get_recommended_model()
    
    return jsonify({
        "models": models_info,
        "current": current_model_name,
        "recommended": {
            "model": recommended,
            "reason": reason
        },
        "has_gpu": torch.cuda.is_available(),
        "downloaded": downloaded_models,
        "model_dir": MODEL_DIR
    })

@app.route('/api/change-model', methods=['POST'])
def change_model():
    global current_model, current_model_name
    
    data = request.json
    model_name = data.get('model', 'base')
    
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2"]
    if model_name not in valid_models:
        return jsonify({'error': 'Invalid model name'}), 400
    
    try:
        with model_lock:
            print(f"Loading model: {model_name} from {MODEL_DIR}")
            current_model = whisper.load_model(model_name, download_root=MODEL_DIR)
            current_model_name = model_name
            print(f"Model {model_name} loaded successfully!")
        
        return jsonify({
            'success': True,
            'model': model_name,
            'message': f'Model berhasil diubah ke {model_name}'
        })
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            print(f"Transcribing audio with model: {current_model_name}")
            
            with model_lock:
                result = current_model.transcribe(
                    tmp_path, 
                    language='id', 
                    fp16=torch.cuda.is_available(),
                    verbose=False
                )
            
            transcription = result['text']
            
            return jsonify({
                'success': True,
                'transcription': transcription,
                'language': result.get('language', 'id'),
                'model_used': current_model_name,
                'timestamp': datetime.datetime.now().isoformat()
            })
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'model': current_model_name,
        'gpu_available': torch.cuda.is_available()
    })

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("="*60)
    print("WHISPER SPEECH TO TEXT APPLICATION")
    print("="*60)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Current model: {current_model_name}")
    print(f"Recommended model: {recommended_model} ({reason})")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    print("\nCATATAN:")
    print("- Semua model akan disimpan di folder 'models/'")
    print("- Model hanya diunduh sekali, selanjutnya offline")
    print("- Anda bisa backup folder 'models/' untuk digunakan lagi")
    print("="*60)
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=True)