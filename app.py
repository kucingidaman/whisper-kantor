from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile
import os
import datetime
import threading
import webbrowser
import time
import torch

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Set custom model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Model directory: {MODEL_DIR}")
print("Letakkan model whisper.cpp (.bin) di folder models/")
print("Download dari: https://huggingface.co/ggerganov/whisper.cpp")

# Global variables
current_model = None
current_model_name = "base"
model_lock = threading.Lock()

# Model info for whisper.cpp
WHISPER_CPP_MODELS = {
    "tiny": {
        "file": "ggml-tiny.bin",
        "size": "75 MB",
        "speed": "~32x",
        "quality": "Rendah",
        "ram": "~390 MB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
    },
    "tiny.en": {
        "file": "ggml-tiny.en.bin",
        "size": "75 MB",
        "speed": "~32x",
        "quality": "Rendah (English)",
        "ram": "~390 MB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
    },
    "base": {
        "file": "ggml-base.bin",
        "size": "142 MB",
        "speed": "~16x",
        "quality": "Baik",
        "ram": "~500 MB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
    },
    "base.en": {
        "file": "ggml-base.en.bin",
        "size": "142 MB",
        "speed": "~16x",
        "quality": "Baik (English)",
        "ram": "~500 MB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    },
    "small": {
        "file": "ggml-small.bin",
        "size": "466 MB",
        "speed": "~6x",
        "quality": "Bagus",
        "ram": "~1 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    },
    "small.en": {
        "file": "ggml-small.en.bin",
        "size": "466 MB",
        "speed": "~6x",
        "quality": "Bagus (English)",
        "ram": "~1 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
    },
    "medium": {
        "file": "ggml-medium.bin",
        "size": "1.5 GB",
        "speed": "~2x",
        "quality": "Sangat Bagus",
        "ram": "~2.6 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
    },
    "medium.en": {
        "file": "ggml-medium.en.bin",
        "size": "1.5 GB",
        "speed": "~2x",
        "quality": "Sangat Bagus (English)",
        "ram": "~2.6 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
    },
    "large-v1": {
        "file": "ggml-large-v1.bin",
        "size": "2.9 GB",
        "speed": "~1x",
        "quality": "Terbaik (v1)",
        "ram": "~4.3 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin"
    },
    "large-v2": {
        "file": "ggml-large-v2.bin",
        "size": "2.9 GB",
        "speed": "~1x",
        "quality": "Terbaik (v2)",
        "ram": "~4.3 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin"
    },
    "large-v3": {
        "file": "ggml-large-v3.bin",
        "size": "2.9 GB",
        "speed": "~1x",
        "quality": "Terbaik (v3)",
        "ram": "~4.3 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
    },
    "large-v3-turbo": {
        "file": "ggml-large-v3-turbo.bin",
        "size": "1.6 GB",
        "speed": "~2x",
        "quality": "Terbaik (Turbo)",
        "ram": "~3.0 GB",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
    }
}

def get_available_models():
    """Check which models are available in models folder"""
    available = []
    for model_name, info in WHISPER_CPP_MODELS.items():
        model_path = os.path.join(MODEL_DIR, info['file'])
        if os.path.exists(model_path):
            available.append(model_name)
    return available

def get_recommended_model():
    """Determine recommended model based on available RAM and GPU"""
    try:
        has_gpu = torch.cuda.is_available()
        
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
        except:
            available_ram_gb = 4
        
        available = get_available_models()
        
        if has_gpu:
            if 'large-v3' in available and available_ram_gb >= 8:
                return "large-v3", "GPU + RAM tinggi"
            elif 'large-v2' in available and available_ram_gb >= 8:
                return "large-v2", "GPU + RAM tinggi"
            elif 'medium' in available and available_ram_gb >= 4:
                return "medium", "GPU + RAM sedang"
            elif 'small' in available:
                return "small", "GPU tersedia"
        
        if available_ram_gb >= 8:
            if 'medium' in available:
                return "medium", "RAM cukup untuk model sedang"
            elif 'small' in available:
                return "small", "RAM cukup"
        elif available_ram_gb >= 4:
            if 'small' in available:
                return "small", "RAM cukup untuk model kecil"
            elif 'base' in available:
                return "base", "RAM sedang"
        
        if 'tiny' in available:
            return "tiny", "Model tercepat"
        elif 'base' in available:
            return "base", "Model default"
        elif available:
            return available[0], "Model tersedia"
        
        return None, "Tidak ada model tersedia"
    except:
        available = get_available_models()
        return available[0] if available else None, "Model tersedia"

def load_whisper_model(model_name):
    """Load whisper model using faster-whisper"""
    model_info = WHISPER_CPP_MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} tidak dikenal")
    
    model_path = os.path.join(MODEL_DIR, model_info['file'])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_info['file']} tidak ditemukan di folder models/")
    
    # Determine device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}, Compute type: {compute_type}")
    
    # For faster-whisper, we use the model size name
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=MODEL_DIR
    )
    
    return model

# Load initial model
print("="*60)
print("Checking available models...")
available_models = get_available_models()

if available_models:
    print(f"Models tersedia: {', '.join(available_models)}")
    recommended_model, reason = get_recommended_model()
    
    if recommended_model:
        print(f"Recommended model: {recommended_model} ({reason})")
        try:
            current_model = load_whisper_model(recommended_model)
            current_model_name = recommended_model
            print(f"Model {recommended_model} berhasil dimuat!")
        except Exception as e:
            print(f"Error loading recommended model: {e}")
            # Try loading first available model
            for model_name in available_models:
                try:
                    current_model = load_whisper_model(model_name)
                    current_model_name = model_name
                    print(f"Loaded fallback model: {model_name}")
                    break
                except:
                    continue
else:
    print("PERINGATAN: Tidak ada model ditemukan di folder models/")
    print("="*60)
    print("Cara mendapatkan model whisper.cpp:")
    print("1. Download dari: https://huggingface.co/ggerganov/whisper.cpp/tree/main")
    print("2. Atau gunakan script download otomatis (lihat README)")
    print("3. Letakkan file .bin di folder models/")
    print("="*60)
    print("Contoh file model:")
    for name, info in WHISPER_CPP_MODELS.items():
        print(f"  - {info['file']} ({info['size']}) untuk model {name}")
    print("="*60)
    print("\n💡 Download cepat dengan script:")
    print("  python download_models.py base")
    print("  python download_models.py small")
    print("  python download_models.py large-v2")
    print("="*60)

@app.route('/api/models', methods=['GET'])
def get_models():
    available = get_available_models()
    recommended, reason = get_recommended_model() if available else (None, "Tidak ada model")
    
    return jsonify({
        "models": WHISPER_CPP_MODELS,
        "current": current_model_name if current_model else None,
        "available": available,
        "recommended": {
            "model": recommended,
            "reason": reason
        } if recommended else None,
        "has_gpu": torch.cuda.is_available(),
        "model_dir": MODEL_DIR
    })

@app.route('/api/change-model', methods=['POST'])
def change_model():
    global current_model, current_model_name
    
    data = request.json
    model_name = data.get('model')
    
    if model_name not in WHISPER_CPP_MODELS:
        return jsonify({'error': 'Model tidak valid'}), 400
    
    if model_name not in get_available_models():
        return jsonify({'error': f'Model {model_name} tidak tersedia. Download dulu model .bin nya'}), 400
    
    try:
        with model_lock:
            print(f"Changing model to: {model_name}")
            current_model = load_whisper_model(model_name)
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

# Global progress tracking
transcription_progress = {"progress": 0, "stage": "idle"}

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    global transcription_progress
    
    if not current_model:
        return jsonify({'error': 'Model belum dimuat. Pastikan ada model di folder models/'}), 500
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        transcription_progress = {"progress": 10, "stage": "Menyimpan file audio..."}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            print(f"Transcribing audio with model: {current_model_name}")
            
            transcription_progress = {"progress": 30, "stage": "Memproses audio..."}
            
            with model_lock:
                transcription_progress = {"progress": 50, "stage": "Transkripsi dengan Whisper..."}
                
                # faster-whisper returns segments, not a dict
                segments, info = current_model.transcribe(
                    tmp_path,
                    language='id',
                    beam_size=5,
                    vad_filter=True
                )
                
                transcription_progress = {"progress": 75, "stage": "Mengumpulkan hasil..."}
                
                # Collect all segments
                transcription = ""
                for segment in segments:
                    transcription += segment.text + " "
                
                transcription = transcription.strip()
            
            transcription_progress = {"progress": 100, "stage": "Selesai!"}
            
            return jsonify({
                'success': True,
                'transcription': transcription,
                'language': 'id',
                'model_used': current_model_name,
                'timestamp': datetime.datetime.now().isoformat()
            })
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            transcription_progress = {"progress": 0, "stage": "idle"}
                
    except Exception as e:
        print(f"Error: {str(e)}")
        transcription_progress = {"progress": 0, "stage": "idle"}
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress', methods=['GET'])
def get_progress():
    return jsonify(transcription_progress)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok' if current_model else 'no_model',
        'model': current_model_name if current_model else None,
        'gpu_available': torch.cuda.is_available(),
        'available_models': get_available_models()
    })

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("="*60)
    print("WHISPER SPEECH TO TEXT - WHISPER.CPP VERSION")
    print("="*60)
    print(f"Model directory: {MODEL_DIR}")
    if current_model:
        print(f"Current model: {current_model_name}")
        print(f"GPU Available: {torch.cuda.is_available()}")
    else:
        print("PERINGATAN: Tidak ada model yang dimuat!")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
