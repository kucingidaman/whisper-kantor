import os
import urllib.request
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = {
    "tiny": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "size": "75 MB"
    },
    "tiny.en": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        "size": "75 MB"
    },
    "base": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "size": "142 MB"
    },
    "base.en": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        "size": "142 MB"
    },
    "small": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "size": "466 MB"
    },
    "small.en": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        "size": "466 MB"
    },
    "medium": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "size": "1.5 GB"
    },
    "medium.en": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
        "size": "1.5 GB"
    },
    "large-v1": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
        "size": "2.9 GB"
    },
    "large-v2": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
        "size": "2.9 GB"
    },
    "large-v3": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        "size": "2.9 GB"
    },
    "large-v3-turbo": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        "size": "1.6 GB"
    }
}

def download_model(model_name):
    if model_name not in MODELS:
        print(f"Model {model_name} tidak tersedia!")
        print(f"Model tersedia: {', '.join(MODELS.keys())}")
        return False
    
    url = MODELS[model_name]["url"]
    filename = os.path.basename(url)
    filepath = os.path.join(MODEL_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"✓ Model {model_name} sudah ada di {filepath}")
        return True
    
    print(f"Downloading {model_name} ({MODELS[model_name]['size']}) from HuggingFace")
    print(f"URL: {url}")
    print(f"Saving to: {filepath}")
    print("Mohon tunggu, ini mungkin memakan waktu...")
    print()
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                downloaded_mb = count * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {percent}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\n✓ Model {model_name} berhasil didownload!")
        print(f"✓ File tersimpan di: {filepath}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

if __name__ == '__main__':
    print("="*70)
    print("WHISPER.CPP MODEL DOWNLOADER")
    print("="*70)
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
        download_model(model_name)
    else:
        print("\n📦 Model yang tersedia untuk download:\n")
        print("MULTILINGUAL (Support 99 bahasa termasuk Indonesia):")
        print("  tiny       - 75 MB   (tercepat, akurasi rendah)")
        print("  base       - 142 MB  (cepat, akurasi baik) ⭐ RECOMMENDED")
        print("  small      - 466 MB  (sedang, akurasi bagus)")
        print("  medium     - 1.5 GB  (lambat, akurasi sangat bagus)")
        print("  large-v1   - 2.9 GB  (paling lambat, akurasi terbaik)")
        print("  large-v2   - 2.9 GB  (improved v2, akurasi terbaik)")
        print("  large-v3   - 2.9 GB  (latest v3, akurasi terbaik)")
        print("  large-v3-turbo - 1.6 GB (v3 optimized, lebih cepat)")
        print()
        print("ENGLISH ONLY (Hanya bahasa Inggris, lebih cepat):")
        print("  tiny.en    - 75 MB   (tercepat)")
        print("  base.en    - 142 MB  (cepat)")
        print("  small.en   - 466 MB  (sedang)")
        print("  medium.en  - 1.5 GB  (lambat)")
        print()
        print("="*70)
        print("\n📖 Cara pakai:")
        print("  python download_models.py base        ← Download 1 model")
        print("  python download_models.py small")
        print("  python download_models.py large-v2")
        print()
        print("💡 Tips:")
        print("  - Untuk mulai, download 'base' atau 'small'")
        print("  - Model .en lebih cepat tapi hanya support English")
        print("  - Model large-v3-turbo: kualitas tinggi tapi lebih cepat dari v3 biasa")
        print("="*70)
        print()
        
        choice = input("Download model mana? (base/small/large-v2): ").strip().lower()
        if choice:
            download_model(choice)
        else:
            print("\nTidak ada pilihan. Keluar.")