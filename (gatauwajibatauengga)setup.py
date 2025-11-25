import subprocess
import sys
import os

HTML_CONTENT = '''[ISI DARI FILE frontend/index.html DI ATAS]'''

def install_requirements():
    print("="*60)
    print("Installing required packages...")
    print("="*60)
    
    packages = [
        'flask==3.0.0',
        'flask-cors==4.0.0',
        'openai-whisper',
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except Exception as e:
            print(f"Warning: Failed to install {package}: {e}")
    
    print("\nAll packages installed!")

def create_frontend():
    print("\nCreating frontend directory...")
    os.makedirs('frontend', exist_ok=True)
    
    # HTML content akan di-paste manual atau baca dari string di atas
    print("Please manually create 'frontend/index.html' file")
    print("Frontend directory created!")

if __name__ == '__main__':
    print("="*60)
    print("WHISPER SPEECH TO TEXT - SETUP")
    print("="*60)
    install_requirements()
    create_frontend()
    print("="*60)
    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Create 'frontend' folder")
    print("2. Copy index.html content to 'frontend/index.html'")
    print("3. Run: python app.py")
    print("="*60)