# ═══════════════════════════════════════════════
#  THE GARGANTUAS VIDEO EDITOR - COLAB SETUP
#  Python 3.11 + GRADIO_SHARE + Matplotlib fix + cryptography fix
# ═══════════════════════════════════════════════

# 1️⃣ Go to /content and clean previous installs
%cd /content
!rm -rf The-Gargantuas-Video-Editor
!rm -rf py311

# 2️⃣ Install system dependencies and Python 3.11
!apt-get update -y
!apt-get install -y ffmpeg git curl python3.11 python3.11-venv python3.11-distutils

# 3️⃣ Create Python 3.11 virtual environment
!python3.11 -m venv /content/py311
!/content/py311/bin/python -m pip install --upgrade pip setuptools wheel

# 4️⃣ Clone repository
!git clone https://github.com/TheGargantuas/The-Gargantuas-Video-Editor.git
%cd /content/The-Gargantuas-Video-Editor

# 5️⃣ Install compatible PyTorch stack
!/content/py311/bin/pip install --no-cache-dir numpy==1.26.4
!/content/py311/bin/pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# 6️⃣ Install CRYPTOGRAPHY FIRST (required by api_key_manager)
print("📦 Installing cryptography (critical dependency)...")
!/content/py311/bin/pip install --no-cache-dir cryptography==46.0.5

# 7️⃣ Install core dependencies
!/content/py311/bin/pip install --no-cache-dir gradio opencv-python==4.9.0.80 pillow ffmpeg-python python-dotenv

# 8️⃣ Install upscaler and lipsync dependencies
!/content/py311/bin/pip install --no-cache-dir basicsr realesrgan gfpgan facexlib librosa soundfile scipy pydub

# 9️⃣ Install subtitle generation tools
!/content/py311/bin/pip install --no-cache-dir faster-whisper openai

# 🔟 Fix Matplotlib backend for Gradio/server usage (prevents matplotlib_inline errors)
%env MPLBACKEND=Agg
%env MATPLOTLIBRC=/tmp/matplotlibrc
!printf "[matplotlib]\nbackend = Agg\n" > /tmp/matplotlibrc

# 1️⃣1️⃣ Enable public Gradio sharing
%env GRADIO_SHARE=true

# 1️⃣2️⃣ Launch the app
print("\n🚀 Launching The Gargantuas Video Editor...")
print("💡 TIP: Use Groq API (100k tokens/day FREE) for subtitles: https://console.groq.com/keys")
print("💡 TIP: Enable 'Single Pass' validation mode to save API quota\n")
!/content/py311/bin/python main.py
