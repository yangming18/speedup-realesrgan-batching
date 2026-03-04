#!/bin/bash
# Google Colab Setup Script for The-Gargantuas-Video-Editor
# Run this script BEFORE executing main.py on Colab

set -e  # Exit on error

echo "🚀 Setting up The-Gargantuas-Video-Editor on Google Colab..."
echo ""

# Check if running in Colab
if [ ! -d "/content" ]; then
    echo "⚠️  WARNING: This script is designed for Google Colab environment"
    echo "   It expects /content directory. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Step 1/6: Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Step 2/6: Installing CRITICAL dependencies first..."
# Install cryptography FIRST before other packages try to import it
pip install cryptography==46.0.5
echo "✅ cryptography installed"

echo "📦 Step 3/6: Installing PyTorch (optimized for Colab GPU)..."
# Colab has CUDA 12.x, use torch 2.1.x for compatibility
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
echo "✅ PyTorch installed"

echo "📦 Step 4/6: Installing core dependencies..."
# Install numpy first (many packages depend on it)
pip install "numpy<2"
echo "✅ NumPy installed"

echo "📦 Step 5/6: Installing all requirements..."
# Install remaining requirements from requirements.txt
pip install -r requirements.txt
echo "✅ All requirements installed"

echo "📦 Step 5.5/6: Fixing dependency conflicts..."
# Fix conflicts automatically
python -c "from utils.dependency_patch import fix_dependencies; fix_dependencies(verbose=True)" || echo "⚠️  Could not run dependency patch (will run at startup)"
echo "✅ Dependencies fixed"

echo "📦 Step 6/6: Installing subtitle generation tools..."
# These might need special handling
pip install faster-whisper>=1.0.0
pip install openai>=1.0.0
echo "✅ Subtitle tools installed"

echo ""
echo "✅ Setup complete! You can now run:"
echo "   python main.py"
echo ""
echo "💡 Tips for Google Colab:"
echo "   • Use Groq API (100k tokens/day FREE) instead of Gemini (20 req/day)"
echo "   • Get Groq key: https://console.groq.com/keys"
echo "   • Enable GPU: Runtime → Change runtime type → GPU"
echo "   • For public access, the app will use share=True automatically"
echo ""
