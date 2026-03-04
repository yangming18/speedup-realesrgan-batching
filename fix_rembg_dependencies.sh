#!/bin/bash
# Fix dependencies conflicts for rembg integration
# Run this after installing rembg to resolve version conflicts

echo "🔧 Fixing dependency conflicts..."

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "❌ Virtual environment not found!"
        exit 1
    fi
fi

echo "📦 Installing compatible versions..."

# Fix numpy and Pillow versions
pip install --force-reinstall "numpy>=1.24.4,<2" "Pillow>=10.0.0,<12.0"

# Fix opencv versions
pip install --force-reinstall opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80

# Verify installations
echo ""
echo "✅ Dependencies fixed! Verifying..."
python -c "import numpy; print(f'numpy: {numpy.__version__}')" 2>/dev/null && \
python -c "import PIL; print(f'Pillow: {PIL.__version__}')" 2>/dev/null && \
python -c "import cv2; print(f'opencv-python: {cv2.__version__}')" 2>/dev/null && \
python -c "import rembg; print(f'rembg: OK')" 2>/dev/null && \
echo "" && \
echo "🎉 All packages working correctly!"

exit 0
