# 🔧 OpenCV Compatibility Patch for LipSync

## 📋 Overview

This patch ensures compatibility between different OpenCV packages that may be required by various dependencies in the Video Editor application.

## 🚨 Problem

When using multiple AI models (RealESRGAN, Wav2Lip, etc.), some dependencies might require different OpenCV packages:

1. **Main Application** → Requires `opencv-python` (with GUI support)
2. **Some AI Models** → May require `opencv-python-headless`

### Conflict
When both packages coexist, some constants like `CAP_PROP_WIDTH` might not be exposed correctly in the `cv2` namespace.

**Typical Error:**
```python
AttributeError: module 'cv2' has no attribute 'CAP_PROP_WIDTH'
```

## ✅ Solution Applied

### Patch in `utils/opencv_patch.py`

```python
import cv2

def apply_opencv_patch():
    """Apply patch for OpenCV constant compatibility issues."""
    constants_to_check = {
        'CAP_PROP_WIDTH': 3,
        'CAP_PROP_HEIGHT': 4,
        'CAP_PROP_FPS': 5,
        'CAP_PROP_FRAME_COUNT': 7,
        'CAP_PROP_POS_FRAMES': 1,
    }
    
    for const_name, const_value in constants_to_check.items():
        if not hasattr(cv2, const_name):
            setattr(cv2, const_name, const_value)
```

### OpenCV Constant Values (Standard)

| Constant | Value | Description |
|----------|-------|-------------|
| `CAP_PROP_POS_FRAMES` | 1 | Current frame position (0-based index) |
| `CAP_PROP_WIDTH` | 3 | Frame width in pixels |
| `CAP_PROP_HEIGHT` | 4 | Frame height in pixels |
| `CAP_PROP_FPS` | 5 | Frame rate (frames per second) |
| `CAP_PROP_FRAME_COUNT` | 7 | Total number of frames |

Reference: [OpenCV VideoCaptureProperties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)

## 📦 Required Dependencies

```bash
opencv-python==4.9.0.80
numpy<2.0.0
```

## 🔄 Integration

The patch is automatically applied on application startup in `main.py`:

```python
# Apply OpenCV patch for compatibility
from utils.opencv_patch import apply_opencv_patch
apply_opencv_patch()
```

## ✅ Verification

The patch logs its actions:

**If patch is needed:**
```
⚠️ OpenCV patch applied for missing constants: CAP_PROP_WIDTH, CAP_PROP_HEIGHT
✓ OpenCV constants patched successfully
```

**If no patch needed:**
```
✓ OpenCV constants already available, no patch needed
```

## 🔍 Troubleshooting

### If you see OpenCV-related errors:

1. **Check OpenCV installation:**
```bash
pip list | grep opencv
```

Expected output:
```
opencv-python                 4.9.0.80
```

2. **Verify numpy version:**
```bash
pip list | grep numpy
```

Make sure numpy is <2.0 (requirement for compatibility with torch 2.1.0)

3. **Reinstall if necessary:**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.9.0.80
```

## 🎯 What This Patch Does

✅ **Ensures constants are available** - Adds missing OpenCV constants if they're not found  
✅ **Non-intrusive** - Only patches if constants are missing  
✅ **Automatic** - Applies on import, no manual intervention needed  
✅ **Logged** - Clear logging of patch application  

## 🚫 What NOT to Do

❌ **Don't install opencv-python-headless** unless specifically required  
❌ **Don't upgrade numpy to 2.x** - Breaks compatibility with torch 2.1.0  
❌ **Don't modify OpenCV source** - Use this patch instead  

## 📝 Notes

- This patch is specifically designed for the LipSync tab integration
- The patch is lightweight and has minimal performance impact
- All standard OpenCV functionality remains unchanged
- Only missing constants are added, existing values are never modified

## 🔗 Related Files

- `utils/opencv_patch.py` - Patch implementation
- `main.py` - Patch application on startup
- `tabs/lipsync_tab.py` - Uses OpenCV for video processing
- `tabs/upscaler_tab.py` - Uses OpenCV for video processing
- `requirements.txt` - Specifies opencv-python version

## 📊 Compatibility Matrix

| Component | OpenCV Requirement | Status |
|-----------|-------------------|--------|
| RealESRGAN Upscaler | opencv-python | ✅ Compatible |
| Wav2Lip LipSync | opencv-python | ✅ Compatible (with patch) |
| Video Processing | opencv-python | ✅ Compatible |
| Image Processing | opencv-python | ✅ Compatible |

---

**Last Updated:** February 2026  
**Patch Version:** 1.0  
**Compatibility:** Python 3.11+, OpenCV 4.9.0.80
