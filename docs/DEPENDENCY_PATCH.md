# Dependency Management

## Automatic Conflict Resolution

This application automatically resolves dependency conflicts at setup and startup.

### The Problem

Some packages have conflicting version requirements:
- `rembg` requires `numpy>=2`
- `moviepy` requires `Pillow<12.0`
- `albumentations` requires `opencv-python-headless>=4.9.0.80`
- Most ML packages require `numpy<2`

### The Solution

The application includes an **automatic dependency patch** system:

1. **At Setup Time**: All setup scripts run `utils/dependency_patch.py`
2. **At Startup**: `main.py` automatically checks and fixes conflicts
3. **Manual Fix**: Run `python utils/dependency_patch.py` anytime

### Compatible Versions

The patch ensures these versions are installed:
```
numpy>=1.24.4,<2
Pillow>=10.0.0,<12.0
opencv-python==4.9.0.80
opencv-python-headless>=4.9.0.80
rembg==2.0.72
```

### Setup Scripts

All setup scripts apply the patch automatically:

- **Local (macOS/Linux)**: `./setup.sh`
- **Local (Windows)**: `setup.bat`
- **Google Colab Shell**: `./setup_colab.sh`
- **Google Colab Notebook**: `colab_notebook_setup.py`

### Manual Fix

If you encounter dependency errors:

```bash
# Activate venv first
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate.bat  # Windows

# Run the patch
python utils/dependency_patch.py
```

Or use the dedicated fix script:
```bash
./fix_rembg_dependencies.sh  # Linux/macOS
```

### How It Works

The patch:
1. Detects if `rembg` is installed (trigger for conflicts)
2. Checks versions of `numpy`, `Pillow`, `opencv-python`
3. Reinstalls incompatible versions with correct ones
4. Runs silently at startup (logs to console if issues found)

### Technical Details

See `utils/dependency_patch.py` for implementation.

The patch is similar to `utils/opencv_patch.py` but handles version conflicts instead of runtime patches.

### Disabling Auto-Patch

If you need to disable the automatic patch:

Edit `main.py` and comment out:
```python
# from utils.dependency_patch import apply_dependency_patch
# apply_dependency_patch(silent=True)
```

**Note**: This may cause import errors if dependencies are incompatible.
