# Project Structure

```
Video Editor/
│
├── 📄 main.py                          # Main application entry point
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .python-version                  # Python version for pyenv
├── 📄 .gitignore                       # Git ignore rules
├── 📄 README.md                        # Project overview and main documentation
│
├── 📁 docs/                            # Documentation folder
│   ├── README.md                       # Documentation index
│   ├── QUICKSTART.md                   # Quick start guide for users
│   ├── LIPSYNC.md                      # LipSync feature complete guide
│   ├── I18N.md                         # Internationalization guide
│   ├── OPENCV_PATCH.md                 # OpenCV compatibility documentation
│   ├── PROJECT_STRUCTURE.md            # This file
│   └── CHANGELOG.md                    # Version history and updates
│
├── 🔧 setup.sh                         # Setup script for macOS/Linux
├── 🔧 setup.bat                        # Setup script for Windows
├── 🔧 run.sh                           # Run script for macOS/Linux
├── 🔧 run.bat                          # Run script for Windows
│
├── 📁 config/                          # Configuration files
│   ├── __init__.py
│   ├── config.py                       # Main configuration (models, colors, settings)
│   └── user_settings.json              # (Runtime) User preferences (language, etc.)
│
├── 📁 locales/                         # Translation files (i18n)
│   ├── en.json                         # English translations
│   └── it.json                         # Italian translations
│
├── 📁 utils/                           # Utility modules
│   ├── __init__.py
│   ├── temp_manager.py                 # Temporary file management
│   ├── device_manager.py               # Compute device (CPU/GPU/MPS) management
│   ├── i18n.py                         # Internationalization manager
│   └── opencv_patch.py                 # OpenCV compatibility patch
│
├── 📁 tabs/                            # Application tabs (features)
│   ├── __init__.py
│   ├── upscaler_tab.py                 # AI upscaling functionality
│   ├── lipsync_tab.py                  # AI lip-sync functionality (4 models)
│   ├── settings_tab.py                 # Settings and preferences
│   └── support_tab.py                  # Support/info tab
│
├── 📁 theme/                           # UI styling
│   ├── __init__.py
│   └── custom_theme.py                 # Custom Gradio theme (Amber/Red/Gray)
│
├── 📁 img/                             # Images and assets
│   └── gradio/                         # Gradio UI screenshots
│
├── 📁 models/                          # (Runtime: AI models - not in git, auto-downloaded)
│   ├── README.md                       # Models documentation
│   └── lipsync/                        # LipSync models (150MB-2GB each)
│       ├── wav2lip/                    # Wav2Lip model repository
│       ├── sadtalker/                  # SadTalker model (downloaded on first use)
│       └── video_retalking/            # Video-Retalking model (downloaded on first use)
│
├── 📁 example/                         # Example media files
│   ├── example_video/                  # Example videos for comparison
│   └── lipsync/                        # LipSync example files
│
├── 📁 temp/                            # (Runtime: temporary files - not in git)
│   ├── frames/                         # Extracted video frames
│   ├── output_frames/                  # Processed video frames
│   └── output/                         # Temporary output files
│
└── 📁 .venv/                           # (Runtime: virtual environment - not in git)
```

## File Descriptions

### Root Level

- **main.py**: Entry point for the application. Initializes all components and launches the Gradio interface.
- **requirements.txt**: Lists all Python package dependencies.
- **.python-version**: Specifies Python 3.11.0 for pyenv.

### Configuration (`config/`)

- **config.py**: Central configuration including:
  - YouTube channel information (for support tab)
  - RealESRGAN model configurations and URLs
  - Color scheme definitions
  - Application settings
  - Directory paths

### Utilities (`utils/`)

- **temp_manager.py**: Manages temporary files:
  - Creates/cleans temp directories
  - Handles frame extraction folders
  - Cleanup on app exit

- **device_manager.py**: Manages compute devices:
  - Detects available devices (CPU/CUDA/MPS)
  - Provides PyTorch device objects
  - Device information and switching

### Tabs (`tabs/`)

- **upscaler_tab.py**: Image and video upscaling:
  - RealESRGAN model loading and management
  - Automatic file type detection (image vs video)
  - Image upscaling with format selection
  - Video frame-by-frame processing with FPS control
  - Progress tracking
  - Gradio UI components

- **support_tab.py**: YouTube channel promotion:
  - Subscribe button with custom YouTube styling
  - Embedded playlist viewer
  - Support options (Subscribe, Share, Coffee)
  - No authentication required - fully optional

### Theme (`theme/`)

- **custom_theme.py**: Custom Gradio theme:
  - Color scheme (Amber, Red, Gray, Black, White)
  - Background image handling with base64 encoding
  - Parallax scrolling effect
  - Custom CSS styling
  - Button styles and animations
  - Interactive element styling

### Images (`img/`)

- **background.jpg**: Background image used for parallax effect:
  - Converted to base64 and embedded in CSS
  - Fixed position with opacity for subtle effect
  - Enhances visual appeal without affecting readability

## Adding New Features

To add a new tab/feature:

1. Create a new file in `tabs/` directory:
   ```
   tabs/new_feature_tab.py
   ```

2. Implement the tab class:
   ```python
   class NewFeatureTab:
       def __init__(self, temp_manager, device_manager):
           # Initialize
           pass
       
       def create_tab(self):
           with gr.Tab("✨ New Feature"):
               # Create UI
               pass
   ```

3. Import and use in `main.py`:
   ```python
   from tabs.new_feature_tab import NewFeatureTab
   
   # In __init__:
   self.new_feature_tab = NewFeatureTab(self.temp_manager, self.device_manager)
   
   # In create_interface:
   self.new_feature_tab.create_tab()
   ```

## Data Flow

```
User uploads file (image or video)
    ↓
File automatically detected by extension
    ↓
Saved to temp directory
    ↓
Processed by selected AI model (CPU/GPU/MPS)
    ↓
Output saved to temp
    ↓
Displayed in Gradio interface
    ↓
User downloads result
    ↓
Temp files cleaned on exit
```

## Module Dependencies

```
main.py
├── config.config
├── utils.temp_manager
├── utils.device_manager
├── tabs.upscaler_tab
├── tabs.support_tab
└── theme.custom_theme

upscaler_tab.py
├── utils.temp_manager
├── utils.device_manager
├── config.config
├── realesrgan
├── basicsr
├── torch
├── opencv
└── gradio

support_tab.py
├── config.config
└── gradio

custom_theme.py
├── base64
└── gradio.themes
```

## Runtime Directories

These directories are created at runtime and not tracked in git:

- **temp/**: All temporary processing files
  - frames/: Extracted video frames during processing
  - output_frames/: Upscaled video frames
  - output/: Temporary output files
- **.venv/**: Python virtual environment

## Key Technologies

- **Gradio 4.0+**: Web interface framework
- **PyTorch 2.1.0**: Deep learning framework
- **RealESRGAN**: AI upscaling models (x2, x4, anime)
- **BasicSR**: Super-resolution framework
- **OpenCV 4.9**: Image/video processing
- **FFmpeg**: Video encoding/decoding
- **NumPy <2**: Numerical computing (compatibility with torch 2.1.0)
- **Pillow**: Image manipulation

## Performance Considerations

- **Models are lazy-loaded**: Only when first used, not at startup
- **Automatic file detection**: No need to specify if image or video
- **Temp files cleaned automatically**: On app exit
- **Device selection critical**: GPU/MPS significantly faster than CPU
- **Video processing resource-intensive**: Especially at high resolutions
- **First run downloads models**: ~50-150 MB per model from GitHub
- **Background image embedded**: Base64 encoded (~2.2MB) in CSS
- **Parallel frame processing**: Could be added for faster video upscaling

## Application Features

### 100% Free & Open
- No authentication required
- No subscriptions or payments
- No ads or tracking
- Fully self-contained application

### AI Upscaling
- Supports images and videos
- Multiple models (general, anime, 2x, 4x)
- Automatic file type detection
- Progress tracking

### Modular Architecture
- Easy to add new tabs/features
- Separate files for each functionality
- Clean dependency structure
- Reusable utilities (temp manager, device manager)

### Custom Theme
- Amber/Red/Gray color scheme
- Parallax background effect
- Smooth animations
- Responsive design
