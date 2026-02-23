"""
Pytest configuration and shared fixtures for Video Editor tests.
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_video(temp_dir):
    """Create a sample video file path (mock, not real video)."""
    video_path = os.path.join(temp_dir, "sample_video.mp4")
    # Create empty file for testing
    Path(video_path).touch()
    return video_path


@pytest.fixture
def sample_audio(temp_dir):
    """Create a sample audio file path (mock, not real audio)."""
    audio_path = os.path.join(temp_dir, "sample_audio.wav")
    # Create empty file for testing
    Path(audio_path).touch()
    return audio_path


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample image file path (mock, not real image)."""
    image_path = os.path.join(temp_dir, "sample_image.jpg")
    # Create empty file for testing
    Path(image_path).touch()
    return image_path


@pytest.fixture
def mock_temp_manager():
    """Mock TempManager for testing."""
    mock_manager = Mock()
    # Use Path object to match real TempManager API
    temp_path = Path(tempfile.gettempdir()) / "video_editor_test"
    mock_manager.temp_dir = temp_path
    mock_manager.frames_dir = temp_path / "frames"
    mock_manager.output_dir = temp_path / "output"
    mock_manager.get_temp_file_path.return_value = temp_path / "test_temp.mp4"
    mock_manager.get_frames_dir.return_value = temp_path / "frames"
    mock_manager.get_output_dir.return_value = temp_path / "output"
    mock_manager.create_temp_subdir.return_value = temp_path / "subdir"
    mock_manager.initialize.return_value = None
    mock_manager.cleanup.return_value = None
    return mock_manager


@pytest.fixture
def mock_device_manager():
    """Mock DeviceManager for testing."""
    mock_manager = Mock()
    # Match real DeviceManager API
    mock_manager.current_device = "CPU"
    mock_manager.available_devices = ["CPU"]
    mock_manager.get_available_devices.return_value = ["CPU"]
    mock_manager.set_device.return_value = True
    mock_manager.get_torch_device.return_value = Mock()  # Mock torch.device
    return mock_manager


@pytest.fixture
def mock_i18n():
    """Mock I18n system for testing."""
    mock_i18n_obj = Mock()
    # Match real I18nManager API
    mock_i18n_obj.current_lang = "en"
    mock_i18n_obj.locales_dir = Path(project_root) / "locales"
    mock_i18n_obj.translations = {}
    mock_i18n_obj.AVAILABLE_LANGUAGES = {"en": "English", "it": "Italiano"}
    
    def mock_translate(key, **kwargs):
        # Simple translation mock that returns the key
        return key
    
    mock_i18n_obj.t = mock_translate
    mock_i18n_obj.load_language.return_value = True
    mock_i18n_obj.get_current_language.return_value = "en"
    
    return mock_i18n_obj


@pytest.fixture
def mock_gradio_progress():
    """Mock Gradio progress function."""
    def progress_fn(progress_value, desc=None):
        pass
    return progress_fn


@pytest.fixture
def lipsync_models_list():
    """List of all available LipSync models."""
    return ["wav2lip", "wav2lip_gan", "sadtalker", "video_retalking", "liveportrait"]


@pytest.fixture
def mock_subprocess_success(monkeypatch):
    """Mock subprocess to simulate successful execution."""
    mock_popen = MagicMock()
    mock_popen.return_value.poll.return_value = 0
    mock_popen.return_value.returncode = 0
    mock_popen.return_value.stdout = None
    mock_popen.return_value.stderr = None
    mock_popen.return_value.communicate.return_value = (b"Success", b"")
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    return mock_popen


@pytest.fixture
def mock_subprocess_failure(monkeypatch):
    """Mock subprocess to simulate failed execution."""
    mock_popen = MagicMock()
    mock_popen.return_value.poll.return_value = 1
    mock_popen.return_value.returncode = 1
    mock_popen.return_value.stdout = None
    mock_popen.return_value.stderr = None
    mock_popen.return_value.communicate.return_value = (b"", b"Error occurred")
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    return mock_popen


@pytest.fixture(autouse=True)
def reset_sys_path():
    """Reset sys.path after each test to avoid pollution."""
    original_path = sys.path.copy()
    yield
    sys.path = original_path


@pytest.fixture
def workspace_root():
    """Return the workspace root directory."""
    return project_root
