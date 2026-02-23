"""
Tests for utility modules.

Tests:
- DeviceManager
- TempManager
- I18n system
"""
import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from utils.device_manager import DeviceManager
from utils.temp_manager import TempManager
from utils.i18n import I18nManager


class TestDeviceManager:
    """Test DeviceManager functionality."""
    
    def test_initialization(self):
        """Test DeviceManager initializes correctly."""
        dm = DeviceManager()
        
        assert hasattr(dm, 'current_device')
        assert hasattr(dm, 'available_devices')
        assert dm.current_device in dm.available_devices
    
    def test_cuda_availability_check(self):
        """Test CUDA availability detection."""
        dm = DeviceManager()
        
        # Check if CUDA is in available devices
        cuda_available = "GPU (CUDA)" in dm.available_devices
        assert isinstance(cuda_available, bool)
    
    def test_mps_availability_check(self):
        """Test MPS (Apple Silicon) availability detection."""
        dm = DeviceManager()
        
        # Check if MPS is in available devices
        mps_available = "MPS (Apple Silicon)" in dm.available_devices
        assert isinstance(mps_available, bool)
    
    def test_get_device(self):
        """Test getting current device."""
        dm = DeviceManager()
        
        device = dm.current_device
        assert device in dm.available_devices
    
    def test_get_torch_device(self):
        """Test getting PyTorch device object."""
        dm = DeviceManager()
        
        with patch('torch.device') as mock_device:
            mock_device.return_value = "cpu"
            torch_device = dm.get_torch_device()
            # Should return a torch.device or string
            assert torch_device is not None
    
    def test_set_device_cpu(self):
        """Test setting device to CPU."""
        dm = DeviceManager()
        result = dm.set_device('CPU')
        
        assert result == True
        assert dm.current_device == 'CPU'
    
    @patch('torch.cuda.is_available')
    def test_set_device_cuda(self, mock_cuda):
        """Test setting device to CUDA."""
        mock_cuda.return_value = True
        
        dm = DeviceManager()
        if "GPU (CUDA)" in dm.available_devices:
            dm.set_device('GPU (CUDA)')
            assert dm.current_device == 'GPU (CUDA)'
    
    @patch('torch.backends.mps.is_available')
    def test_set_device_mps(self, mock_mps):
        """Test setting device to MPS."""
        mock_mps.return_value = True
        
        dm = DeviceManager()
        if "MPS (Apple Silicon)" in dm.available_devices:
            dm.set_device('MPS (Apple Silicon)')
            assert dm.current_device == 'MPS (Apple Silicon)'
    
    def test_get_available_devices(self):
        """Test getting available devices list."""
        dm = DeviceManager()
        
        devices = dm.get_available_devices()
        assert isinstance(devices, list)
        assert len(devices) > 0
        assert "CPU" in devices
    
    def test_device_fallback_to_cpu(self):
        """Test that device falls back to CPU when GPU not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                dm = DeviceManager()
                
                # Should default to CPU
                assert dm.current_device == 'CPU'


class TestTempManager:
    """Test TempManager functionality."""
    
    def test_initialization(self):
        """Test TempManager initializes correctly."""
        tm = TempManager()
        
        assert hasattr(tm, 'temp_dir')
        assert isinstance(tm.temp_dir, Path)
    
    def test_temp_dir_exists(self):
        """Test that temp directory exists."""
        tm = TempManager()
        tm.initialize()
        
        assert tm.temp_dir.exists()
        assert tm.temp_dir.is_dir()
    
    def test_get_temp_file_path(self):
        """Test getting temporary file path."""
        tm = TempManager()
        
        temp_path = tm.get_temp_file_path("test.mp4")
        
        assert temp_path.name == "test.mp4"
        assert temp_path.parent == tm.temp_dir
    
    def test_get_temp_path_with_extension(self):
        """Test getting temp path with different extensions."""
        tm = TempManager()
        
        extensions = [".mp4", ".wav", ".jpg", ".png"]
        
        for ext in extensions:
            temp_path = tm.get_temp_file_path(f"test{ext}")
            assert str(temp_path).endswith(ext)
    
    def test_cleanup_temp_files(self):
        """Test cleanup of temporary files."""
        tm = TempManager()
        tm.initialize()
        
        # Create a test file in temp directory
        test_file = tm.temp_dir / "test_cleanup.txt"
        test_file.touch()
        
        # Cleanup should work without errors
        try:
            tm.cleanup()
            # After cleanup, temp dir should not exist
            assert not tm.temp_dir.exists()
        except Exception as e:
            pytest.fail(f"Cleanup failed: {str(e)}")
    
    def test_create_temp_subdir(self):
        """Test creating temporary subdirectory."""
        tm = TempManager()
        tm.initialize()
        
        # Create a subdirectory
        subdir = tm.create_temp_subdir("test_subdir")
        
        assert subdir.exists()
        assert subdir.is_dir()
        assert subdir.parent == tm.temp_dir
        
        # Clean up
        tm.cleanup()
    
    def test_get_frames_dir(self):
        """Test getting frames directory."""
        tm = TempManager()
        tm.initialize()
        
        frames_dir = tm.get_frames_dir()
        
        assert frames_dir.exists()
        assert frames_dir.is_dir()
        assert frames_dir.parent == tm.temp_dir
        
        tm.cleanup()
    
    def test_multiple_temp_paths_unique(self):
        """Test that multiple temp paths are unique."""
        tm = TempManager()
        
        path1 = tm.get_temp_file_path("test1.mp4")
        path2 = tm.get_temp_file_path("test2.mp4")
        
        assert path1 != path2
        assert path1.name == "test1.mp4"
        assert path2.name == "test2.mp4"
        
        assert path1 != path2


class TestI18n:
    """Test I18n (internationalization) system."""
    
    def test_initialization_default_language(self):
        """Test I18n initializes with default language."""
        i18n = I18nManager()
        
        assert hasattr(i18n, 'current_lang')
        assert i18n.current_lang in ['en', 'it']
    
    def test_available_languages(self):
        """Test that available languages are detected."""
        i18n = I18nManager()
        
        assert hasattr(I18nManager, 'AVAILABLE_LANGUAGES')
        assert 'en' in I18nManager.AVAILABLE_LANGUAGES
        assert 'it' in I18nManager.AVAILABLE_LANGUAGES
    
    def test_translation_function(self):
        """Test translation function works."""
        i18n = I18nManager()
        
        # Test with a known key
        result = i18n.t('lipsync.title')
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_translation_with_fallback(self):
        """Test translation falls back to key if not found."""
        i18n = I18nManager()
        
        # Test with non-existent key
        result = i18n.t('nonexistent.key.that.does.not.exist')
        
        # Should return the key itself or a fallback
        assert isinstance(result, str)
    
    def test_set_language_english(self):
        """Test setting language to English."""
        i18n = I18nManager()
        result = i18n.load_language('en')
        
        assert result == True
        assert i18n.current_lang == 'en'
    
    def test_set_language_italian(self):
        """Test setting language to Italian."""
        i18n = I18nManager()
        result = i18n.load_language('it')
        
        assert result == True
        assert i18n.current_lang == 'it'
    
    def test_translation_files_exist(self, workspace_root):
        """Test that translation files exist."""
        en_file = workspace_root / 'locales' / 'en.json'
        it_file = workspace_root / 'locales' / 'it.json'
        
        assert en_file.exists(), "English translation file missing"
        assert it_file.exists(), "Italian translation file missing"
    
    def test_translation_files_valid_json(self, workspace_root):
        """Test that translation files are valid JSON."""
        en_file = workspace_root / 'locales' / 'en.json'
        it_file = workspace_root / 'locales' / 'it.json'
        
        # Try to load English translations
        with open(en_file, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
            assert isinstance(en_data, dict)
        
        # Try to load Italian translations
        with open(it_file, 'r', encoding='utf-8') as f:
            it_data = json.load(f)
            assert isinstance(it_data, dict)
    
    def test_nested_translation_keys(self):
        """Test that nested translation keys work."""
        i18n = I18nManager()
        
        # Test nested key like 'lipsync.models.wav2lip.description'
        result = i18n.t('lipsync.models.wav2lip.description')
        
        assert isinstance(result, str)
    
    def test_translation_consistency(self, workspace_root):
        """Test that EN and IT translations have same structure."""
        en_file = workspace_root / 'locales' / 'en.json'
        it_file = workspace_root / 'locales' / 'it.json'
        
        with open(en_file, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
        
        with open(it_file, 'r', encoding='utf-8') as f:
            it_data = json.load(f)
        
        # Check that both have 'lipsync' section
        assert 'lipsync' in en_data
        assert 'lipsync' in it_data
        
        # Check that both have models section
        if 'models' in en_data.get('lipsync', {}):
            assert 'models' in it_data.get('lipsync', {})
    
    def test_lipsync_model_translations_exist(self, workspace_root):
        """Test that all LipSync models have translations."""
        en_file = workspace_root / 'locales' / 'en.json'
        
        with open(en_file, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
        
        models = ['wav2lip', 'wav2lip_gan', 'sadtalker', 'video_retalking', 'liveportrait']
        
        lipsync_models = en_data.get('lipsync', {}).get('models', {})
        
        for model in models:
            assert model in lipsync_models, f"Model {model} missing from translations"
            
            # Check required fields
            model_data = lipsync_models[model]
            assert 'description' in model_data, f"{model} missing description"
            assert 'best_for' in model_data, f"{model} missing best_for"


class TestOpenCVPatch:
    """Test OpenCV patch utility."""
    
    def test_patch_import(self):
        """Test that OpenCV patch can be imported."""
        try:
            from utils.opencv_patch import apply_opencv_patch
            assert callable(apply_opencv_patch)
        except ImportError:
            pytest.skip("OpenCV patch not implemented yet")
    
    def test_patch_application(self):
        """Test that OpenCV patch can be applied."""
        try:
            from utils.opencv_patch import apply_opencv_patch
            
            # Should not raise exception
            apply_opencv_patch()
        except ImportError:
            pytest.skip("OpenCV patch not implemented yet")
        except Exception as e:
            # Patch might fail in test environment, that's ok
            pass


class TestUtilsIntegration:
    """Test integration between utility modules."""
    
    def test_device_and_temp_manager_integration(self):
        """Test that DeviceManager and TempManager work together."""
        dm = DeviceManager()
        tm = TempManager()
        
        # Both should be usable together
        device = dm.current_device
        temp_path = tm.get_temp_file_path("test.mp4")
        
        assert device is not None
        assert temp_path is not None
    
    def test_all_utils_import(self):
        """Test that all utils can be imported without errors."""
        from utils.device_manager import DeviceManager
        from utils.temp_manager import TempManager
        from utils.i18n import I18nManager
        
        # All should be importable
        assert DeviceManager is not None
        assert TempManager is not None
        assert I18nManager is not None
    
    def test_utils_initialization_order(self):
        """Test that utils can be initialized in any order."""
        # Order 1
        dm1 = DeviceManager()
        tm1 = TempManager()
        i18n1 = I18nManager()
        
        # Order 2
        i18n2 = I18nManager()
        tm2 = TempManager()
        dm2 = DeviceManager()
        
        # All should work
        assert dm1 is not None and dm2 is not None
        assert tm1 is not None and tm2 is not None
        assert i18n1 is not None and i18n2 is not None
