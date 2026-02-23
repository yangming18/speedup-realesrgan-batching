"""
Integration tests for Video Editor application.

Tests the complete application flow:
- Application initialization
- Tab integration
- End-to-end workflows
- System integration
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Ensure main module can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestApplicationInitialization:
    """Test complete application initialization."""
    
    @patch('gradio.Blocks')
    def test_main_module_imports(self, mock_blocks):
        """Test that main module can be imported."""
        try:
            from main import VideoEditorApp
            assert VideoEditorApp is not None
        except Exception as e:
            pytest.fail(f"Failed to import main module: {str(e)}")
    
    @patch('gradio.Blocks')
    def test_app_initialization(self, mock_blocks):
        """Test that VideoEditorApp can be initialized."""
        from main import VideoEditorApp
        
        try:
            app = VideoEditorApp()
            assert app is not None
        except Exception as e:
            pytest.fail(f"Failed to initialize app: {str(e)}")
    
    @patch('gradio.Blocks')
    def test_app_has_required_tabs(self, mock_blocks):
        """Test that app has all required tabs."""
        from main import VideoEditorApp
        
        app = VideoEditorApp()
        
        # Should have these tabs
        assert hasattr(app, 'lipsync_tab')
        assert hasattr(app, 'upscaler_tab')
    
    @patch('gradio.Blocks')
    def test_app_has_managers(self, mock_blocks):
        """Test that app initializes managers."""
        from main import VideoEditorApp
        
        app = VideoEditorApp()
        
        # Should have these managers
        assert hasattr(app, 'temp_manager') or hasattr(app, 'device_manager')


class TestTabsIntegration:
    """Test integration between tabs."""
    
    def test_lipsync_tab_imports(self):
        """Test that LipSync tab can be imported."""
        from tabs.lipsync_tab import LipSyncTab
        assert LipSyncTab is not None
    
    def test_upscaler_tab_imports(self):
        """Test that Upscaler tab can be imported."""
        from tabs.upscaler_tab import UpscalerTab
        assert UpscalerTab is not None
    
    def test_settings_tab_imports(self):
        """Test that Settings tab can be imported."""
        from tabs.settings_tab import SettingsTab
        assert SettingsTab is not None
    
    def test_support_tab_imports(self):
        """Test that Support tab can be imported."""
        from tabs.support_tab import SupportTab
        assert SupportTab is not None
    
    def test_all_tabs_initialize_together(self, mock_temp_manager, 
                                          mock_device_manager, mock_i18n):
        """Test that all tabs can be initialized together."""
        from tabs.lipsync_tab import LipSyncTab
        from tabs.upscaler_tab import UpscalerTab
        from tabs.settings_tab import SettingsTab
        from tabs.support_tab import SupportTab
        
        # Initialize all tabs
        lipsync = LipSyncTab(mock_temp_manager, mock_device_manager, mock_i18n)
        upscaler = UpscalerTab(mock_temp_manager, mock_device_manager)
        settings = SettingsTab(mock_i18n)
        support = SupportTab()
        
        # All should exist
        assert lipsync is not None
        assert upscaler is not None
        assert settings is not None
        assert support is not None


class TestUtilsIntegration:
    """Test utils integration with main application."""
    
    @patch('gradio.Blocks')
    def test_utils_available_to_app(self, mock_blocks):
        """Test that utils are available to application."""
        from main import VideoEditorApp
        from utils.device_manager import DeviceManager
        from utils.temp_manager import TempManager
        from utils.i18n import I18n
        
        # All utils should be importable
        assert DeviceManager is not None
        assert TempManager is not None
        assert I18n is not None
    
    def test_device_manager_integration(self):
        """Test DeviceManager integration."""
        from utils.device_manager import DeviceManager
        
        dm = DeviceManager()
        device = dm.get_device()
        
        # Should return valid device
        assert device in ['cpu', 'cuda', 'mps']
    
    def test_i18n_integration_with_tabs(self, mock_temp_manager, mock_device_manager):
        """Test I18n integration with tabs."""
        from utils.i18n import I18n
        from tabs.lipsync_tab import LipSyncTab
        
        i18n = I18n()
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, i18n)
        
        # Tab should have access to i18n
        assert tab.i18n == i18n


class TestConfigIntegration:
    """Test configuration integration."""
    
    def test_config_module_imports(self):
        """Test that config module can be imported."""
        from config.config import MODELS
        
        assert MODELS is not None
        assert isinstance(MODELS, dict)
    
    def test_config_has_upscaler_models(self):
        """Test that config has upscaler model definitions."""
        from config.config import MODELS
        
        # Should have at least one model
        assert len(MODELS) > 0
        
        # Each model should have required fields
        for model_name, model_config in MODELS.items():
            assert 'scale' in model_config
            assert 'url' in model_config
    
    def test_config_user_settings(self):
        """Test user settings configuration."""
        from config.config import USER_SETTINGS_PATH
        
        # User settings path should be defined
        assert USER_SETTINGS_PATH is not None


class TestThemeIntegration:
    """Test theme integration."""
    
    def test_theme_imports(self):
        """Test that custom theme can be imported."""
        try:
            from theme.custom_theme import create_custom_theme
            assert callable(create_custom_theme)
        except ImportError:
            pytest.skip("Custom theme not implemented")
    
    def test_theme_with_app(self):
        """Test that theme integrates with app."""
        try:
            from theme.custom_theme import create_custom_theme
            
            theme = create_custom_theme()
            assert theme is not None
        except ImportError:
            pytest.skip("Custom theme not implemented")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_lipsync_workflow_initialization(self, mock_temp_manager, 
                                            mock_device_manager, mock_i18n):
        """Test LipSync workflow can be initialized."""
        from tabs.lipsync_tab import LipSyncTab, LipSyncProcessor
        
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, mock_i18n)
        
        # Should be able to create processor
        processor = LipSyncProcessor("wav2lip", mock_temp_manager, mock_device_manager)
        
        assert processor is not None
        assert processor.model_name == "wav2lip"
    
    def test_upscaler_workflow_initialization(self, mock_temp_manager, mock_device_manager):
        """Test Upscaler workflow can be initialized."""
        from tabs.upscaler_tab import UpscalerTab
        
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Should have upscale methods
        assert hasattr(tab, 'upscale_image')
        assert hasattr(tab, 'upscale_video')
        assert callable(tab.upscale_image)
        assert callable(tab.upscale_video)
    
    @pytest.mark.parametrize("model_name", [
        "wav2lip",
        "wav2lip_gan",
        "sadtalker",
        "video_retalking",
        "liveportrait"
    ])
    def test_all_lipsync_models_workflow(self, model_name, mock_temp_manager, 
                                        mock_device_manager):
        """Test workflow for each LipSync model."""
        from tabs.lipsync_tab import LipSyncProcessor
        
        processor = LipSyncProcessor(model_name, mock_temp_manager, mock_device_manager)
        
        assert processor.model_name == model_name
        
        # Should have processing method
        method_name = f"_process_{model_name}"
        assert hasattr(processor, method_name)


class TestErrorHandlingIntegration:
    """Test error handling across the application."""
    
    def test_missing_model_handling(self, mock_temp_manager, mock_device_manager):
        """Test handling of missing models."""
        from tabs.lipsync_tab import LipSyncProcessor
        
        # Create processor with non-existent model
        processor = LipSyncProcessor("nonexistent_model", mock_temp_manager, mock_device_manager)
        
        # Should not crash
        assert processor is not None
    
    def test_invalid_file_handling(self, mock_temp_manager, mock_device_manager):
        """Test handling of invalid input files."""
        from tabs.upscaler_tab import UpscalerTab
        
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Try to process None
        result_img, result_msg = tab.upscale_image(None, "realesr-general-x4v3", "cpu")
        
        # Should handle gracefully
        assert result_img is None
        assert result_msg is not None
    
    def test_device_fallback(self):
        """Test device fallback when GPU not available."""
        from utils.device_manager import DeviceManager
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                dm = DeviceManager()
                
                # Should fallback to CPU
                assert dm.get_device() == 'cpu'


class TestPatchSystemIntegration:
    """Test patch system integration."""
    
    def test_patches_available_to_lipsync(self, mock_temp_manager, mock_device_manager):
        """Test that patches are available when using LipSync."""
        from tabs.lipsync_tab import LipSyncProcessor
        from utils.sadtalker_patch import check_if_patch_needed
        from utils.liveportrait_patch import check_if_patch_needed as check_liveportrait
        
        # Patches should be importable
        assert callable(check_if_patch_needed)
        assert callable(check_liveportrait)
        
        # Processors should work
        sadtalker = LipSyncProcessor("sadtalker", mock_temp_manager, mock_device_manager)
        liveportrait = LipSyncProcessor("liveportrait", mock_temp_manager, mock_device_manager)
        
        assert sadtalker is not None
        assert liveportrait is not None
    
    def test_patch_auto_application(self, mock_temp_manager, mock_device_manager):
        """Test that patches are auto-applied when needed."""
        from tabs.lipsync_tab import LipSyncProcessor
        
        # When creating processor for patched models, should check patches
        processor = LipSyncProcessor("sadtalker", mock_temp_manager, mock_device_manager)
        
        # Should complete without errors
        assert processor.model_name == "sadtalker"


class TestLanguageSupport:
    """Test multi-language support integration."""
    
    def test_english_language(self):
        """Test English language support."""
        from utils.i18n import I18n
        
        i18n = I18n()
        i18n.set_language('en')
        
        # Should work with English
        result = i18n.t('lipsync.title')
        assert isinstance(result, str)
    
    def test_italian_language(self):
        """Test Italian language support."""
        from utils.i18n import I18n
        
        i18n = I18n()
        i18n.set_language('it')
        
        # Should work with Italian
        result = i18n.t('lipsync.title')
        assert isinstance(result, str)
    
    def test_language_switching_in_tabs(self, mock_temp_manager, mock_device_manager):
        """Test language switching in tabs."""
        from utils.i18n import I18n
        from tabs.lipsync_tab import LipSyncTab
        
        i18n = I18n()
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, i18n)
        
        # Switch to Italian
        i18n.set_language('it')
        result_it = tab._t('lipsync.title', fallback="Test")
        
        # Switch to English
        i18n.set_language('en')
        result_en = tab._t('lipsync.title', fallback="Test")
        
        # Both should return strings
        assert isinstance(result_it, str)
        assert isinstance(result_en, str)


class TestDependencies:
    """Test that all required dependencies are available."""
    
    def test_pytorch_import(self):
        """Test that PyTorch can be imported."""
        import torch
        assert torch is not None
    
    def test_gradio_import(self):
        """Test that Gradio can be imported."""
        import gradio as gr
        assert gr is not None
    
    def test_opencv_import(self):
        """Test that OpenCV can be imported."""
        import cv2
        assert cv2 is not None
    
    def test_numpy_import(self):
        """Test that NumPy can be imported."""
        import numpy as np
        assert np is not None
    
    def test_pil_import(self):
        """Test that PIL can be imported."""
        from PIL import Image
        assert Image is not None
    
    def test_ffmpeg_import(self):
        """Test that ffmpeg-python can be imported."""
        import ffmpeg
        assert ffmpeg is not None


class TestSystemCompatibility:
    """Test system compatibility."""
    
    def test_python_version(self):
        """Test Python version is compatible."""
        import sys
        
        # Should be Python 3.8+
        assert sys.version_info >= (3, 8)
    
    def test_file_paths_handling(self, temp_dir):
        """Test that file paths are handled correctly."""
        from utils.temp_manager import TempManager
        
        tm = TempManager()
        temp_path = tm.get_temp_path("test.mp4")
        
        # Should handle paths correctly
        assert isinstance(temp_path, str)
        assert temp_path.endswith(".mp4")
    
    def test_workspace_structure(self, workspace_root):
        """Test that workspace has required structure."""
        # Required directories
        required_dirs = [
            'tabs',
            'utils',
            'config',
            'locales',
            'models',
            'theme'
        ]
        
        for dir_name in required_dirs:
            dir_path = workspace_root / dir_name
            assert dir_path.exists(), f"Missing directory: {dir_name}"
    
    def test_required_files_exist(self, workspace_root):
        """Test that required files exist."""
        required_files = [
            'main.py',
            'requirements.txt',
            'README.md',
            'config/config.py',
            'locales/en.json',
            'locales/it.json'
        ]
        
        for file_path in required_files:
            full_path = workspace_root / file_path
            assert full_path.exists(), f"Missing file: {file_path}"
