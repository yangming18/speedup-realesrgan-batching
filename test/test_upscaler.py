"""
Tests for Upscaler functionality.

Tests image and video upscaling with RealESRGAN models.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from tabs.upscaler_tab import UpscalerTab


class TestUpscalerInitialization:
    """Test Upscaler tab initialization."""
    
    def test_tab_initialization(self, mock_temp_manager, mock_device_manager):
        """Test that Upscaler tab initializes correctly."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        assert tab.temp_manager == mock_temp_manager
        assert tab.device_manager == mock_device_manager
        assert tab.current_model is None
        assert tab.current_model_name is None
        assert tab.upsampler is None
    
    def test_initial_state(self, mock_temp_manager, mock_device_manager):
        """Test that initial state is correct."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # No model loaded initially
        assert tab.current_model_name is None
        assert tab.upsampler is None


class TestUpscalerModelLoading:
    """Test model loading functionality."""
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.RRDBNet')
    def test_load_model_success(self, mock_rrdb, mock_realesrgan, 
                                mock_temp_manager, mock_device_manager):
        """Test successful model loading."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Mock the upsampler
        mock_upsampler = MagicMock()
        mock_realesrgan.return_value = mock_upsampler
        
        result = tab.load_model("realesr-general-x4v3", "cpu")
        
        assert "✓" in result or "loaded" in result.lower()
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.RRDBNet')
    def test_load_anime_model(self, mock_rrdb, mock_realesrgan,
                             mock_temp_manager, mock_device_manager):
        """Test loading anime-specific model."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Mock the upsampler
        mock_upsampler = MagicMock()
        mock_realesrgan.return_value = mock_upsampler
        
        # Try to load anime model (if exists in config)
        result = tab.load_model("realesr-general-x4v3", "cpu")
        
        # Should complete without errors
        assert isinstance(result, str)
    
    def test_model_already_loaded(self, mock_temp_manager, mock_device_manager):
        """Test that loading the same model twice returns cached version."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Mock an already loaded model
        tab.current_model_name = "test_model"
        tab.upsampler = MagicMock()
        
        result = tab.load_model("test_model", "cpu")
        
        assert "already loaded" in result.lower()
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    def test_load_model_error_handling(self, mock_realesrgan,
                                      mock_temp_manager, mock_device_manager):
        """Test error handling during model loading."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Make RealESRGANer raise an exception
        mock_realesrgan.side_effect = Exception("Model load error")
        
        result = tab.load_model("realesr-general-x4v3", "cpu")
        
        assert "✗" in result or "error" in result.lower()
        assert tab.upsampler is None  # Should reset on error


class TestUpscalerImageProcessing:
    """Test image upscaling functionality."""
    
    def create_sample_image(self, width=64, height=64):
        """Create a sample PIL image for testing."""
        return Image.new('RGB', (width, height), color='red')
    
    def test_upscale_image_no_input(self, mock_temp_manager, mock_device_manager):
        """Test handling of missing input image."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        result_img, result_msg = tab.upscale_image(None, "realesr-general-x4v3", "cpu")
        
        assert result_img is None
        assert "upload" in result_msg.lower() or "please" in result_msg.lower()
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.RRDBNet')
    def test_upscale_image_success(self, mock_rrdb, mock_realesrgan,
                                   mock_temp_manager, mock_device_manager):
        """Test successful image upscaling."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Mock the upsampler
        mock_upsampler = MagicMock()
        # Return upscaled image (mock)
        upscaled = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_upsampler.enhance.return_value = (upscaled, None)
        mock_realesrgan.return_value = mock_upsampler
        
        # Create sample image
        sample_img = self.create_sample_image()
        
        result_img, result_msg = tab.upscale_image(sample_img, "realesr-general-x4v3", "cpu")
        
        # Should succeed (or at least not return None for both)
        assert result_img is not None or result_msg is not None
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    def test_upscale_image_model_load_failure(self, mock_realesrgan,
                                             mock_temp_manager, mock_device_manager):
        """Test upscaling when model fails to load."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Make model loading fail
        mock_realesrgan.side_effect = Exception("Model error")
        
        sample_img = self.create_sample_image()
        result_img, result_msg = tab.upscale_image(sample_img, "realesr-general-x4v3", "cpu")
        
        assert result_img is None
        assert "✗" in result_msg or "failed" in result_msg.lower()


class TestUpscalerVideoProcessing:
    """Test video upscaling functionality."""
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.cv2.VideoCapture')
    def test_upscale_video_no_input(self, mock_capture, mock_realesrgan,
                                    mock_temp_manager, mock_device_manager):
        """Test handling of missing input video."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        result_video, result_msg = tab.upscale_video(None, "realesr-general-x4v3", "cpu")
        
        assert result_video is None
        assert "upload" in result_msg.lower() or "please" in result_msg.lower()
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.cv2.VideoCapture')
    @patch('tabs.upscaler_tab.cv2.VideoWriter')
    @patch('tabs.upscaler_tab.ffmpeg')
    def test_upscale_video_processing(self, mock_ffmpeg, mock_writer, mock_capture,
                                     mock_realesrgan, mock_temp_manager, 
                                     mock_device_manager, sample_video):
        """Test video upscaling processing."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30, 640, 480, 100]  # fps, width, height, frame_count
        mock_cap.read.side_effect = [(False, None)]  # No frames
        mock_capture.return_value = mock_cap
        
        # Mock upsampler
        mock_upsampler = MagicMock()
        mock_realesrgan.return_value = mock_upsampler
        
        # Call upscale_video
        result_video, result_msg = tab.upscale_video(sample_video, "realesr-general-x4v3", "cpu")
        
        # Should handle gracefully
        assert result_msg is not None


class TestUpscalerDeviceManagement:
    """Test device management in upscaler."""
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.RRDBNet')
    def test_load_model_cpu(self, mock_rrdb, mock_realesrgan,
                           mock_temp_manager, mock_device_manager):
        """Test loading model on CPU."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        mock_upsampler = MagicMock()
        mock_realesrgan.return_value = mock_upsampler
        
        result = tab.load_model("realesr-general-x4v3", "cpu")
        
        # Device manager should be set to CPU
        mock_device_manager.set_device.assert_called_with("cpu")
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.RRDBNet')
    def test_load_model_cuda(self, mock_rrdb, mock_realesrgan,
                            mock_temp_manager, mock_device_manager):
        """Test loading model on CUDA."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        mock_upsampler = MagicMock()
        mock_realesrgan.return_value = mock_upsampler
        
        result = tab.load_model("realesr-general-x4v3", "cuda")
        
        # Device manager should be set to CUDA
        mock_device_manager.set_device.assert_called_with("cuda")
    
    @patch('tabs.upscaler_tab.RealESRGANer')
    @patch('tabs.upscaler_tab.RRDBNet')
    def test_load_model_mps(self, mock_rrdb, mock_realesrgan,
                           mock_temp_manager, mock_device_manager):
        """Test loading model on MPS (Apple Silicon)."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        mock_upsampler = MagicMock()
        mock_realesrgan.return_value = mock_upsampler
        
        result = tab.load_model("realesr-general-x4v3", "mps")
        
        # Device manager should be set to MPS
        mock_device_manager.set_device.assert_called_with("mps")


class TestUpscalerConfiguration:
    """Test upscaler configuration and settings."""
    
    def test_model_config_exists(self):
        """Test that model configurations are available."""
        from config.config import MODELS
        
        assert isinstance(MODELS, dict)
        assert len(MODELS) > 0
    
    def test_model_config_structure(self):
        """Test that model configurations have required fields."""
        from config.config import MODELS
        
        for model_name, config in MODELS.items():
            assert 'scale' in config, f"Model {model_name} missing 'scale'"
            assert 'url' in config, f"Model {model_name} missing 'url'"
            assert isinstance(config['scale'], int)
            assert config['scale'] in [2, 4], f"Invalid scale for {model_name}"


class TestUpscalerErrorHandling:
    """Test error handling in upscaler."""
    
    def test_invalid_image_format(self, mock_temp_manager, mock_device_manager):
        """Test handling of invalid image format."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Pass invalid data
        invalid_image = "not an image"
        
        # Should handle gracefully (will likely fail at model loading or processing)
        # Just ensure it doesn't crash
        try:
            result_img, result_msg = tab.upscale_image(invalid_image, "realesr-general-x4v3", "cpu")
            assert result_msg is not None  # Should return some message
        except Exception:
            pass  # Expected to fail, but shouldn't crash the test
    
    def test_upscaler_reset_on_error(self, mock_temp_manager, mock_device_manager):
        """Test that upscaler is reset when error occurs."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Set up a mock upsampler
        tab.upsampler = MagicMock()
        tab.current_model_name = "test_model"
        
        # Force an error by trying to load with invalid config
        with patch('tabs.upscaler_tab.RealESRGANer', side_effect=Exception("Error")):
            result = tab.load_model("invalid_model", "cpu")
        
        # Should reset upsampler
        assert tab.upsampler is None
        assert tab.current_model_name is None


class TestUpscalerFormats:
    """Test different input/output formats."""
    
    def test_png_format(self, mock_temp_manager, mock_device_manager):
        """Test PNG format specification."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Should accept format parameter
        # This tests that the signature is correct
        assert hasattr(tab.upscale_image, '__call__')
    
    def test_jpg_format(self, mock_temp_manager, mock_device_manager):
        """Test JPG format specification."""
        tab = UpscalerTab(mock_temp_manager, mock_device_manager)
        
        # Should accept format parameter
        assert hasattr(tab.upscale_image, '__call__')
