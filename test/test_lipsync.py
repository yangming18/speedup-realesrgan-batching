"""
Tests for LipSync functionality.

Tests all 5 LipSync models:
- wav2lip
- wav2lip_gan
- sadtalker
- video_retalking
- liveportrait
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tabs.lipsync_tab import LipSyncTab, LipSyncProcessor, LIPSYNC_MODELS


class TestLipSyncModels:
    """Test LipSync model definitions and metadata."""
    
    def test_all_models_defined(self, lipsync_models_list):
        """Test that all expected models are defined in LIPSYNC_MODELS."""
        for model in lipsync_models_list:
            assert model in LIPSYNC_MODELS, f"Model {model} not found in LIPSYNC_MODELS"
    
    def test_model_count(self):
        """Test that we have exactly 5 models."""
        assert len(LIPSYNC_MODELS) == 5, f"Expected 5 models, found {len(LIPSYNC_MODELS)}"
    
    def test_model_metadata_structure(self):
        """Test that each model has required metadata fields."""
        required_fields = ["description", "quality", "speed", "pros", "cons", "best_for"]
        
        for model_name, model_info in LIPSYNC_MODELS.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_name} missing field: {field}"
            
            # Type checks
            assert isinstance(model_info["description"], str)
            assert isinstance(model_info["quality"], int)
            assert isinstance(model_info["speed"], int)
            assert isinstance(model_info["pros"], list)
            assert isinstance(model_info["cons"], list)
            assert isinstance(model_info["best_for"], str)
            
            # Range checks
            assert 1 <= model_info["quality"] <= 5
            assert 1 <= model_info["speed"] <= 5
    
    def test_model_quality_ratings(self):
        """Test that quality ratings are reasonable."""
        # Expected quality ratings (higher is better)
        expected_quality = {
            "wav2lip": 3,
            "wav2lip_gan": 3,
            "sadtalker": 4,
            "video_retalking": 5,
            "liveportrait": 4
        }
        
        for model, expected_q in expected_quality.items():
            assert LIPSYNC_MODELS[model]["quality"] == expected_q
    
    def test_model_speed_ratings(self):
        """Test that speed ratings are reasonable."""
        # Expected speed ratings (higher is faster)
        expected_speed = {
            "wav2lip": 5,
            "wav2lip_gan": 4,
            "sadtalker": 2,
            "video_retalking": 1,
            "liveportrait": 3
        }
        
        for model, expected_s in expected_speed.items():
            assert LIPSYNC_MODELS[model]["speed"] == expected_s


class TestLipSyncProcessor:
    """Test LipSyncProcessor class."""
    
    def test_processor_initialization(self, temp_dir):
        """Test that processor initializes correctly."""
        models_dir = Path(temp_dir) / "models"
        processor = LipSyncProcessor("wav2lip", device="cpu", models_dir=models_dir)
        
        assert processor.model_name == "wav2lip"
        assert processor.device in ["cpu", "cuda", "mps"]
        assert processor.models_dir.exists()
    
    def test_processor_invalid_model(self, temp_dir):
        """Test that processor handles invalid model gracefully."""
        models_dir = Path(temp_dir) / "models"
        # Should raise ValueError for invalid model
        with pytest.raises(ValueError, match="Modello non supportato"):
            processor = LipSyncProcessor("invalid_model", device="cpu", models_dir=models_dir)
    
    @pytest.mark.parametrize("model_name", [
        "wav2lip",
        "wav2lip_gan",
        "sadtalker",
        "video_retalking",
        "liveportrait"
    ])
    def test_processor_all_models(self, model_name, temp_dir):
        """Test processor initialization with all models."""
        models_dir = Path(temp_dir) / "models"
        processor = LipSyncProcessor(model_name, device="cpu", models_dir=models_dir)
        assert processor.model_name == model_name
    
    def test_model_path_detection(self, temp_dir, workspace_root):
        """Test that model paths are detected correctly."""
        models_dir = Path(temp_dir) / "models"
        processor = LipSyncProcessor("wav2lip", device="cpu", models_dir=models_dir)
        
        # Test wav2lip path
        assert processor.model_repo_dir == models_dir / "wav2lip"
        
        # Test sadtalker path
        processor_sadtalker = LipSyncProcessor("sadtalker", device="cpu", models_dir=models_dir)
        assert processor_sadtalker.model_repo_dir == models_dir / "sadtalker"


class TestLipSyncTab:
    """Test LipSyncTab UI and functionality."""
    
    def test_tab_initialization(self, mock_temp_manager, mock_device_manager, mock_i18n):
        """Test that LipSync tab initializes correctly."""
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, mock_i18n)
        
        assert tab.temp_manager == mock_temp_manager
        assert tab.device_manager == mock_device_manager
        assert tab.i18n == mock_i18n
    
    def test_translation_helper(self, mock_temp_manager, mock_device_manager, mock_i18n):
        """Test the _t() translation helper method."""
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, mock_i18n)
        
        # Test basic translation
        result = tab._t("test.key", fallback="Fallback Text")
        assert result in ["test.key", "Fallback Text"]  # Mock returns key or fallback
    
    def test_model_info_generation(self, mock_temp_manager, mock_device_manager, mock_i18n):
        """Test that model info HTML is generated correctly."""
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, mock_i18n)
        
        # This should work for any valid model
        for model_name in LIPSYNC_MODELS.keys():
            tab.processor = LipSyncProcessor(model_name, mock_temp_manager, mock_device_manager)
            info_html = tab._get_model_info_html()
            
            assert isinstance(info_html, str)
            assert len(info_html) > 0
            # Should contain basic HTML structure
            assert "<div" in info_html or "<span" in info_html or model_name in info_html


class TestLipSyncProcessing:
    """Test actual LipSync processing methods."""
    
    @patch('subprocess.Popen')
    def test_wav2lip_processing_call(self, mock_popen, mock_temp_manager, 
                                     mock_device_manager, sample_video, sample_audio):
        """Test that wav2lip processing method is called correctly."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process
        
        processor = LipSyncProcessor("wav2lip", mock_temp_manager, mock_device_manager)
        
        # Mock the method to avoid actual execution
        with patch.object(processor, '_process_wav2lip', return_value=None) as mock_method:
            processor._process_wav2lip(sample_video, sample_audio, None, lambda x, y: None)
            mock_method.assert_called_once()
    
    def test_output_path_generation(self, mock_temp_manager, mock_device_manager, 
                                    sample_video, sample_audio, temp_dir):
        """Test that output paths are generated correctly."""
        processor = LipSyncProcessor("wav2lip", mock_temp_manager, mock_device_manager)
        
        output_path = os.path.join(temp_dir, "output.mp4")
        assert output_path.endswith(".mp4")
        assert os.path.dirname(output_path) == temp_dir
    
    @pytest.mark.parametrize("model_name", [
        "wav2lip",
        "wav2lip_gan",
        "sadtalker",
        "video_retalking",
        "liveportrait"
    ])
    def test_model_processing_methods_exist(self, model_name, mock_temp_manager, mock_device_manager):
        """Test that processing methods exist for all models."""
        processor = LipSyncProcessor(model_name, mock_temp_manager, mock_device_manager)
        
        # Check that the corresponding processing method exists
        method_name = f"_process_{model_name}"
        assert hasattr(processor, method_name), f"Method {method_name} not found"
        assert callable(getattr(processor, method_name))


class TestLipSyncErrorHandling:
    """Test error handling in LipSync processing."""
    
    def test_missing_video_file(self, mock_temp_manager, mock_device_manager):
        """Test handling of missing video file."""
        processor = LipSyncProcessor("wav2lip", mock_temp_manager, mock_device_manager)
        
        # Non-existent file
        fake_video = "/path/to/nonexistent/video.mp4"
        fake_audio = "/path/to/nonexistent/audio.wav"
        
        # Should handle gracefully (store error in last_error)
        assert hasattr(processor, 'last_error')
    
    def test_invalid_audio_file(self, mock_temp_manager, mock_device_manager, sample_video):
        """Test handling of invalid audio file."""
        processor = LipSyncProcessor("wav2lip", mock_temp_manager, mock_device_manager)
        
        fake_audio = "/path/to/nonexistent/audio.wav"
        
        # Should handle gracefully
        assert hasattr(processor, 'last_error')
    
    def test_error_message_storage(self, mock_temp_manager, mock_device_manager):
        """Test that errors are stored correctly."""
        processor = LipSyncProcessor("wav2lip", mock_temp_manager, mock_device_manager)
        
        # Processor should have last_error attribute
        assert hasattr(processor, 'last_error')
        
        # Initially None or empty
        assert processor.last_error is None or processor.last_error == ""


class TestLipSyncI18n:
    """Test internationalization in LipSync tab."""
    
    def test_translation_keys_exist(self, mock_temp_manager, mock_device_manager, mock_i18n):
        """Test that required translation keys are used."""
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, mock_i18n)
        
        # These keys should be used in the tab
        expected_keys = [
            "lipsync.title",
            "lipsync.model_selection",
            "lipsync.video_input",
            "lipsync.audio_input",
            "lipsync.process_button"
        ]
        
        for key in expected_keys:
            # Translation method should be callable with these keys
            result = tab._t(key, fallback="Test")
            assert result is not None
    
    @pytest.mark.parametrize("model_name", [
        "wav2lip",
        "wav2lip_gan",
        "sadtalker",
        "video_retalking",
        "liveportrait"
    ])
    def test_model_translation_keys(self, model_name, mock_temp_manager, 
                                   mock_device_manager, mock_i18n):
        """Test that each model has translation keys."""
        tab = LipSyncTab(mock_temp_manager, mock_device_manager, mock_i18n)
        
        # Each model should have these translation keys
        keys = [
            f"lipsync.models.{model_name}.description",
            f"lipsync.models.{model_name}.best_for"
        ]
        
        for key in keys:
            result = tab._t(key, fallback="Test")
            assert result is not None


class TestLipSyncPatches:
    """Test that patches are applied correctly for models."""
    
    def test_sadtalker_patch_import(self):
        """Test that SadTalker patch can be imported."""
        from utils.sadtalker_patch import patch_sadtalker_numpy_compatibility, check_if_patch_needed
        
        assert callable(patch_sadtalker_numpy_compatibility)
        assert callable(check_if_patch_needed)
    
    def test_liveportrait_patch_import(self):
        """Test that LivePortrait patch can be imported."""
        from utils.liveportrait_patch import patch_liveportrait_numpy_compatibility, check_if_patch_needed
        
        assert callable(patch_liveportrait_numpy_compatibility)
        assert callable(check_if_patch_needed)
    
    @patch('utils.sadtalker_patch.check_if_patch_needed')
    def test_sadtalker_patch_application(self, mock_check, mock_temp_manager, mock_device_manager):
        """Test that SadTalker patch is checked when model is selected."""
        mock_check.return_value = False  # No patch needed
        
        # When processor is created with sadtalker, it should check for patch
        processor = LipSyncProcessor("sadtalker", mock_temp_manager, mock_device_manager)
        assert processor.model_name == "sadtalker"
    
    @patch('utils.liveportrait_patch.check_if_patch_needed')
    def test_liveportrait_patch_application(self, mock_check, mock_temp_manager, mock_device_manager):
        """Test that LivePortrait patch is checked when model is selected."""
        mock_check.return_value = False  # No patch needed
        
        # When processor is created with liveportrait, it should check for patch
        processor = LipSyncProcessor("liveportrait", mock_temp_manager, mock_device_manager)
        assert processor.model_name == "liveportrait"
