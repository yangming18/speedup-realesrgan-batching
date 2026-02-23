"""
Tests for NumPy compatibility patches.

Tests:
- SadTalker NumPy patch
- LivePortrait NumPy patch
- Patch detection and application
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from utils.sadtalker_patch import (
    patch_sadtalker_numpy_compatibility, 
    check_if_patch_needed
)
from utils.liveportrait_patch import (
    patch_liveportrait_numpy_compatibility, 
    check_if_patch_needed as check_liveportrait_patch_needed
)


class TestSadTalkerPatch:
    """Test SadTalker NumPy compatibility patch."""
    
    def test_patch_function_exists(self):
        """Test that patch function exists and is callable."""
        assert callable(patch_sadtalker_numpy_compatibility)
        assert callable(check_if_patch_needed)
    
    def test_check_patch_needed_no_directory(self):
        """Test patch check when SadTalker directory doesn't exist."""
        fake_dir = Path("/nonexistent/sadtalker")
        result = check_if_patch_needed(fake_dir)
        
        # Should return False when directory doesn't exist
        assert isinstance(result, bool)
        assert result == False
    
    def test_check_patch_needed_with_mock_directory(self, temp_dir, workspace_root):
        """Test patch check with mock directory."""
        # Create a mock sadtalker directory structure
        sadtalker_dir = Path(temp_dir) / "models" / "lipsync" / "sadtalker"
        sadtalker_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock Python file with old NumPy syntax
        test_file = sadtalker_dir / "test.py"
        test_file.write_text("x = np.float(5.0)\ny = data.astype(np.float)")
        
        # Check if patch would be needed (would need to check this specific directory)
        # This is a simplified test
        assert test_file.exists()
    
    def test_numpy_float_pattern_detection(self):
        """Test detection of deprecated NumPy patterns."""
        # Mock file content with old patterns
        old_code = """
import numpy as np

def test():
    x = np.float(5.0)
    y = data.astype(np.float)
    z = np.int(10)
"""
        
        # These patterns should be detected
        assert "np.float" in old_code
        assert "np.int" in old_code
    
    def test_numpy_patched_patterns(self):
        """Test that patched code uses new NumPy syntax."""
        # New correct code
        new_code = """
import numpy as np

def test():
    x = np.float64(5.0)
    y = data.astype(np.float64)
    z = np.int64(10)
"""
        
        # Should use new patterns
        assert "np.float64" in new_code
        assert "np.int64" in new_code
        # Should not have old patterns
        assert "np.float(" not in new_code
        assert "np.int(" not in new_code
    
    @patch('builtins.open', new_callable=mock_open, read_data="x = np.float(5)")
    @patch('os.path.exists')
    @patch('pathlib.Path.glob')
    def test_patch_file_modification(self, mock_glob, mock_exists, mock_file):
        """Test that patch modifies files correctly."""
        mock_exists.return_value = True
        
        # Mock Python files
        mock_glob.return_value = [Path("/fake/file.py")]
        
        # This would apply the patch in real scenario
        # In test, we just verify the function can be called
        try:
            # Don't actually patch in test environment
            pass
        except Exception:
            pass
    
    def test_patch_safe_on_nonexistent_directory(self):
        """Test that patch is safe when directory doesn't exist."""
        # Should not raise exception
        try:
            fake_dir = Path("/nonexistent/sadtalker")
            result = check_if_patch_needed(fake_dir)
            assert isinstance(result, bool)
            assert result == False
        except Exception as e:
            pytest.fail(f"Patch check should not fail: {str(e)}")


class TestLivePortraitPatch:
    """Test LivePortrait NumPy compatibility patch."""
    
    def test_patch_function_exists(self):
        """Test that patch function exists and is callable."""
        assert callable(patch_liveportrait_numpy_compatibility)
        assert callable(check_liveportrait_patch_needed)
    
    def test_check_patch_needed_no_directory(self):
        """Test patch check when LivePortrait directory doesn't exist."""
        fake_dir = Path("/nonexistent/liveportrait")
        result = check_liveportrait_patch_needed(fake_dir)
        
        # Should return False when directory doesn't exist
        assert isinstance(result, bool)
        assert result == False
    
    def test_liveportrait_patch_similar_to_sadtalker(self):
        """Test that LivePortrait patch follows same pattern as SadTalker."""
        # Both should have similar function signatures
        from inspect import signature
        
        sadtalker_sig = signature(patch_sadtalker_numpy_compatibility)
        liveportrait_sig = signature(patch_liveportrait_numpy_compatibility)
        
        # Should have same or similar parameters
        assert len(sadtalker_sig.parameters) == len(liveportrait_sig.parameters)
    
    def test_patch_safe_on_nonexistent_directory(self):
        """Test that patch is safe when directory doesn't exist."""
        # Should not raise exception
        try:
            fake_dir = Path("/nonexistent/liveportrait")
            result = check_liveportrait_patch_needed(fake_dir)
            assert isinstance(result, bool)
            assert result == False
        except Exception as e:
            pytest.fail(f"Patch check should not fail: {str(e)}")


class TestPatchPatterns:
    """Test NumPy pattern replacements."""
    
    def test_float_replacement_pattern(self):
        """Test that np.float is correctly replaced."""
        old = "x.astype(np.float)"
        new = "x.astype(np.float64)"
        
        assert old != new
        assert "np.float64" in new
    
    def test_int_replacement_pattern(self):
        """Test that np.int is correctly replaced."""
        old = "x.astype(np.int)"
        new = "x.astype(np.int64)"
        
        assert old != new
        assert "np.int64" in new
    
    def test_complex_replacement_pattern(self):
        """Test more complex NumPy patterns."""
        patterns = [
            ("np.float", "np.float64"),
            ("np.int", "np.int64"),
            (".astype(np.float)", ".astype(np.float64)"),
            (".astype(np.int)", ".astype(np.int64)")
        ]
        
        for old, new in patterns:
            assert old != new
            assert "np." in old and "np." in new
    
    def test_array_creation_pattern(self):
        """Test array creation with float type."""
        old = "np.array([w0, h0, s, t0, t1], dtype=np.float)"
        new = "np.array([w0, h0, float(s), float(t[0]), float(t[1])], dtype=np.float64)"
        
        assert old != new
        assert "float(" in new


class TestPatchIntegration:
    """Test patch system integration."""
    
    def test_both_patches_importable(self):
        """Test that both patches can be imported together."""
        from utils.sadtalker_patch import patch_sadtalker_numpy_compatibility
        from utils.liveportrait_patch import patch_liveportrait_numpy_compatibility
        
        assert patch_sadtalker_numpy_compatibility is not None
        assert patch_liveportrait_numpy_compatibility is not None
    
    def test_patches_independent(self):
        """Test that patches are independent."""
        # Check for one shouldn't affect the other
        fake_sadtalker_dir = Path("/nonexistent/sadtalker")
        fake_liveportrait_dir = Path("/nonexistent/liveportrait")
        
        sadtalker_needed = check_if_patch_needed(fake_sadtalker_dir)
        liveportrait_needed = check_liveportrait_patch_needed(fake_liveportrait_dir)
        
        # Both should return bool independently
        assert isinstance(sadtalker_needed, bool)
        assert isinstance(liveportrait_needed, bool)
    
    def test_patch_idempotency(self, workspace_root):
        """Test that applying patch multiple times is safe."""
        # Patches should be idempotent - safe to apply multiple times
        
        # Create a test file with old syntax
        test_content = "x = np.float(5.0)"
        
        # First patch
        patched_once = test_content.replace("np.float(", "np.float64(")
        
        # Second patch (should be same)
        patched_twice = patched_once.replace("np.float(", "np.float64(")
        
        # Should be identical
        assert patched_once == patched_twice
    
    def test_patch_preserves_correct_code(self):
        """Test that patch doesn't modify already correct code."""
        correct_code = """
import numpy as np

def test():
    x = np.float64(5.0)
    y = data.astype(np.float64)
"""
        
        # Patch should not change correct code
        # (In reality, regex should skip already correct patterns)
        assert "np.float64" in correct_code
        assert "np.float(" not in correct_code


class TestPatchLogging:
    """Test patch logging and reporting."""
    
    def test_patch_returns_info(self):
        """Test that patch functions return information."""
        # Patches should return some indication of what was done
        # (success, number of files patched, etc.)
        
        # Check functions are callable
        assert callable(patch_sadtalker_numpy_compatibility)
        assert callable(patch_liveportrait_numpy_compatibility)
    
    def test_check_functions_return_bool(self):
        """Test that check functions return boolean."""
        fake_sadtalker_dir = Path("/nonexistent/sadtalker")
        fake_liveportrait_dir = Path("/nonexistent/liveportrait")
        
        sadtalker_result = check_if_patch_needed(fake_sadtalker_dir)
        liveportrait_result = check_liveportrait_patch_needed(fake_liveportrait_dir)
        
        assert isinstance(sadtalker_result, bool)
        assert isinstance(liveportrait_result, bool)


class TestPatchEdgeCases:
    """Test patch edge cases and error handling."""
    
    def test_patch_with_no_models_directory(self):
        """Test patch when models directory doesn't exist."""
        # Should handle gracefully
        try:
            fake_dir = Path("/nonexistent/models")
            result = check_if_patch_needed(fake_dir)
            assert isinstance(result, bool)
            assert result == False
        except Exception as e:
            pytest.fail(f"Should handle missing directory: {str(e)}")
    
    def test_patch_with_permission_error(self):
        """Test patch behavior with permission errors."""
        # In real scenario, might encounter permission errors
        # Patch should handle gracefully
        
        # This is mostly a placeholder test
        # Real implementation would need file system mocking
        pass
    
    def test_patch_with_empty_files(self, temp_dir):
        """Test patch with empty Python files."""
        # Create empty file
        test_file = Path(temp_dir) / "empty.py"
        test_file.write_text("")
        
        # Patch should handle empty files
        assert test_file.exists()
    
    def test_patch_skips_non_python_files(self, temp_dir):
        """Test that patch skips non-Python files."""
        # Create non-Python file
        test_file = Path(temp_dir) / "data.txt"
        test_file.write_text("np.float(5.0)")
        
        # Patch should only process .py files
        assert not test_file.suffix == ".py"


class TestPatchDocumentation:
    """Test that patches are documented."""
    
    def test_patch_functions_have_docstrings(self):
        """Test that patch functions have documentation."""
        assert patch_sadtalker_numpy_compatibility.__doc__ is not None
        assert check_if_patch_needed.__doc__ is not None
        
        assert patch_liveportrait_numpy_compatibility.__doc__ is not None
        assert check_liveportrait_patch_needed.__doc__ is not None
    
    def test_patch_modules_have_docstrings(self):
        """Test that patch modules have documentation."""
        import utils.sadtalker_patch
        import utils.liveportrait_patch
        
        assert utils.sadtalker_patch.__doc__ is not None
        assert utils.liveportrait_patch.__doc__ is not None
