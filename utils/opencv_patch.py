"""
OpenCV Compatibility Patch
Handles potential conflicts between opencv-python packages
"""
import cv2
import logging

logger = logging.getLogger(__name__)


def apply_opencv_patch():
    """
    Apply patch for OpenCV constant compatibility issues.
    
    When opencv-python and opencv-python-headless coexist, some constants
    like CAP_PROP_WIDTH may not be exposed correctly in the cv2 namespace.
    This patch ensures these constants are available.
    """
    try:
        # Check if constants are missing
        missing_constants = []
        
        constants_to_check = {
            'CAP_PROP_WIDTH': 3,
            'CAP_PROP_HEIGHT': 4,
            'CAP_PROP_FPS': 5,
            'CAP_PROP_FRAME_COUNT': 7,
            'CAP_PROP_POS_FRAMES': 1,
        }
        
        for const_name, const_value in constants_to_check.items():
            if not hasattr(cv2, const_name):
                missing_constants.append(const_name)
                setattr(cv2, const_name, const_value)
        
        if missing_constants:
            logger.warning(f"⚠️ OpenCV patch applied for missing constants: {', '.join(missing_constants)}")
            logger.info("✓ OpenCV constants patched successfully")
        else:
            logger.debug("✓ OpenCV constants already available, no patch needed")
            
    except Exception as e:
        logger.error(f"❌ Error applying OpenCV patch: {e}")


# Apply patch on import
apply_opencv_patch()
