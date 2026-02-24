"""
LivePortrait checkpoint downloader
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_liveportrait_checkpoints(pretrained_weights_dir):
    """
    Download LivePortrait pretrained weights from HuggingFace using Python API
    
    Args:
        pretrained_weights_dir: Path to pretrained_weights directory
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        
        pretrained_weights_dir = Path(pretrained_weights_dir)
        pretrained_weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        required_files = [
            'liveportrait/base_models/appearance_feature_extractor.pth',
            'liveportrait/base_models/motion_extractor.pth',
            'liveportrait/base_models/spade_generator.pth',
            'liveportrait/base_models/warping_module.pth',
            'liveportrait/landmark.onnx',
            'liveportrait/retinaface_det_static.onnx'
        ]
        
        all_exist = all((pretrained_weights_dir / f).exists() for f in required_files)
        if all_exist:
            logger.info("✅ LivePortrait checkpoints already downloaded")
            return True
        
        logger.info("📥 Downloading LivePortrait pretrained weights from HuggingFace...")
        logger.info("This may take a few minutes (download size: ~500MB)")
        
        # Download using Python API
        snapshot_download(
            repo_id="KlingTeam/LivePortrait",
            local_dir=str(pretrained_weights_dir),
            ignore_patterns=["*.git*", "README.md", "docs/*"],
            local_dir_use_symlinks=False
        )
        
        # Verify download
        if all((pretrained_weights_dir / f).exists() for f in required_files):
            logger.info("✅ LivePortrait checkpoints downloaded successfully!")
            return True
        else:
            logger.error("❌ Download completed but some files are missing")
            return False
            
    except ImportError:
        logger.error("❌ huggingface_hub not found! Install with: pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error(f"❌ Error downloading checkpoints: {e}", exc_info=True)
        return False


def check_liveportrait_ready(liveportrait_dir):
    """
    Check if LivePortrait is ready (checkpoints downloaded)
    
    Returns:
        bool: True if ready, False if needs download
    """
    pretrained_weights_dir = Path(liveportrait_dir) / 'pretrained_weights'
    
    required_files = [
        'liveportrait/base_models/appearance_feature_extractor.pth',
        'liveportrait/base_models/motion_extractor.pth',
        'liveportrait/base_models/spade_generator.pth',
        'liveportrait/base_models/warping_module.pth'
    ]
    
    return all((pretrained_weights_dir / f).exists() for f in required_files)
