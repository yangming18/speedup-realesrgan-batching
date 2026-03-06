#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Gargantuas Hybrid LipSync
Advanced lip-sync pipeline combining Wav2Lip GAN with face enhancement
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import subprocess
import sys
import shutil
import tempfile

logger = logging.getLogger(__name__)


class HybridLipSyncProcessor:
    """
    Advanced hybrid lip-sync processor that combines:
    1. Wav2Lip GAN for lip-sync animation
    2. GFPGAN for face restoration and enhancement
    
    This simplified approach applies Wav2Lip GAN first, then enhances
    the result with GFPGAN for improved face quality.
    """
    
    def __init__(self, models_dir: Path, device: str = 'cuda'):
        """
        Initialize the hybrid processor.
        
        Args:
            models_dir: Base models directory
            device: Device to use ('cuda', 'cpu', 'mps')
        """
        self.models_dir = models_dir
        self.device = device
        self.temp_dir = Path(tempfile.gettempdir()) / "hybrid_lipsync"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Paths to required models
        self.wav2lip_dir = models_dir / "lipsync" / "wav2lip"
        self.wav2lip_checkpoint = self.wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"
        
        logger.info(f"HybridLipSyncProcessor initialized on {device}")
    
    def _apply_wav2lip_gan(
        self,
        video_path: str,
        audio_path: str,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Apply Wav2Lip GAN to create lip-synced video.
        Returns path to processed video.
        """
        if progress_callback:
            progress_callback(10, "🎭 Applying Wav2Lip GAN...")
        
        temp_wav2lip_output = self.temp_dir / "wav2lip_result.avi"
        
        # Remove old output if exists
        if temp_wav2lip_output.exists():
            temp_wav2lip_output.unlink()
        
        cmd = [
            sys.executable,
            str(self.wav2lip_dir / "inference.py"),
            '--checkpoint_path', str(self.wav2lip_checkpoint),
            '--face', str(video_path),
            '--audio', str(audio_path),
            '--outfile', str(temp_wav2lip_output),
        ]
        
        logger.info(f"Running Wav2Lip GAN: {' '.join(cmd)}")
        
        if progress_callback:
            progress_callback(15, "⏳ Processing with Wav2Lip GAN...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 or not temp_wav2lip_output.exists():
            error_details = result.stderr if result.stderr else result.stdout
            raise RuntimeError(f"Wav2Lip GAN failed: {error_details}")
        
        if progress_callback:
            progress_callback(50, "✅ Wav2Lip GAN complete")
        
        return str(temp_wav2lip_output)
    
    def _enhance_with_gfpgan(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Apply GFPGAN face enhancement to video.
        
        Args:
            video_path: Input video path
            
        Returns:
            Path to enhanced video
        """
        if progress_callback:
            progress_callback(55, "✨ Enhancing face quality with GFPGAN...")
        
        try:
            from utils.detail_enhancer import DetailEnhancer
        except ImportError:
            logger.warning("DetailEnhancer not available, skipping face enhancement")
            return video_path
        
        try:
            enhancer = DetailEnhancer(model_name='gfpgan_face_restore', device_preference=self.device)
        except Exception as e:
            logger.warning(f"Could not initialize GFPGAN: {e}, skipping enhancement")
            return video_path
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_enhanced = self.temp_dir / "gfpgan_enhanced.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_enhanced), fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance frame with GFPGAN
            try:
                frame_enhanced = enhancer.enhance(frame)
            except Exception as e:
                logger.warning(f"Frame {frame_idx} enhancement failed: {e}, using original")
                frame_enhanced = frame
            
            out.write(frame_enhanced)
            frame_idx += 1
            
            if progress_callback and frame_idx % 10 == 0:
                progress = 55 + int(40 * frame_idx / total_frames)
                progress_callback(progress, f"🌟 Enhancing faces: {frame_idx}/{total_frames}")
        
        cap.release()
        out.release()
        
        logger.info(f"GFPGAN enhancement completed: {frame_idx} frames")
        return str(output_enhanced)
    
    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bool:
        """
        Main processing pipeline for The Gargantuas Hybrid LipSync.
        
        Args:
            video_path: Input video or image path
            audio_path: Input audio path
            output_path: Output video path
            progress_callback: Called with (percentage, message)
            
        Returns:
            True if successful
        """
        try:
            if progress_callback:
                progress_callback(0, "🚀 Starting The Gargantuas Hybrid LipSync...")
            
            # Step 1: Apply Wav2Lip GAN for lip-sync
            video_with_lipsync = self._apply_wav2lip_gan(
                video_path, audio_path, progress_callback
            )
            
            # Step 2: Enhance face quality with GFPGAN
            video_with_enhanced_face = self._enhance_with_gfpgan(
                video_with_lipsync, progress_callback
            )
            
            # Step 3: Add audio and finalize with ffmpeg
            if progress_callback:
                progress_callback(96, "🎵 Adding audio and finalizing...")
            
            # Convert all paths to Path objects first, then to strings
            from pathlib import Path
            video_enhanced_path = Path(video_with_enhanced_face)
            audio_input_path = Path(audio_path)
            final_output_path = Path(output_path)
            
            # Ensure all paths exist before passing to ffmpeg
            if not video_enhanced_path.exists():
                raise FileNotFoundError(f"Enhanced video not found: {video_enhanced_path}")
            if not audio_input_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_input_path}")
            
            logger.info(f"Final ffmpeg merge:")
            logger.info(f"  Video: {video_enhanced_path}")
            logger.info(f"  Audio: {audio_input_path}")
            logger.info(f"  Output: {final_output_path}")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_enhanced_path),
                '-i', str(audio_input_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(final_output_path)
            ]
            
            logger.info(f"FFmpeg command: {cmd}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            if progress_callback:
                progress_callback(100, "✅ The Gargantuas Hybrid LipSync complete!")
            
            logger.info(f"Hybrid lipsync completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Hybrid lipsync failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(0, f"❌ Error: {str(e)}")
            self.last_error = str(e)
            return False
        
        finally:
            # Clean up temp files
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Remove temporary files"""
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.glob("*"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.debug(f"Could not remove {file}: {e}")
                logger.info("Temporary files cleaned")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

