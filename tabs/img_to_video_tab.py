#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Image to Video Tab
Uses Stable Video Diffusion to generate videos from images
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import tempfile
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ImgToVideoTab:
    """Manages the AI image to video generation tab"""
    
    def __init__(self, i18n_manager):
        self.i18n = i18n_manager
        self.pipe = None
        self.device = None
        self.model_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required libraries are available"""
        try:
            import torch
            import diffusers
            from diffusers import StableVideoDiffusionPipeline
            
            # Detect device
            # Force CPU even on MPS: Conv3D operations are not supported on MPS
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                if torch.backends.mps.is_available():
                    logger.info("MPS detected but Conv3D not supported. Using CPU for Stable Video Diffusion.")
            
            self.model_available = True
            logger.info(f"Stable Video Diffusion available on device: {self.device}")
        except ImportError as e:
            logger.warning(f"Stable Video Diffusion not available: {e}")
            self.model_available = False
    
    def load_model(self, progress=None):
        """Load the Stable Video Diffusion model (lazy loading)"""
        if self.pipe is None and self.model_available:
            try:
                if progress:
                    progress(0, desc="📦 Loading Stable Video Diffusion model...")
                
                from diffusers import StableVideoDiffusionPipeline
                import torch
                
                logger.info("Loading StableVideoDiffusionPipeline...")
                # Use float32 on CPU, float16 on CUDA
                dtype = torch.float32 if self.device == "cpu" else torch.float16
                variant = None if self.device == "cpu" else "fp16"
                
                self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=dtype,
                    variant=variant
                )
                
                if progress:
                    progress(0.5, desc="⚡ Moving model to device...")
                
                self.pipe.to(self.device)
                self.pipe.enable_attention_slicing()
                
                if progress:
                    progress(1.0, desc="✅ Model loaded!")
                
                logger.info("Model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}", exc_info=True)
                return False
        return self.pipe is not None
    
    def detect_aspect_ratio(self, image_path: str) -> str:
        """Detect image aspect ratio and suggest best resolution"""
        try:
            img = Image.open(image_path)
            w, h = img.size
            ratio = w / h
            
            if ratio > 1.5:  # Landscape
                return "1024x576"
            elif ratio < 0.7:  # Portrait
                return "576x1024"
            else:  # Square-ish
                return "768x768"
        except:
            return "768x768"
    
    def resize_image(self, image: Image.Image, target_width: int, target_height: int, mode: str = "crop") -> Image.Image:
        """Resize image with different strategies"""
        img = image.convert("RGB")
        target_ratio = target_width / target_height
        img_ratio = img.width / img.height
        
        if mode == "crop":
            # Resize maintaining aspect ratio, then crop center
            if img_ratio > target_ratio:
                new_height = target_height
                new_width = int(target_height * img_ratio)
            else:
                new_width = target_width
                new_height = int(target_width / img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop center
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            img = img.crop((left, top, left + target_width, top + target_height))
            
        elif mode == "fit":
            # Fit with padding (black bars)
            img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            background = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            offset = ((target_width - img.width) // 2, (target_height - img.height) // 2)
            background.paste(img, offset)
            img = background
            
        elif mode == "stretch":
            # Stretch to fill (distortion)
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return img
    
    def generate_video(
        self,
        image_path: str,
        resolution: str,
        resize_mode: str,
        duration: int,
        fps: int,
        num_inference_steps: int,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], str]:
        """
        Generate video from image using Stable Video Diffusion.
        
        Returns:
            Tuple of (video_path, status_message)
        """
        if not self.model_available:
            return None, "❌ Stable Video Diffusion not installed. Run: pip install diffusers accelerate"
        
        if not image_path:
            return None, "❌ Please upload an image"
        
        try:
            # Load model
            progress(0, desc="🔄 Initializing...")
            if not self.load_model(progress):
                return None, "❌ Failed to load model"
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Load and process image
            progress(0.1, desc=f"🖼️ Processing image ({resize_mode} mode)...")
            image = Image.open(image_path)
            orig_size = f"{image.width}x{image.height}"
            
            # Resize according to selected mode
            image = self.resize_image(image, width, height, resize_mode)
            
            # Calculate number of frames
            num_frames = int(duration * fps)
            
            # Progress callback
            start_time = time.time()
            
            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                progress_pct = (step + 1) / num_inference_steps
                elapsed = time.time() - start_time
                eta = elapsed / (step + 1) * (num_inference_steps - step - 1)
                
                progress(
                    0.2 + progress_pct * 0.7,
                    desc=f"🎬 Generating... Step {step + 1}/{num_inference_steps} | "
                         f"⏱️ {elapsed:.1f}s | ETA: {eta:.1f}s"
                )
                return callback_kwargs
            
            # Generate video
            progress(0.2, desc=f"🎬 Starting generation ({num_frames} frames @ {fps}fps)...")
            
            logger.info(f"Generating video: {width}x{height}, {num_frames} frames, {fps}fps")
            
            output = self.pipe(
                image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                fps=fps,
                decode_chunk_size=8,
                callback_on_step_end=callback_on_step_end
            )
            
            frames = output.frames[0]
            
            # Save video
            progress(0.95, desc="💾 Saving video...")
            
            output_dir = Path("temp/img2video")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"video_{timestamp}.mp4"
            
            # Export to video using imageio
            import imageio
            writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264')
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            
            elapsed_total = time.time() - start_time
            
            progress(1.0, desc="✅ Video generated!")
            
            info = f"""
✅ **Video Generated Successfully!**

📊 **Stats:**
- Original image: {orig_size}
- Output resolution: {width}x{height}
- Duration: {duration}s @ {fps}fps
- Total frames: {num_frames}
- Inference steps: {num_inference_steps}
- Resize mode: {resize_mode}
- Generation time: {elapsed_total:.1f}s
- Device: {self.device}
"""
            
            logger.info(f"Video generated: {output_path} ({elapsed_total:.1f}s)")
            
            return str(output_path), info
            
        except Exception as e:
            logger.error(f"Error generating video: {e}", exc_info=True)
            return None, f"❌ Error: {str(e)}"
    
    def create_tab(self):
        """Create the tab in Gradio interface"""
        _ = self.i18n.t
        
        with gr.Tab(_("tabs.img_to_video")):
            if not self.model_available:
                gr.Markdown("⚠️ **Stable Video Diffusion not installed**\n\nRun: `pip install diffusers accelerate`")
            
            self.create_ui()
    
    def create_ui(self):
        """Create the image to video UI"""
        _ = self.i18n.t
        
        # Resolution options with descriptions
        resolutions = {
            # Landscape 16:9
            "1024x576": "1024x576 (16:9 Landscape - YouTube/TV)",
            "896x504": "896x504 (16:9 Landscape - Balanced)",
            "768x432": "768x432 (16:9 Landscape - Fast)",
            
            # Portrait 9:16
            "576x1024": "576x1024 (9:16 Portrait - TikTok/Instagram)",
            "504x896": "504x896 (9:16 Portrait - Balanced)",
            "432x768": "432x768 (9:16 Portrait - Fast)",
            
            # Square 1:1
            "1024x1024": "1024x1024 (1:1 Square - Max Quality)",
            "768x768": "768x768 (1:1 Square - Recommended)",
            "512x512": "512x512 (1:1 Square - Fast)",
            "256x256": "256x256 (1:1 Square - Test)",
            
            # Cinematic 21:9
            "1024x438": "1024x438 (21:9 Cinematic)",
        }
        
        resize_modes = {
            "crop": "Crop (Center) - Crops excess, maintains center",
            "fit": "Fit (Padding) - Keeps all, adds black bars",
            "stretch": "Stretch - Distorts to fill"
        }
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"### {_('img_to_video.title')}")
                
                # Input
                image_input = gr.Image(
                    label=_("img_to_video.input_image"),
                    type="filepath",
                    sources=["upload"]
                )
                
                # Resolution
                resolution_dropdown = gr.Dropdown(
                    choices=list(resolutions.values()),
                    value=resolutions["768x768"],
                    label=_("img_to_video.resolution"),
                    info="Auto-detected based on image aspect ratio"
                )
                
                # Resize mode
                resize_mode_radio = gr.Radio(
                    choices=list(resize_modes.values()),
                    value=resize_modes["crop"],
                    label=_("img_to_video.resize_mode"),
                    info="How to handle different aspect ratios"
                )
                
                # Duration and FPS
                with gr.Row():
                    duration_slider = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=5,
                        step=1,
                        label=_("img_to_video.duration")
                    )
                    fps_slider = gr.Slider(
                        minimum=5,
                        maximum=15,
                        value=7,
                        step=1,
                        label=_("img_to_video.fps")
                    )
                
                # Inference steps
                steps_slider = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=25,
                    step=5,
                    label=_("img_to_video.inference_steps"),
                    info=_("img_to_video.inference_steps_info")
                )
                
                # Generate button
                generate_btn = gr.Button(
                    _("img_to_video.generate_button"),
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown(f"### {_('img_to_video.output_video')}")
                
                # Output video
                video_output = gr.Video(
                    label=_("img_to_video.output_video"),
                    format="mp4"
                )
                
                # Info output
                info_output = gr.Markdown()
        
        # Auto-detect resolution on image upload
        def on_image_upload(image_path):
            if image_path:
                detected_res = self.detect_aspect_ratio(image_path)
                return resolutions[detected_res]
            return resolutions["768x768"]
        
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[resolution_dropdown]
        )
        
        # Generate video
        def generate_wrapper(image, res_label, resize_label, dur, fps_val, steps):
            # Convert labels back to keys
            res_key = [k for k, v in resolutions.items() if v == res_label][0]
            resize_key = [k for k, v in resize_modes.items() if v == resize_label][0]
            
            return self.generate_video(image, res_key, resize_key, dur, fps_val, steps)
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[image_input, resolution_dropdown, resize_mode_radio, duration_slider, fps_slider, steps_slider],
            outputs=[video_output, info_output]
        )
