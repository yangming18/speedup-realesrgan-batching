#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Background Remover Tab
Uses AI models to automatically remove backgrounds from images
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import tempfile

logger = logging.getLogger(__name__)


class BackgroundRemoverTab:
    """Manages the AI background remover tab"""
    
    def __init__(self, i18n_manager):
        self.i18n = i18n_manager
        self.rembg_available = False
        self._check_rembg()
    
    def _check_rembg(self):
        """Check if rembg is available"""
        try:
            import rembg
            self.rembg_available = True
            logger.info("rembg library available")
        except ImportError:
            logger.warning("rembg not installed. Install with: pip install rembg")
            self.rembg_available = False
    
    def remove_background(
        self,
        input_image: Optional[np.ndarray],
        background_type: str,
        bg_color: str,
        rgb_r: int,
        rgb_g: int,
        rgb_b: int,
        alpha_matting: bool,
        progress=gr.Progress()
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Remove background from image using AI.
        
        Args:
            input_image: Input image as numpy array
            background_type: "transparent" or "color"
            bg_color: Color from color picker (hex)
            rgb_r: Red value (0-255)
            rgb_g: Green value (0-255)
            rgb_b: Blue value (0-255)
            alpha_matting: Enable alpha matting for better edges
            progress: Gradio progress tracker
        
        Returns:
            Tuple of (output_image, status_message)
        """
        if not self.rembg_available:
            return None, "❌ rembg not installed. Run: pip install rembg"
        
        if input_image is None:
            return None, "❌ Please upload an image"
        
        try:
            from rembg import remove
            
            progress(0.2, "Loading AI model...")
            logger.info("Starting background removal process")
            
            # Convert numpy array to PIL Image
            if isinstance(input_image, np.ndarray):
                input_pil = Image.fromarray(input_image)
            else:
                input_pil = input_image
            
            progress(0.4, "Removing background with AI...")
            
            # Remove background
            output_pil = remove(
                input_pil,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10
            )
            
            progress(0.7, "Processing result...")
            
            # If user wants colored background instead of transparent
            if background_type == "color":
                # Determine which color to use
                if bg_color and bg_color != "#ffffff":
                    # Use color picker value
                    bg_rgb = self._hex_to_rgb(bg_color)
                else:
                    # Use RGB sliders
                    bg_rgb = (rgb_r, rgb_g, rgb_b)
                
                logger.info(f"Applying background color: RGB{bg_rgb}")
                
                # Create new image with colored background
                final_image = Image.new("RGB", output_pil.size, bg_rgb)
                
                # Paste the foreground on top (using alpha channel as mask)
                if output_pil.mode == 'RGBA':
                    final_image.paste(output_pil, (0, 0), output_pil)
                else:
                    final_image = output_pil.convert("RGB")
                
                output_pil = final_image
            
            progress(1.0, "✓ Complete!")
            
            # Convert back to numpy array for Gradio
            output_array = np.array(output_pil)
            
            size_info = f"{output_pil.width}x{output_pil.height}"
            mode_info = output_pil.mode
            
            return output_array, f"✓ Background removed successfully! ({size_info}, {mode_info})"
            
        except Exception as e:
            logger.error(f"Error removing background: {e}", exc_info=True)
            return None, f"❌ Error: {str(e)}"
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """Convert RGB to hex color"""
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def create_preview_grid(self) -> np.ndarray:
        """Create a color picker preview grid"""
        # Create a simple RGB color wheel/grid
        size = 300
        img = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(img)
        
        # Draw color gradient
        for x in range(size):
            for y in range(size):
                r = int(255 * x / size)
                g = int(255 * y / size)
                b = 128  # Fixed blue component
                draw.point((x, y), fill=(r, g, b))
        
        return np.array(img)
    
    def sync_color_to_rgb(self, color: str) -> Tuple[int, int, int]:
        """Sync color picker value to RGB sliders"""
        if not color or color == "#ffffff":
            return 255, 255, 255
        rgb = self._hex_to_rgb(color)
        return rgb
    
    def sync_rgb_to_color(self, r: int, g: int, b: int) -> str:
        """Sync RGB sliders to color picker"""
        return self._rgb_to_hex(r, g, b)
    
    def update_background_options(self, bg_type: str) -> dict:
        """Show/hide color options based on background type"""
        if bg_type == "color":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
    
    def create_tab(self):
        """Create the tab in Gradio interface"""
        _ = self.i18n.t
        
        with gr.Tab(_("tabs.background_remover")):
            self.create_ui()
    
    def create_ui(self) -> gr.Blocks:
        """Create the background remover UI"""
        _ = self.i18n.t
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"### {_('ai_bg_remover.title')}")
                
                # Status check
                if not self.rembg_available:
                    gr.Markdown("⚠️ **rembg not installed**. Run: `pip install rembg`")
                else:
                    gr.Markdown("✓ AI model ready")
                
                # Input
                input_image = gr.Image(
                    label=_("ai_bg_remover.input_image"),
                    type="numpy",
                    sources=["upload"],
                    height=300
                )
                
                # Background type selection
                background_type = gr.Radio(
                    choices=[
                        (_("ai_bg_remover.transparent"), "transparent"),
                        (_("ai_bg_remover.colored"), "color")
                    ],
                    value="transparent",
                    label=_("ai_bg_remover.background_type"),
                    interactive=True
                )
                
                # Color options (initially hidden)
                with gr.Group(visible=False) as color_options:
                    gr.Markdown(f"#### {_('ai_bg_remover.choose_color')}")
                    
                    # Color picker
                    bg_color = gr.ColorPicker(
                        label=_("ai_bg_remover.color_picker"),
                        value="#ffffff",
                        interactive=True
                    )
                    
                    gr.Markdown(_("ai_bg_remover.or_rgb"))
                    
                    # RGB sliders
                    with gr.Row():
                        rgb_r = gr.Slider(
                            minimum=0,
                            maximum=255,
                            value=255,
                            step=1,
                            label="R",
                            interactive=True
                        )
                        rgb_g = gr.Slider(
                            minimum=0,
                            maximum=255,
                            value=255,
                            step=1,
                            label="G",
                            interactive=True
                        )
                        rgb_b = gr.Slider(
                            minimum=0,
                            maximum=255,
                            value=255,
                            step=1,
                            label="B",
                            interactive=True
                        )
                
                # Advanced options
                with gr.Accordion(_("ai_bg_remover.advanced"), open=False):
                    alpha_matting = gr.Checkbox(
                        label=_("ai_bg_remover.alpha_matting"),
                        value=False,
                        info=_("ai_bg_remover.alpha_matting_info")
                    )
                
                # Process button
                process_btn = gr.Button(
                    _("ai_bg_remover.remove_bg"),
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown(f"### {_('ai_bg_remover.result')}")
                
                # Output
                output_image = gr.Image(
                    label=_("ai_bg_remover.output_image"),
                    type="numpy",
                    height=400
                )
                
                # Status
                status = gr.Textbox(
                    label=_("common.info"),
                    interactive=False,
                    lines=2
                )
                
                # Download button
                download_btn = gr.File(
                    label=_("common.download"),
                    visible=False
                )
        
        # Examples
        gr.Markdown(f"### {_('common.info')}")
        gr.Markdown(_("ai_bg_remover.examples_text"))
        
        # Event handlers
        
        # Show/hide color options based on background type
        background_type.change(
            fn=self.update_background_options,
            inputs=[background_type],
            outputs=[color_options]
        )
        
        # Sync color picker with RGB sliders
        bg_color.change(
            fn=self.sync_color_to_rgb,
            inputs=[bg_color],
            outputs=[rgb_r, rgb_g, rgb_b]
        )
        
        # Sync RGB sliders with color picker
        for slider in [rgb_r, rgb_g, rgb_b]:
            slider.change(
                fn=self.sync_rgb_to_color,
                inputs=[rgb_r, rgb_g, rgb_b],
                outputs=[bg_color]
            )
        
        # Process button
        process_btn.click(
            fn=self.remove_background,
            inputs=[
                input_image,
                background_type,
                bg_color,
                rgb_r, rgb_g, rgb_b,
                alpha_matting
            ],
            outputs=[output_image, status]
        )
        
        return gr.Row()
