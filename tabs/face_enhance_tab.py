"""
Face Enhancement Tab
AI-powered face and image enhancement without upscaling
"""
import os
import logging
import gradio as gr
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from utils.detail_enhancer import DetailEnhancer, get_detail_model_choices

logger = logging.getLogger(__name__)


class FaceEnhanceTab:
    """Face and image enhancement tab"""
    
    def __init__(self, temp_manager, device_manager, i18n=None):
        self.temp_manager = temp_manager
        self.device_manager = device_manager
        self.i18n = i18n
        self.enhancer = None
        self.current_model = None
        
    def _t(self, key, **kwargs):
        """Helper per traduzione con fallback"""
        if self.i18n:
            return self.i18n.t(key, **kwargs)
        return key
    
    def _get_model_choices(self):
        """Ottiene lista modelli disponibili"""
        choices = get_detail_model_choices()
        # Filtra "none" e restituisce solo le etichette disponibili
        available = [(label, key) for key, label, avail in choices if avail and key != "none"]
        return available
    
    def _enhance_image(
        self,
        input_file,
        model_name: str,
        device: str,
        progress=gr.Progress()
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Applica enhancement all'immagine
        
        Args:
            input_file: File immagine caricato
            model_name: Nome modello (etichetta visualizzata)
            device: Dispositivo (cpu/mps/cuda)
            progress: Gradio progress tracker
            
        Returns:
            Tuple (immagine migliorata, messaggio stato)
        """
        try:
            if input_file is None:
                return None, self._t('face_enhance.error_no_input')
            
            # Mappa etichetta -> chiave modello
            model_choices = self._get_model_choices()
            model_key = None
            for label, key in model_choices:
                if label == model_name:
                    model_key = key
                    break
            
            if not model_key:
                return None, f"❌ {self._t('common.error')}: Invalid model"
            
            progress(0.1, self._t('face_enhance.loading_model'))
            
            # Leggi immagine
            input_path = input_file if isinstance(input_file, str) else input_file.name
            image_bgr = cv2.imread(input_path)
            
            if image_bgr is None:
                return None, f"❌ {self._t('common.error')}: Cannot read image"
            
            progress(0.3, self._t('face_enhance.processing'))
            
            # Inizializza enhancer se necessario
            device_pref = None if device == "auto" else device
            
            if self.enhancer is None or self.current_model != model_key:
                logger.info(f"Initializing enhancer: {model_key}")
                if self.enhancer:
                    self.enhancer.release()
                
                # Inizializza nuovo enhancer
                self.enhancer = DetailEnhancer(model_key, device_preference=device_pref)
                self.current_model = model_key
            
            progress(0.5, self._t('face_enhance.applying'))
            
            # Applica enhancement
            enhanced_bgr = self.enhancer.enhance(image_bgr, focus_areas=None)
            
            progress(0.9, self._t('face_enhance.finalizing'))
            
            # Converti BGR -> RGB per Gradio
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            
            success_msg = self._t('face_enhance.success', model=model_name)
            
            return enhanced_rgb, f"✅ {success_msg}"
            
        except Exception as e:
            logger.error(f"Error in face enhancement: {e}", exc_info=True)
            error_msg = self._t('face_enhance.error')
            return None, f"❌ {error_msg}: {str(e)}"
    
    def _enhance_video(
        self,
        input_file,
        model_name: str,
        device: str,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], str]:
        """
        Applica enhancement a tutti i frame del video
        
        Args:
            input_file: File video caricato
            model_name: Nome modello (etichetta visualizzata)
            device: Dispositivo (cpu/mps/cuda)
            progress: Gradio progress tracker
            
        Returns:
            Tuple (path video output, messaggio stato)
        """
        try:
            if input_file is None:
                return None, self._t('face_enhance.error_no_input')
            
            # Mappa etichetta -> chiave modello
            model_choices = self._get_model_choices()
            model_key = None
            for label, key in model_choices:
                if label == model_name:
                    model_key = key
                    break
            
            if not model_key:
                return None, f"❌ {self._t('common.error')}: Invalid model"
            
            progress(0.05, self._t('face_enhance.loading_model'))
            
            input_path = input_file if isinstance(input_file, str) else input_file.name
            
            # Inizializza enhancer
            device_pref = None if device == "auto" else device
            
            if self.enhancer is None or self.current_model != model_key:
                logger.info(f"Initializing enhancer: {model_key}")
                if self.enhancer:
                    self.enhancer.release()
                
                self.enhancer = DetailEnhancer(model_key, device_preference=device_pref)
                self.current_model = model_key
            
            progress(0.1, self._t('face_enhance.reading_video'))
            
            # Apri video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return None, f"❌ {self._t('common.error')}: Cannot open video"
            
            # Ottieni proprietà video
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Crea file output temporaneo
            output_path = self.temp_manager.get_temp_file_path(f"enhanced_{Path(input_path).stem}.mp4")
            
            # Writer video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_idx = 0
            
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                # Applica enhancement
                enhanced_bgr = self.enhancer.enhance(frame_bgr, focus_areas=None)
                
                # Scrivi frame
                out.write(enhanced_bgr)
                
                frame_idx += 1
                
                # Aggiorna progress
                if frame_idx % 10 == 0:
                    prog = 0.1 + (0.8 * frame_idx / total_frames)
                    progress(prog, f"{self._t('face_enhance.processing')} ({frame_idx}/{total_frames})")
            
            cap.release()
            out.release()
            
            progress(0.95, self._t('face_enhance.finalizing'))
            
            # Copia audio dal video originale
            import subprocess
            final_output = self.temp_manager.get_temp_file_path(f"enhanced_final_{Path(input_path).stem}.mp4")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(output_path),
                '-i', input_path,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                str(final_output)
            ]
            
            subprocess.run(cmd, capture_output=True, check=False)
            
            # Se ffmpeg fallisce, usa il video senza audio
            if not final_output.exists():
                final_output = output_path
            
            success_msg = self._t('face_enhance.success', model=model_name)
            return str(final_output), f"✅ {success_msg}"
            
        except Exception as e:
            logger.error(f"Error in video enhancement: {e}", exc_info=True)
            error_msg = self._t('face_enhance.error')
            return None, f"❌ {error_msg}: {str(e)}"
    
    def create_tab(self):
        """Crea e restituisce l'interfaccia del tab Face Enhancement"""
        with gr.Tab(self._t('tabs.face_enhance')):
            gr.Markdown("""
            # {title}
            {description}
            
            ---
            """.format(
                title=self._t('face_enhance.title'),
                description=self._t('face_enhance.description')
            ))
            
            with gr.Row():
                with gr.Column():
                    # Selezione tipo input
                    input_type = gr.Radio(
                        choices=[
                            self._t('face_enhance.image'),
                            self._t('face_enhance.video')
                        ],
                        value=self._t('face_enhance.image'),
                        label=self._t('face_enhance.input_type'),
                        info=self._t('face_enhance.input_type_info')
                    )
                    
                    # Upload immagine
                    image_input = gr.Image(
                        label=self._t('face_enhance.upload_image'),
                        type="filepath",
                        visible=True
                    )
                    
                    # Upload video
                    video_input = gr.Video(
                        label=self._t('face_enhance.upload_video'),
                        visible=False
                    )
                    
                    # Selezione modello
                    model_choices = self._get_model_choices()
                    model_labels = [label for label, _ in model_choices]
                    
                    model_dropdown = gr.Dropdown(
                        choices=model_labels,
                        value=model_labels[0] if model_labels else None,
                        label=self._t('face_enhance.model'),
                        info=self._t('face_enhance.model_info')
                    )
                    
                    # Selezione dispositivo
                    device_dropdown = gr.Dropdown(
                        choices=self.device_manager.get_available_devices(),
                        value=self.device_manager.current_device,
                        label=self._t('face_enhance.device'),
                        info=self._t('face_enhance.device_info')
                    )
                    
                    # Pulsante elabora
                    process_btn = gr.Button(
                        self._t('face_enhance.process_button'),
                        variant="primary"
                    )
                
                with gr.Column():
                    # Output immagine
                    image_output = gr.Image(
                        label=self._t('face_enhance.output_image'),
                        type="numpy",
                        visible=True
                    )
                    
                    # Output video
                    video_output = gr.Video(
                        label=self._t('face_enhance.output_video'),
                        visible=False
                    )
                    
                    # Messaggio stato
                    status_output = gr.Textbox(
                        label=self._t('face_enhance.info'),
                        interactive=False
                    )
            
            # Tips section
            gr.Markdown(f"""
            ### {self._t('face_enhance.tips_title')}
            {self._t('face_enhance.tips_content')}
            """)
            
            # Gestione cambio tipo input
            def update_visibility(input_choice):
                is_image = input_choice == self._t('face_enhance.image')
                return {
                    image_input: gr.update(visible=is_image),
                    video_input: gr.update(visible=not is_image),
                    image_output: gr.update(visible=is_image),
                    video_output: gr.update(visible=not is_image)
                }
            
            input_type.change(
                fn=update_visibility,
                inputs=[input_type],
                outputs=[image_input, video_input, image_output, video_output]
            )
            
            # Gestione processing
            def process_enhancement(input_choice, img_file, vid_file, model, device, progress=gr.Progress()):
                is_image = input_choice == self._t('face_enhance.image')
                
                if is_image:
                    return self._enhance_image(img_file, model, device, progress)
                else:
                    result_path, message = self._enhance_video(vid_file, model, device, progress)
                    # Per video, restituiamo None per image_output
                    return None, message, result_path
            
            # Click handler per immagini
            def handle_image_process(input_choice, img_file, vid_file, model, device, progress=gr.Progress()):
                is_image = input_choice == self._t('face_enhance.image')
                if is_image:
                    result_img, message = self._enhance_image(img_file, model, device, progress)
                    return result_img, message, None
                return None, "", None
            
            # Click handler per video
            def handle_video_process(input_choice, img_file, vid_file, model, device, progress=gr.Progress()):
                is_image = input_choice == self._t('face_enhance.image')
                if not is_image:
                    result_path, message = self._enhance_video(vid_file, model, device, progress)
                    return None, message, result_path
                return None, "", None
            
            # Connetti eventi (gestione separata per immagini e video)
            process_btn.click(
                fn=handle_image_process,
                inputs=[input_type, image_input, video_input, model_dropdown, device_dropdown],
                outputs=[image_output, status_output, video_output]
            ).then(
                fn=handle_video_process,
                inputs=[input_type, image_input, video_input, model_dropdown, device_dropdown],
                outputs=[image_output, status_output, video_output]
            )
            
            # Aggiorna device quando cambia
            def update_device(new_device):
                self.device_manager.set_device(new_device)
                return f"✓ {self._t('common.info')}: {new_device}"
            
            device_dropdown.change(
                fn=update_device,
                inputs=[device_dropdown],
                outputs=[status_output]
            )
