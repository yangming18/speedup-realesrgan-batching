"""
Lip Sync Tab
AI-powered lip synchronization using multiple models
"""
import os
import subprocess
import sys
import shutil
import yaml
from pathlib import Path
from typing import Optional, Callable
import logging
import gradio as gr
from PIL import Image
from utils.sadtalker_patch import patch_sadtalker_numpy_compatibility, check_if_patch_needed as check_sadtalker_patch
from utils.liveportrait_patch import patch_liveportrait_numpy_compatibility, check_if_patch_needed as check_liveportrait_patch
from utils.audio_countdown import add_countdown_to_audio, sync_audio_video_with_countdown
from utils.liveportrait_downloader import download_liveportrait_checkpoints, check_liveportrait_ready

logger = logging.getLogger(__name__)


def _is_image_file(file_path: str) -> bool:
    """Verifica se il file è un'immagine"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    return Path(file_path).suffix.lower() in image_extensions


def _get_audio_duration(audio_path: str) -> float:
    """Ottiene la durata dell'audio in secondi usando ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Impossibile ottenere durata audio: {e}")
    
    # Fallback: 5 secondi di default
    return 5.0


def _convert_image_to_video(image_path: str, audio_path: str, output_video_path: str, max_height: int = 480) -> bool:
    """
    Converte un'immagine in un video con la durata dell'audio.
    Mantiene il ratio dell'immagine limitando l'altezza a max_height (default 480p).
    
    Args:
        image_path: Path dell'immagine di input
        audio_path: Path dell'audio per calcolare la durata
        output_video_path: Path del video di output
        max_height: Altezza massima del video (default 480p)
    
    Returns:
        bool: True se la conversione ha successo
    """
    try:
        # Ottieni dimensioni originali
        with Image.open(image_path) as img:
            orig_width, orig_height = img.size
        
        # Calcola nuove dimensioni mantenendo il ratio
        if orig_height > max_height:
            ratio = max_height / orig_height
            new_height = max_height
            new_width = int(orig_width * ratio)
            # Assicurati che siano pari (richiesto da molti codec)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
        else:
            new_width = orig_width - (orig_width % 2)
            new_height = orig_height - (orig_height % 2)
        
        # Ottieni durata audio
        duration = _get_audio_duration(audio_path)
        
        logger.info(f"Conversione immagine → video: {orig_width}x{orig_height} → {new_width}x{new_height}, durata: {duration:.2f}s")
        
        # Crea video dall'immagine
        cmd = [
            'ffmpeg',
            '-y',  # Sovrascrivi se esiste
            '-loop', '1',  # Loop dell'immagine
            '-i', image_path,
            '-t', str(duration),  # Durata dal file audio
            '-vf', f'scale={new_width}:{new_height}',  # Ridimensiona mantenendo ratio
            '-c:v', 'libx264',  # Codec video
            '-pix_fmt', 'yuv420p',  # Formato pixel compatibile
            '-r', '25',  # 25 fps (standard per Video-Retalking)
            output_video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and Path(output_video_path).exists():
            logger.info(f"✅ Video creato: {output_video_path}")
            return True
        else:
            logger.error(f"Errore ffmpeg: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Errore conversione immagine→video: {e}")
        return False


# Funzioni per il download automatico dei checkpoint Video-Retalking
def _install_gdown():
    """Installa gdown se non presente"""
    try:
        import gdown
        return True
    except ImportError:
        logger.info("Installazione gdown per download da Google Drive...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import gdown
            return True
        except Exception as e:
            logger.error(f"Errore installazione gdown: {e}")
            return False

def _download_checkpoints_folder(checkpoints_dir, progress_callback=None):
    """Scarica l'intera cartella checkpoints da Google Drive"""
    import gdown
    
    # Folder ID della cartella checkpoints su Google Drive
    folder_url = "https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0"
    
    try:
        if progress_callback:
            progress_callback(0, "📥 Download cartella checkpoints da Google Drive (~2GB)...")
        
        logger.info("Tentativo download cartella Google Drive...")
        # Scarica l'intera cartella
        gdown.download_folder(folder_url, output=str(checkpoints_dir), quiet=False, use_cookies=False)
        return True
    except Exception as e:
        logger.error(f"Errore download cartella: {e}")
        return False

# Checkpoint richiesti per Video-Retalking
VIDEO_RETALKING_CHECKPOINTS = [
    "RetinaFace-R50.pth",
    "GPEN-BFR-512.pth",
    "DNet.pt",
    "ENet.pth",
    "LNet.pth",
    "face3d_pretrain_epoch_20.pth"
]


# Modelli disponibili con informazioni dettagliate
LIPSYNC_MODELS = {
    'wav2lip': {
        'name': 'Wav2Lip',
        'description': 'Modello base - Veloce e affidabile',
        'quality': 3,  # 3 stelle
        'speed': 5,    # 5 fulmini (molto veloce)
        'pros': ['Veloce', 'Stabile', 'Basso uso di memoria'],
        'cons': ['Qualità inferiore', 'Possibili artefatti'],
        'best_for': 'Preview rapide e video a bassa risoluzione',
        'repo': 'https://github.com/Rudrabha/Wav2Lip.git',
        'checkpoint': 'wav2lip.pth',
    },
    'wav2lip_gan': {
        'name': 'Wav2Lip GAN',
        'description': 'Versione migliorata - Qualità superiore',
        'quality': 4,  # 4 stelle
        'speed': 4,    # 4 fulmini (veloce)
        'pros': ['Qualità migliore', 'Meno artefatti', 'Relativamente veloce'],
        'cons': ['Uso memoria maggiore', 'Richiede GPU per performance ottimali'],
        'best_for': 'Produzione con buon compromesso qualità/velocità',
        'repo': 'https://github.com/Rudrabha/Wav2Lip.git',
        'checkpoint': 'wav2lip_gan.pth',
    },
    'sadtalker': {
        'name': 'SadTalker',
        'description': 'Animazione completa - Include espressioni e movimenti',
        'quality': 4,  # 4 stelle
        'speed': 2,    # 2 fulmini (lento)
        'pros': ['Anima espressioni facciali', 'Movimenti naturali', 'Risultati realistici', 'Mantiene scena completa'],
        'cons': ['Molto lento', 'Alto uso di memoria', 'Solo immagini statiche'],
        'best_for': 'Animazione di ritratti statici con massimo realismo',
        'repo': 'https://github.com/OpenTalker/SadTalker.git',
        'checkpoint': None,  # Usa multipli checkpoint
    },
    'video_retalking': {
        'name': 'Video-Retalking',
        'description': 'Massima qualità - Include miglioramento del viso',
        'quality': 5,  # 5 stelle
        'speed': 1,    # 1 fulmine (molto lento)
        'pros': ['Qualità superiore', 'Migliora anche il viso', 'Risultati professionali'],
        'cons': ['Molto lento', 'Richiede GPU potente', 'Setup complesso'],
        'best_for': 'Produzione professionale dove la qualità è prioritaria',
        'repo': 'https://github.com/OpenTalker/video-retalking.git',
        'checkpoint': None,
    },
    # 'thegargantuas_lipsync': {  # 🚧 IN FASE DI SVILUPPO - Temporaneamente disabilitato
    #     'name': 'TheGargantuas LipSync (in sviluppo)',
    #     'description': '🔥 Pipeline completa - Animazione corpo + lip sync perfetto',
    #     'quality': 5,  # 5 stelle (massima qualità)
    #     'speed': 2,    # 2 fulmini (medio-lento, doppia pipeline)
    #     'pros': ['Anima corpo + faccia', 'Lip sync perfetto (Wav2Lip GAN)', 'Movimenti naturali', 'Qualità professionale', 'Controllo completo con driving video'],
    #     'cons': ['Richiede driving video o webcam', 'Più lento (doppia pipeline)', 'Solo immagini statiche come source'],
    #     'best_for': 'Produzioni professionali con movimenti corpo naturali + lip sync perfetto. Ideale per talking portraits con gesti.',
    #     'repo': None,  # Pipeline combinata (LivePortrait + Wav2Lip GAN)
    #     'checkpoint': None,
    #     'requires_driving_video': True,  # Flag speciale
    # },
}


class LipSyncProcessor:
    """Gestisce il processing di lip-sync con modelli AI"""
    
    def __init__(
        self,
        model_name: str = 'wav2lip',
        device: Optional[str] = None,
        models_dir: Optional[Path] = None
    ):
        """
        Inizializza il processor lip-sync.
        
        Args:
            model_name: Nome del modello
            device: Device PyTorch ('cuda', 'cpu', 'mps', None per auto)
            models_dir: Directory per salvare i modelli
        """
        self.model_name = model_name.lower()
        if self.model_name not in LIPSYNC_MODELS:
            raise ValueError(f"Modello non supportato: {model_name}")
        
        self.device = device or self._detect_device()
        
        # Setup directories
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models" / "lipsync"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Wav2Lip e Wav2Lip_GAN condividono lo stesso repo
        repo_name = 'wav2lip' if self.model_name in ['wav2lip', 'wav2lip_gan'] else self.model_name
        self.model_repo_dir = self.models_dir / repo_name
        self.model_weights_dir = self.model_repo_dir / "checkpoints"
        
        # Store last error for debugging
        self.last_error = None
        
        # Applica patch NumPy per SadTalker se necessario
        if self.model_name == 'sadtalker' and self.model_repo_dir.exists():
            if check_sadtalker_patch(self.model_repo_dir):
                logger.info("Applicazione patch NumPy per SadTalker...")
                patch_sadtalker_numpy_compatibility(self.model_repo_dir)
        
        # Applica patch NumPy per LivePortrait se necessario
        if self.model_name == 'liveportrait' and self.model_repo_dir.exists():
            if check_liveportrait_patch(self.model_repo_dir):
                logger.info("Applicazione patch NumPy per LivePortrait...")
                patch_liveportrait_numpy_compatibility(self.model_repo_dir)
        
        logger.info(f"LipSyncProcessor inizializzato: {self.model_name} su {self.device}")
    
    def _detect_device(self) -> str:
        """Rileva automaticamente il device migliore disponibile"""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
        return 'cpu'
    
    def is_model_downloaded(self) -> bool:
        """Verifica se il modello è già scaricato"""
        if not self.model_repo_dir.exists():
            return False
        
        # Per wav2lip e wav2lip_gan controlla i checkpoint
        if self.model_name in ['wav2lip', 'wav2lip_gan']:
            checkpoint = LIPSYNC_MODELS[self.model_name]['checkpoint']
            return (self.model_weights_dir / checkpoint).exists()
        
        # Per sadtalker controlla i file essenziali
        elif self.model_name == 'sadtalker':
            checkpoints_dir = self.model_repo_dir / "checkpoints"
            essential_files = [
                checkpoints_dir / "SadTalker_V0.0.2_256.safetensors",
                checkpoints_dir / "SadTalker_V0.0.2_512.safetensors",
                checkpoints_dir / "mapping_00229-model.pth.tar"
            ]
            return all(f.exists() for f in essential_files)
        
        # Per video_retalking controlla il repository
        elif self.model_name == 'video_retalking':
            return (self.model_repo_dir / "inference.py").exists()
        
        return True
    
    def download_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Scarica il modello e le sue dipendenze.
        
        Args:
            progress_callback: Funzione chiamata con messaggi di progresso
            
        Returns:
            True se il download ha successo
        """
        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
        
        try:
            model_info = LIPSYNC_MODELS[self.model_name]
            
            # Clone repository se non esiste
            if not self.model_repo_dir.exists():
                log(f"📥 Clonazione repository {model_info['name']}...")
                subprocess.run(
                    ['git', 'clone', model_info['repo'], str(self.model_repo_dir)],
                    check=True,
                    capture_output=True
                )
                log(f"✅ Repository clonato")
            
            # Download modelli specifici
            if self.model_name in ['wav2lip', 'wav2lip_gan']:
                self._download_wav2lip_models(log)
            elif self.model_name == 'sadtalker':
                self._download_sadtalker_models(log)
            elif self.model_name == 'video_retalking':
                self._download_video_retalking_models(log)
            
            log(f"✅ Modello {model_info['name']} pronto")
            return True
            
        except Exception as e:
            log(f"❌ Errore durante il download: {str(e)}")
            logger.error(f"Errore download modello {self.model_name}", exc_info=True)
            return False
    
    def _download_file(self, url: str, destination: Path, log_callback: Callable) -> bool:
        """Scarica un file via HTTP usando requests"""
        try:
            import requests
        except ImportError:
            log_callback("❌ Libreria 'requests' non trovata. Installa con: pip install requests")
            return False

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix(destination.suffix + ".download")

        try:
            log_callback(f"⬇️ Scarico {destination.name}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded / total * 100
                        log_callback(f"   {downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB ({percent:.0f}%)")

            temp_path.replace(destination)
            log_callback(f"✅ {destination.name} scaricato")
            return True

        except Exception as e:
            log_callback(f"❌ Errore download {destination.name}: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return False

    def _download_wav2lip_models(self, log_callback: Callable):
        """Download dei modelli Wav2Lip"""
        self.model_weights_dir.mkdir(parents=True, exist_ok=True)

        log_callback("📥 Download modelli Wav2Lip...")

        models_to_download = {
            'wav2lip.pth': 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth',
            'wav2lip_gan.pth': 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth'
        }

        for model_name, url in models_to_download.items():
            model_path = self.model_weights_dir / model_name

            if model_path.exists():
                log_callback(f"✓ {model_name} già presente")
                continue

            if not self._download_file(url, model_path, log_callback):
                log_callback(f"⚠️ Scarica manualmente {model_name} e posizionalo in {self.model_weights_dir}")

    def _download_sadtalker_models(self, log_callback: Callable):
        """Download dei modelli SadTalker"""
        log_callback("📥 Download modelli SadTalker...")

        checkpoints_dir = self.model_repo_dir / "checkpoints"
        enhancer_dir = self.model_repo_dir / "gfpgan" / "weights"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        enhancer_dir.mkdir(parents=True, exist_ok=True)

        files_to_download = {
            checkpoints_dir / "mapping_00109-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
            checkpoints_dir / "mapping_00229-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
            checkpoints_dir / "SadTalker_V0.0.2_256.safetensors": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
            checkpoints_dir / "SadTalker_V0.0.2_512.safetensors": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
            enhancer_dir / "alignment_WFLW_4HG.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
            enhancer_dir / "detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            enhancer_dir / "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            enhancer_dir / "parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        }

        all_ok = True
        for destination, url in files_to_download.items():
            if destination.exists():
                log_callback(f"✓ {destination.name} già presente")
                continue

            if not self._download_file(url, destination, log_callback):
                all_ok = False

        if all_ok:
            log_callback("✅ Modelli SadTalker scaricati")
        else:
            log_callback("⚠️ Alcuni file SadTalker non sono stati scaricati")

    def _download_video_retalking_models(self, log_callback: Callable):
        """Download dei modelli Video-Retalking"""
        log_callback("📥 Download modelli Video-Retalking...")
        log_callback("⚠️ Segui le istruzioni nel README del repository")
    
    def process(
        self,
        image_or_video_path: str,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> bool:
        """
        Processa un'immagine/video con audio per generare lip-sync.
        
        Args:
            image_or_video_path: Path dell'immagine o video sorgente
            audio_path: Path del file audio
            output_path: Path del video output
            progress_callback: Funzione chiamata con (percentuale, messaggio)
            **kwargs: Parametri aggiuntivi specifici del modello
            
        Returns:
            True se il processing ha successo
        """
        if not self.is_model_downloaded():
            raise RuntimeError(f"Modello {self.model_name} non scaricato. Usa download_model() prima.")
        
        try:
            if self.model_name in ['wav2lip', 'wav2lip_gan']:
                return self._process_wav2lip(
                    image_or_video_path, audio_path, output_path, 
                    progress_callback, **kwargs
                )
            elif self.model_name == 'sadtalker':
                return self._process_sadtalker(
                    image_or_video_path, audio_path, output_path,
                    progress_callback, **kwargs
                )
            elif self.model_name == 'video_retalking':
                return self._process_video_retalking(
                    image_or_video_path, audio_path, output_path,
                    progress_callback, **kwargs
                )
            else:
                error_msg = f"Modello non supportato: {self.model_name}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, f"❌ {error_msg}")
                return False
        except Exception as e:
            error_msg = f"Errore durante processing {self.model_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, f"❌ Errore: {str(e)}")
            return False
    
    def _process_wav2lip(
        self,
        face_path: str,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable],
        **kwargs
    ) -> bool:
        """Processing con Wav2Lip"""
        if progress_callback:
            progress_callback(5, "🎬 Inizializzazione Wav2Lip...")
        
        # Converti path in assoluti e verifica esistenza
        face_path_abs = Path(face_path).resolve()
        audio_path_abs = Path(audio_path).resolve()
        
        if not face_path_abs.exists():
            error_msg = f"File input non trovato: {face_path_abs}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ File video/immagine non trovato")
            return False
        
        if not audio_path_abs.exists():
            error_msg = f"File audio non trovato: {audio_path_abs}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ File audio non trovato")
            return False
        
        # WORKAROUND CRITICO: Wav2Lip inference.py NON quota correttamente i path con spazi
        # Anche se usiamo nomi senza spazi, il path del progetto "Video Editor" causa problemi
        # Soluzione: Usa cartella temp di sistema (path garantito senza spazi su macOS/Linux)
        import tempfile
        
        system_temp = Path(tempfile.gettempdir()) / "wav2lip_temp"
        system_temp.mkdir(exist_ok=True)
        
        temp_output_raw = system_temp / "result_raw.avi"
        temp_face = system_temp / f"input_face{face_path_abs.suffix}"
        temp_audio = system_temp / f"input_audio{audio_path_abs.suffix}"
        
        # Pulisci file temp precedenti
        for temp_file in [temp_output_raw, temp_face, temp_audio]:
            if temp_file.exists():
                temp_file.unlink()
        
        # Copia i file nella temp di sistema
        shutil.copy2(face_path_abs, temp_face)
        shutil.copy2(audio_path_abs, temp_audio)
        
        logger.info(f"File copiati in system temp: {system_temp}")
        
        # Verifica checkpoint e inference script
        checkpoint = LIPSYNC_MODELS[self.model_name]['checkpoint']
        checkpoint_path = self.model_weights_dir / checkpoint
        
        if not checkpoint_path.exists():
            if progress_callback:
                progress_callback(0, f"❌ Checkpoint non trovato: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
        
        inference_script = self.model_repo_dir / "inference.py"
        if not inference_script.exists():
            if progress_callback:
                progress_callback(0, "❌ inference.py non trovato")
            raise FileNotFoundError(f"inference.py non trovato in {self.model_repo_dir}")
        
        cmd = [
            sys.executable, str(inference_script),
            '--checkpoint_path', str(checkpoint_path),
            '--face', str(temp_face),
            '--audio', str(temp_audio),
            '--outfile', str(temp_output_raw),
        ]
        
        if kwargs.get('resize_factor'):
            cmd.extend(['--resize_factor', str(kwargs['resize_factor'])])
        if kwargs.get('nosmooth'):
            cmd.append('--nosmooth')
        
        # Log comando per debugging
        logger.info(f"Esecuzione comando Wav2Lip: {' '.join(cmd)}")
        
        if progress_callback:
            progress_callback(10, "🎭 Generazione lip-sync...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.model_repo_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Limit output length but keep important info
                stderr = result.stderr if len(result.stderr) < 2000 else result.stderr[-2000:]
                stdout = result.stdout if len(result.stdout) < 500 else result.stdout[-500:]
                error_msg = f"Wav2Lip failed:\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, f"❌ Errore Wav2Lip - vedi dettagli sotto")
                return False
            
            if not temp_output_raw.exists():
                # Include command output even if returncode was 0
                stderr = result.stderr if len(result.stderr) < 2000 else result.stderr[-2000:]
                stdout = result.stdout if len(result.stdout) < 1000 else result.stdout[-1000:]
                error_msg = f"File temporaneo non creato: {temp_output_raw}\n\nComando eseguito:\n{' '.join(cmd)}\n\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, f"❌ File temp non creato - vedi dettagli sotto")
                return False
            
            # Aggiungi audio al video finale
            if progress_callback:
                progress_callback(95, "🎵 Aggiunta audio finale...")
            self._add_audio_to_video(str(temp_output_raw), str(temp_audio), output_path)
            
            if not Path(output_path).exists():
                error_msg = f"File output non creato: {output_path}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, f"❌ Output finale non creato")
                return False
            
            # Pulizia file temporanei
            for temp_file in [temp_output_raw, temp_face, temp_audio]:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Impossibile eliminare file temp {temp_file}: {cleanup_error}")
            
            if progress_callback:
                progress_callback(100, "✅ Completato!")
            return True
            
        except Exception as e:
            error_msg = f"Exception running Wav2Lip: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, f"❌ Eccezione: {str(e)[:100]}")
            
            # Pulizia file temporanei anche in caso di errore
            try:
                for temp_file in [temp_output_raw, temp_face, temp_audio]:
                    if temp_file.exists():
                        temp_file.unlink()
            except:
                pass
            
            return False
    
    def _add_audio_to_video(self, video_path: str, audio_path: str, output_path: str):
        """Aggiungi audio a video usando ffmpeg"""
        logger.info(f"Aggiunta audio: {video_path} + {audio_path} -> {output_path}")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise RuntimeError(f"Errore aggiunta audio: {result.stderr[:200]}")
    
    def _process_sadtalker(
        self,
        source_image: str,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable],
        **kwargs
    ) -> bool:
        """Processing con SadTalker"""
        if progress_callback:
            progress_callback(10, "🎬 Inizializzazione SadTalker...")
        
        inference_script = self.model_repo_dir / "inference.py"
        checkpoints_dir = self.model_repo_dir / "checkpoints"
        
        # Verifica file esistenza
        if not Path(source_image).exists():
            error_msg = f"Immagine non trovata: {source_image}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ Immagine sorgente non trovata")
            return False
        
        if not Path(audio_path).exists():
            error_msg = f"Audio non trovato: {audio_path}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ File audio non trovato")
            return False
        
        if not inference_script.exists():
            error_msg = f"inference.py non trovato in {self.model_repo_dir}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ Script SadTalker non trovato")
            return False
        
        # Assicura che la directory di output esista e usa path assoluto
        output_dir = Path(output_path).parent.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory output (assoluto): {output_dir}")
        
        # Converti tutti i path in assoluti per evitare problemi con cwd
        source_image_abs = str(Path(source_image).resolve())
        audio_path_abs = str(Path(audio_path).resolve())
        
        cmd = [
            sys.executable,
            str(inference_script),
            '--driven_audio', audio_path_abs,
            '--source_image', source_image_abs,
            '--result_dir', str(output_dir),
            '--checkpoint_dir', str(checkpoints_dir),
            '--still',  # Mantiene l'immagine completa invece di crop sul viso
            '--preprocess', 'full',  # Usa l'immagine completa senza crop
        ]
        
        logger.info(f"Esecuzione comando SadTalker: {' '.join(cmd)}")
        
        if progress_callback:
            progress_callback(30, "🎭 Processing con SadTalker...")
        
        # Usa Popen per leggere output in tempo reale
        import re
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.model_repo_dir),
            bufsize=1,
            universal_newlines=True
        )
        
        # Raccogli output
        stdout_lines = []
        stderr_lines = []
        
        # Leggi stderr in tempo reale (messaggi informativi di SadTalker)
        for line in process.stderr:
            stderr_lines.append(line)
            line_stripped = line.strip()
            
            if line_stripped:
                logger.info(f"SadTalker: {line_stripped}")
                
                # Aggiorna progress in base ai messaggi chiave
                if '3DMM Extraction' in line or 'coeffs' in line.lower():
                    if progress_callback:
                        progress_callback(40, "🔍 Estrazione features 3D...")
                elif 'audio2pose' in line.lower() or 'audio2exp' in line.lower():
                    if progress_callback:
                        progress_callback(50, "🎵 Analisi audio...")
                elif 'face render' in line.lower() or 'generating' in line.lower():
                    if progress_callback:
                        progress_callback(70, "🎬 Rendering frames...")
                elif 'saving' in line.lower() or 'final' in line.lower():
                    if progress_callback:
                        progress_callback(90, "💾 Salvataggio video...")
                
                # Cerca progress bar percentuali
                percent_match = re.search(r'(\d+)%', line)
                if percent_match:
                    pct = int(percent_match.group(1))
                    if pct % 20 == 0:  # Aggiorna ogni 20%
                        if progress_callback:
                            progress_callback(40 + int(pct * 0.5), f"⏳ Elaborazione: {pct}%")
        
        # Leggi stdout rimanente
        stdout_remaining = process.stdout.read()
        if stdout_remaining:
            stdout_lines.append(stdout_remaining)
        
        # Attendi completamento
        returncode = process.wait()
        
        if returncode == 0:
            # SadTalker crea il file in una sottocartella con timestamp
            # Cerca il file generato dall'output
            generated_file = None
            for line in output_lines:
                if 'The generated video is named:' in line:
                    # Estrai il path dal messaggio
                    parts = line.split('The generated video is named:')
                    if len(parts) > 1:
                        generated_file = Path(parts[1].strip())
                        break
            
            # Se non trovato nel messaggio, cerca nella cartella result_dir
            if not generated_file or not generated_file.exists():
                result_dir = Path(output_path).parent
                # Cerca file .mp4 nella result_dir e sue sottocartelle
                mp4_files = list(result_dir.glob('**/*.mp4'))
                if mp4_files:
                    # Prendi il file più recente
                    generated_file = max(mp4_files, key=lambda p: p.stat().st_mtime)
            
            if generated_file and generated_file.exists():
                # Sposta/copia il file nella posizione prevista
                try:
                    shutil.move(str(generated_file), output_path)
                    logger.info(f"File spostato da {generated_file} a {output_path}")
                    
                    # Pulisci cartelle temp create da SadTalker
                    if generated_file.parent != Path(output_path).parent:
                        try:
                            shutil.rmtree(generated_file.parent)
                        except Exception as e:
                            logger.warning(f"Impossibile pulire cartella temp: {e}")
                    
                    if progress_callback:
                        progress_callback(100, "✅ Completato!")
                    return True
                except Exception as e:
                    error_msg = f"Errore nello spostamento del file: {e}"
                    logger.error(error_msg)
                    self.last_error = error_msg
                    if progress_callback:
                        progress_callback(0, "❌ Errore spostamento file")
                    return False
            else:
                # File non trovato
                output = '\n'.join(output_lines[-50:])  # ultime 50 righe
                error_msg = f"SadTalker completato ma file non trovato.\nCercavo: {output_path}\n\nOutput:\n{output}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, "❌ File output non trovato - vedi sotto")
                return False
        else:
            # Errore durante processing
            output = '\n'.join(output_lines[-100:])  # ultime 100 righe
            error_msg = f"SadTalker fallito (exit code {returncode}):\n\n{output}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, f"❌ Errore SadTalker (code {returncode}) - vedi dettagli")
            return False
    
    def _process_video_retalking(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable],
        **kwargs
    ) -> bool:
        """Processing con Video-Retalking"""
        if progress_callback:
            progress_callback(10, "🎬 Inizializzazione Video-Retalking...")
        
        inference_script = self.model_repo_dir / "inference.py"
        
        # Verifica file esistenza
        if not Path(video_path).exists():
            error_msg = f"Video/immagine non trovato: {video_path}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ File sorgente non trovato")
            return False
        
        if not Path(audio_path).exists():
            error_msg = f"Audio non trovato: {audio_path}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ File audio non trovato")
            return False
        
        # Gestisci immagini: converte in video temporaneo
        temp_video_path = None
        actual_video_path = video_path
        
        if _is_image_file(video_path):
            logger.info(f"Immagine rilevata: {video_path}, conversione in video...")
            if progress_callback:
                progress_callback(15, "🖼️ Conversione immagine → video (480p)...")
            
            # Crea video temporaneo nella stessa directory del file di output
            from utils.temp_manager import TempManager
            temp_manager = TempManager()
            temp_video_path = temp_manager.get_temp_file_path("video_retalking_temp.mp4")
            
            if not _convert_image_to_video(video_path, audio_path, temp_video_path, max_height=480):
                error_msg = "Impossibile convertire l'immagine in video. Verifica che ffmpeg sia installato."
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, "❌ Conversione immagine fallita")
                return False
            
            actual_video_path = temp_video_path
            logger.info(f"✅ Video temporaneo creato: {actual_video_path}")
            if progress_callback:
                progress_callback(20, "✅ Video creato, avvio processing...")
        
        if not inference_script.exists():
            error_msg = f"inference.py non trovato in {self.model_repo_dir}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, "❌ Script Video-Retalking non trovato")
            # Cleanup temp file
            if temp_video_path and Path(temp_video_path).exists():
                Path(temp_video_path).unlink()
            return False
        
        # Verifica e scarica automaticamente i checkpoint se mancanti
        checkpoints_dir = self.model_repo_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        missing_checkpoints = [cp for cp in VIDEO_RETALKING_CHECKPOINTS 
                              if not (checkpoints_dir / cp).exists()]
        
        if missing_checkpoints:
            logger.info(f"Video-Retalking: {len(missing_checkpoints)} checkpoint mancanti, avvio download automatico...")
            
            if progress_callback:
                progress_callback(0, f"📥 Download modelli Video-Retalking ({len(missing_checkpoints)}/6 mancanti, ~2GB)...")
            
            # Installa gdown se necessario
            if not _install_gdown():
                error_msg = (
                    f"❌ Impossibile installare 'gdown' per il download automatico\n\n"
                    f"📥 DOWNLOAD MANUALE RICHIESTO:\n"
                    f"  1. Vai su: https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0\n"
                    f"  2. Scarica tutti i file (~2GB)\n"
                    f"  3. Mettili in: {checkpoints_dir}\n\n"
                    f"📖 Guida: {self.model_repo_dir / 'INSTALL_MODELS.md'}"
                )
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, "❌ Download automatico fallito - download manuale richiesto")
                # Cleanup temp file
                if temp_video_path and Path(temp_video_path).exists():
                    Path(temp_video_path).unlink()
                return False
            
            # Scarica l'intera cartella checkpoints
            if progress_callback:
                progress_callback(50, "📥 Download in corso da Google Drive (~2GB)...")
            
            download_success = _download_checkpoints_folder(checkpoints_dir, progress_callback)
            
            if not download_success:
                error_msg = (
                    f"❌ Download automatico fallito (limitazioni Google Drive)\n\n"
                    f"⚠️ I file su Google Drive potrebbero avere restrizioni di accesso.\n\n"
                    f"📥 DOWNLOAD MANUALE (soluzione più affidabile):\n"
                    f"  1. Apri il browser e vai su:\n"
                    f"     https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0\n"
                    f"  2. Seleziona tutti i file nella cartella\n"
                    f"  3. Click destro → Download\n"
                    f"  4. Metti tutti i file in: {checkpoints_dir}\n\n"
                    f"📖 Guida dettagliata: {self.model_repo_dir / 'INSTALL_MODELS.md'}\n\n"
                    f"💡 ALTERNATIVA: Script Python (può funzionare):\n"
                    f"   cd {self.model_repo_dir}\n"
                    f"   python download_checkpoints.py"
                )
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, "❌ Download fallito - usa il browser per scaricare manualmente")
                # Cleanup temp file
                if temp_video_path and Path(temp_video_path).exists():
                    Path(temp_video_path).unlink()
                return False
            
            # Verifica che tutti i file siano stati scaricati
            still_missing = [cp for cp in VIDEO_RETALKING_CHECKPOINTS 
                           if not (checkpoints_dir / cp).exists()]
            
            if still_missing:
                error_msg = (
                    f"⚠️ Download incompleto: {len(still_missing)}/{len(VIDEO_RETALKING_CHECKPOINTS)} file ancora mancanti\n\n"
                    f"File mancanti:\n" + "\n".join(f"  • {cp}" for cp in still_missing) + "\n\n"
                    f"📥 Scarica i file mancanti dal browser:\n"
                    f"  https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0\n\n"
                    f"📖 Guida: {self.model_repo_dir / 'INSTALL_MODELS.md'}"
                )
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, f"❌ {len(still_missing)} file ancora mancanti")
                # Cleanup temp file
                if temp_video_path and Path(temp_video_path).exists():
                    Path(temp_video_path).unlink()
                return False
            
            logger.info(f"✅ Tutti i checkpoint Video-Retalking scaricati con successo!")
            if progress_callback:
                progress_callback(100, "✅ Download completato, avvio elaborazione...")
        
        # Costruisci comando con il video (o video temporaneo se era un'immagine)
        cmd = [
            sys.executable,
            str(inference_script),
            '--face', str(actual_video_path),  # Converti Path a str
            '--audio', str(audio_path),
            '--outfile', str(output_path),
        ]
        
        logger.info(f"Esecuzione comando Video-Retalking: {' '.join(cmd)}")
        
        if progress_callback:
            progress_callback(30, "🎭 Processing con Video-Retalking...")
        
        try:
            # Usa Popen per leggere output in tempo reale
            import re
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.model_repo_dir),
                bufsize=1,
                universal_newlines=True
            )
            
            # Raccogli output per eventuali errori
            stdout_lines = []
            stderr_lines = []
            
            # Mappa step → progress %
            step_progress = {
                'Step 0': 35,
                'Step 1': 45,
                'Step 2': 55,
                'Step 3': 65,
                'Step 4': 70,
                'Step 5': 75,
                'Step 6': 80,
                'Step 7': 90,
                'Step 8': 95
            }
            
            # Leggi stderr in tempo reale (qui ci sono i progressi)
            for line in process.stderr:
                stderr_lines.append(line)
                line_stripped = line.strip()
                
                # Logga solo le linee informative (no barre progresso ripetute)
                if line_stripped and not line_stripped.startswith('\r') and '[A' not in line:
                    logger.info(f"Video-Retalking: {line_stripped}")
                
                # Cerca step markers
                for step, progress in step_progress.items():
                    if step in line:
                        step_desc = line_stripped.split(']')[1].strip() if ']' in line_stripped else step
                        if progress_callback:
                            progress_callback(progress, f"🎬 {step_desc[:50]}")
                        break
                
                # Cerca progress bar percentuali
                if '%|' in line:
                    # Estrai percentuale (es: "landmark Det::  97%|")
                    match = re.search(r'(\d+)%', line)
                    if match:
                        pct = int(match.group(1))
                        # Scala la percentuale all'interno del range corrente
                        if 'landmark' in line.lower():
                            scaled = 45 + (pct * 0.10)  # 45-55%
                        elif '3dmm' in line.lower() or 'extraction' in line.lower():
                            scaled = 55 + (pct * 0.10)  # 55-65%
                        elif 'lip' in line.lower() or 'synthesis' in line.lower():
                            scaled = 80 + (pct * 0.15)  # 80-95%
                        else:
                            scaled = 30 + (pct * 0.60)  # Generico
                        
                        if progress_callback and pct % 10 == 0:  # Aggiorna ogni 10%
                            progress_callback(int(scaled), f"⏳ Elaborazione: {pct}%")
            
            # Leggi stdout rimanente
            stdout_remaining = process.stdout.read()
            if stdout_remaining:
                stdout_lines.append(stdout_remaining)
            
            # Attendi completamento
            returncode = process.wait()
            
            if returncode == 0:
                # Verifica che il file di output esista
                if Path(output_path).exists():
                    if progress_callback:
                        progress_callback(100, "✅ Completato!")
                    # Cleanup temp file se esiste
                    if temp_video_path and Path(temp_video_path).exists():
                        try:
                            Path(temp_video_path).unlink()
                            logger.info(f"🗑️ File temporaneo rimosso: {temp_video_path}")
                        except Exception as e:
                            logger.warning(f"Impossibile rimuovere file temporaneo: {e}")
                    return True
                else:
                    # File non creato anche se returncode=0
                    stderr = ''.join(stderr_lines[-50:])  # Ultimi 50 linee
                    stdout = ''.join(stdout_lines[-20:])  # Ultimi 20 linee
                    error_msg = f"Video-Retalking completato ma file non creato: {output_path}\n\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
                    logger.error(error_msg)
                    self.last_error = error_msg
                    if progress_callback:
                        progress_callback(0, "❌ File output non creato - vedi sotto")
                    # Cleanup temp file
                    if temp_video_path and Path(temp_video_path).exists():
                        Path(temp_video_path).unlink()
                    return False
            else:
                # Limit output length but keep important info
                stderr = ''.join(stderr_lines[-50:])
                stdout = ''.join(stdout_lines[-20:])
                error_msg = f"Video-Retalking fallito (exit code {returncode}):\n\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, f"❌ Errore Video-Retalking (code {returncode})")
                # Cleanup temp file
                if temp_video_path and Path(temp_video_path).exists():
                    Path(temp_video_path).unlink()
                return False
        except Exception as e:
            error_msg = f"Eccezione durante Video-Retalking: {str(e)}"
            logger.error(error_msg)
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, f"❌ Errore: {str(e)}")
            # Cleanup temp file
            if temp_video_path and Path(temp_video_path).exists():
                Path(temp_video_path).unlink()
            return False
    
    def _process_liveportrait(
        self,
        source_image: str,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable],
        **kwargs
    ) -> bool:
        """Processing con LivePortrait"""
        try:
            if progress_callback:
                progress_callback(10, "🎬 Inizializzazione LivePortrait...")
            
            inference_script = self.model_repo_dir / "inference.py"
            logger.info(f"LivePortrait model_repo_dir: {self.model_repo_dir}")
            logger.info(f"LivePortrait inference_script: {inference_script}")
            
            # Verifica file esistenza
            if not Path(source_image).exists():
                error_msg = f"Immagine non trovata: {source_image}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, "❌ Immagine sorgente non trovata")
                return False
            
            if not Path(audio_path).exists():
                error_msg = f"Audio non trovato: {audio_path}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, "❌ File audio non trovato")
                return False
            
            if not inference_script.exists():
                error_msg = f"inference.py non trovato in {self.model_repo_dir}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, "❌ Script LivePortrait non trovato")
                return False
            
            cmd = [
                sys.executable,
                str(inference_script),
                '--source_image', source_image,
                '--driving_audio', audio_path,
                '--output', output_path,
            ]
            
            logger.info(f"Esecuzione comando LivePortrait: {' '.join(cmd)}")
            logger.info(f"CWD: {self.model_repo_dir}")
            
            if progress_callback:
                progress_callback(30, "🎭 Processing con LivePortrait...")
            
            # Usa Popen per leggere output in tempo reale
            import re
            logger.info("Avvio subprocess LivePortrait...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.model_repo_dir),
                bufsize=1,
                universal_newlines=True
            )
            logger.info(f"Subprocess avviato, PID: {process.pid}")
            
            # Raccogli output
            stdout_lines = []
            stderr_lines = []
            last_progress = 0
            
            # Leggi stderr in tempo reale
            logger.info("Lettura stderr in tempo reale...")
            for line in process.stderr:
                stderr_lines.append(line)
                line_stripped = line.strip()
                
                if line_stripped:
                    logger.info(f"LivePortrait STDERR: {line_stripped}")
                    
                    # Aggiorna progress in base ai messaggi chiave
                    if 'detect' in line_stripped.lower() or 'face' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(40, "🔍 Rilevamento viso...")
                    elif 'extract' in line_stripped.lower() or 'feat' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(50, "📊 Estrazione features...")
                    elif 'generat' in line_stripped.lower() or 'render' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(70, "🎬 Rendering video...")
                    elif 'writ' in line_stripped.lower() or 'sav' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(90, "💾 Salvataggio...")
                    
                    # Cerca progress bar percentuali
                    percent_match = re.search(r'(\d+)%', line)
                    if percent_match:
                        pct = int(percent_match.group(1))
                        # Aggiorna ogni 10%
                        if pct >= last_progress + 10:
                            last_progress = pct
                            if progress_callback:
                                progress_callback(30 + int(pct * 0.6), f"⏳ Elaborazione: {pct}%")
            
            # Leggi stdout rimanente
            logger.info("Lettura stdout...")
            stdout_remaining = process.stdout.read()
            if stdout_remaining:
                stdout_lines.append(stdout_remaining)
                logger.info(f"LivePortrait STDOUT: {stdout_remaining}")
            
            # Attendi completamento
            logger.info("Attendo completamento processo...")
            returncode = process.wait()
            logger.info(f"Processo terminato con exit code: {returncode}")
            
            # Crea output_lines per compatibilità con il codice esistente
            output_lines = stdout_lines + [f"STDERR: {line}" for line in stderr_lines]
            
            # Log completo dell'output
            logger.info(f"Output completo ({len(output_lines)} linee):")
            for i, line in enumerate(output_lines[-20:]):  # Log ultime 20 linee
                logger.info(f"  [{i}] {line}")
            
            if returncode == 0:
                # LivePortrait potrebbe creare il file con nome diverso, cerchiamolo
                logger.info(f"Controllo esistenza file output: {output_path}")
                if Path(output_path).exists():
                    logger.info("File output trovato!")
                    if progress_callback:
                        progress_callback(100, "✅ Completato!")
                    return True
                else:
                    # Cerca file generato nella cartella output
                    logger.warning(f"File output non trovato, cerco nella cartella...")
                    output_dir = Path(output_path).parent
                    logger.info(f"Cerco in: {output_dir}")
                    mp4_files = list(output_dir.glob('**/*.mp4'))
                    logger.info(f"Trovati {len(mp4_files)} file mp4")
                    if mp4_files:
                        generated_file = max(mp4_files, key=lambda p: p.stat().st_mtime)
                        logger.info(f"File più recente: {generated_file}")
                        try:
                            shutil.move(str(generated_file), output_path)
                            logger.info(f"File spostato da {generated_file} a {output_path}")
                            if progress_callback:
                                progress_callback(100, "✅ Completato!")
                            return True
                        except Exception as e:
                            error_msg = f"Errore nello spostamento del file: {e}"
                            logger.error(error_msg)
                            logger.exception("Stack trace:")
                            self.last_error = error_msg
                            if progress_callback:
                                progress_callback(0, "❌ Errore spostamento file")
                            return False
                    else:
                        output = '\n'.join(output_lines[-50:])
                        error_msg = f"LivePortrait completato ma file non trovato: {output_path}\n\nOutput:\n{output}"
                        logger.error(error_msg)
                        self.last_error = error_msg
                        if progress_callback:
                            progress_callback(0, "❌ File output non trovato")
                        return False
            else:
                output = '\n'.join(output_lines[-100:])
                error_msg = f"LivePortrait fallito (exit code {returncode}):\n\n{output}"
                logger.error(error_msg)
                self.last_error = error_msg
                if progress_callback:
                    progress_callback(0, f"❌ Errore LivePortrait (code {returncode})")
                return False
        
        except Exception as e:
            error_msg = f"Eccezione Python in _process_liveportrait: {e}"
            logger.error(error_msg)
            logger.exception("Stack trace completo:")
            self.last_error = error_msg
            if progress_callback:
                progress_callback(0, f"❌ Errore: {str(e)}")
            return False


class LipSyncTab:
    """Handles lip sync functionality with multiple AI models"""
    
    def __init__(self, temp_manager, device_manager, i18n=None):
        self.temp_manager = temp_manager
        self.device_manager = device_manager
        self.processor = None
        self.current_model = None
        self.i18n = i18n
    
    def _t(self, key, **kwargs):
        """Helper per traduzione con fallback"""
        if self.i18n:
            return self.i18n.t(key, **kwargs)
        return key
    
    def _get_model_info_html(self, model_name: str) -> str:
        """Genera HTML con le informazioni dettagliate del modello"""
        if not model_name:
            return ""
        
        model = LIPSYNC_MODELS[model_name]
        
        # Traduzioni - usa fallback se le chiavi non esistono
        if self.i18n:
            try:
                # Prova ad accedere alle traduzioni strutturate
                model_translations = self.i18n.translations.get('lipsync', {}).get('models', {}).get(model_name, {})
                description = model_translations.get('description', model['description'])
                pros = model_translations.get('pros', model['pros'])
                cons = model_translations.get('cons', model['cons'])
                best_for = model_translations.get('best_for', model['best_for'])
            except:
                # Fallback ai valori hardcoded
                description = model['description']
                pros = model['pros']
                cons = model['cons']
                best_for = model['best_for']
        else:
            # Nessun i18n, usa valori hardcoded
            description = model['description']
            pros = model['pros']
            cons = model['cons']
            best_for = model['best_for']
        
        # Stelle per qualità
        stars = "⭐" * model['quality'] + "☆" * (5 - model['quality'])
        
        # Fulmini per velocità
        bolts = "⚡" * model['speed'] + "🔌" * (5 - model['speed'])
        
        # Pro e cons come liste HTML
        pros_html = "".join([f"<li style='color: #4CAF50;'>✓ {pro}</li>" for pro in pros])
        cons_html = "".join([f"<li style='color: #FF5252;'>✗ {con}</li>" for con in cons])
        
        html = f"""
        <div style="background: linear-gradient(135deg, #2D2D2D 0%, #1A1A1A 100%); 
                    padding: 20px; border-radius: 15px; margin: 10px 0; 
                    border: 2px solid #FFA500;">
            <h2 style="color: #FFA500; margin-top: 0;">{model['name']}</h2>
            <p style="color: #CCCCCC; font-size: 1.1em;">{description}</p>
            
            <div style="display: flex; gap: 30px; margin: 15px 0;">
                <div>
                    <strong style="color: #FFB833;">{self._t('lipsync.model_quality')}</strong>
                    <div style="font-size: 1.5em;">{stars}</div>
                </div>
                <div>
                    <strong style="color: #FFB833;">{self._t('lipsync.model_speed')}</strong>
                    <div style="font-size: 1.5em;">{bolts}</div>
                </div>
            </div>
            
            <div style="margin: 15px 0;">
                <strong style="color: #FFB833;">{self._t('lipsync.model_pros')}</strong>
                <ul style="margin: 10px 0; padding-left: 20px;">{pros_html}</ul>
            </div>
            
            <div style="margin: 15px 0;">
                <strong style="color: #FFB833;">{self._t('lipsync.model_cons')}</strong>
                <ul style="margin: 10px 0; padding-left: 20px;">{cons_html}</ul>
            </div>
            
            <div style="background: rgba(255, 165, 0, 0.1); padding: 10px; 
                        border-radius: 8px; margin-top: 15px;">
                <strong style="color: #FFB833;">{self._t('lipsync.model_best_for')}</strong>
                <p style="color: #FFFFFF; margin: 5px 0;">{best_for}</p>
            </div>
        </div>
        """
        
        return html
    
    def _process_audio_video_pipeline(
        self,
        input_file,
        audio_file,
        driving_video,
        device,
        resize_factor,
        nosmooth,
        progress
    ):
        """
        Pipeline: LivePortrait (IMG + DRIVING_VIDEO) → Wav2Lip GAN (VIDEO + AUDIO)
        
        Returns:
            tuple: (output_video_path, info_message)
        """
        try:
            models_dir = Path(__file__).parent.parent / "models" / "lipsync"
            liveportrait_dir = models_dir / "liveportrait"
            pretrained_weights_dir = liveportrait_dir / "pretrained_weights"
            
            # Check and download LivePortrait checkpoints if needed
            if not check_liveportrait_ready(liveportrait_dir):
                progress(0.02, desc="📥 Downloading LivePortrait checkpoints (first time only, ~500MB)...")
                logger.info("LivePortrait checkpoints missing, downloading...")
                
                success = download_liveportrait_checkpoints(pretrained_weights_dir)
                if not success:
                    return None, ("❌ Failed to download LivePortrait checkpoints.\n\n"
                                "Try manually:\n"
                                "pip install -U 'huggingface_hub[cli]'\n"
                                "huggingface-cli download KlingTeam/LivePortrait --local-dir models/lipsync/liveportrait/pretrained_weights")
                
                progress(0.05, desc="✅ Checkpoints downloaded!")
            
            # Step 1: LivePortrait motion transfer
            progress(0.05, desc="🎬 Step 1/2: LivePortrait motion transfer...")
            
            # LivePortrait output path
            liveportrait_output = self.temp_manager.get_temp_file_path("liveportrait_motion.mp4")
            
            def liveportrait_progress(percent, msg):
                # Map to 10-45% of total progress
                progress((10 + percent * 0.35) / 100, desc=f"🎬 LivePortrait: {msg}")
            
            # Call LivePortrait directly (no processor needed)
            logger.info(f"Running LivePortrait: {input_file} + {driving_video} → {liveportrait_output}")
            success = self._run_liveportrait_inference(
                source_image=input_file,
                driving_video=driving_video,
                output_path=str(liveportrait_output),
                device=device,
                progress_callback=liveportrait_progress
            )
            
            if not success:
                return None, "❌ Errore LivePortrait motion transfer"
            
            if not Path(liveportrait_output).exists():
                return None, "❌ LivePortrait non ha generato il video intermedio"
            
            logger.info(f"✅ LivePortrait completato: {liveportrait_output}")
            
            # Step 2: Wav2Lip GAN lip sync
            progress(0.5, desc="💋 Step 2/2: Wav2Lip GAN lip sync...")
            
            wav2lip_processor = LipSyncProcessor(
                model_name='wav2lip_gan',
                device=device,
                models_dir=models_dir
            )
            
            # Download Wav2Lip GAN if needed
            if not wav2lip_processor.is_model_downloaded():
                progress(0.55, desc="📥 Download Wav2Lip GAN...")
                
                download_log = []
                def log_download(msg):
                    download_log.append(msg)
                    progress(0.55, desc=msg)
                
                success = wav2lip_processor.download_model(progress_callback=log_download)
                if not success:
                    return None, "❌ Errore download Wav2Lip GAN:\n" + "\n".join(download_log[-10:])
            
            # Process with Wav2Lip GAN
            final_output = self.temp_manager.get_temp_file_path("lipsync_final.mp4")
            
            def wav2lip_progress(percent, msg):
                # Map to 55-95% of total progress
                progress((55 + percent * 0.4) / 100, desc=f"💋 Wav2Lip GAN: {msg}")
            
            success = wav2lip_processor.process(
                image_or_video_path=str(liveportrait_output),
                audio_path=audio_file,
                output_path=str(final_output),
                progress_callback=wav2lip_progress,
                resize_factor=resize_factor,
                nosmooth=nosmooth
            )
            
            if success and Path(final_output).exists():
                progress(1.0, desc="✅ Pipeline completata!")
                return str(final_output), (
                    "✅ Pipeline completata con successo!\n\n"
                    "🎬 Step 1: LivePortrait → motion transfer completato\n"
                    "💋 Step 2: Wav2Lip GAN → lip sync completato\n\n"
                    "💡 Suggerimento: Per migliorare ulteriormente la qualità, "
                    "usa il tab Upscaler con RealESRGAN x2 o x4"
                )
            else:
                error_details = ""
                if wav2lip_processor.last_error:
                    error_details = f"\n\nDettagli:\n{wav2lip_processor.last_error}"
                return None, f"❌ Errore Wav2Lip GAN{error_details}"
            
        except Exception as e:
            logger.error(f"Errore nella pipeline audio+video: {e}", exc_info=True)
            return None, f"❌ Errore pipeline: {str(e)}"
    
    def _run_liveportrait_inference(
        self,
        source_image,
        driving_video,
        output_path,
        device,
        progress_callback
    ):
        """
        Esegue LivePortrait inference con argomenti corretti
        LivePortrait usa --source (immagine) e --driving (video) come da ArgumentConfig
        
        NOTE: LivePortrait uses Conv3D which is NOT supported on MPS (Apple Silicon GPU)
        We force CPU execution for MPS devices
        """
        try:
            models_dir = Path(__file__).parent.parent / "models" / "lipsync"
            liveportrait_dir = models_dir / "liveportrait"
            inference_script = liveportrait_dir / "inference.py"
            
            if not inference_script.exists():
                logger.error(f"inference.py non trovato: {inference_script}")
                return False
            
            # Warn if using MPS: LivePortrait will be forced to CPU (Conv3D not supported)
            if device == "mps":
                logger.warning("⚠️ LivePortrait using CPU (Conv3D not supported on MPS)")
                if progress_callback:
                    progress_callback(5, "⚠️ Usando CPU per LivePortrait (MPS non supporta Conv3D)")
            
            # Converti path in assoluti
            source_abs = str(Path(source_image).resolve())
            driving_abs = str(Path(driving_video).resolve())
            output_abs = str(Path(output_path).resolve())
            
            # Output directory per LivePortrait
            output_dir = Path(output_path).parent
            
            cmd = [
                sys.executable,
                str(inference_script),
                '--source', source_abs,
                '--driving', driving_abs,
                '--output_dir', str(output_dir)
            ]
            
            if progress_callback:
                progress_callback(10, "Avvio LivePortrait...")
            
            logger.info(f"Esecuzione LivePortrait: {' '.join(cmd)}")
            logger.info(f"CWD: {liveportrait_dir}")
            
            # Prepare environment - FORCE CPU for MPS
            env = os.environ.copy()
            if device == "mps":
                # Disable MPS to force CPU (Conv3D not supported on MPS)
                env['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
                env['CUDA_VISIBLE_DEVICES'] = ''
                logger.info("🖥️ Forcing CPU execution (MPS disabled for Conv3D operations)")
            
            # Esegui subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(liveportrait_dir),
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            # Leggi output
            stderr_lines = []
            for line in process.stderr:
                stderr_lines.append(line)
                line_stripped = line.strip()
                
                if line_stripped:
                    logger.info(f"LivePortrait: {line_stripped}")
                    
                    # Update progress based on output
                    if 'detect' in line_stripped.lower() or 'face' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(30, "🔍 Rilevamento viso...")
                    elif 'extract' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(50, "📊 Estrazione features...")
                    elif 'generat' in line_stripped.lower() or 'render' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(70, "🎬 Rendering video...")
                    elif 'writ' in line_stripped.lower() or 'sav' in line_stripped.lower():
                        if progress_callback:
                            progress_callback(90, "💾 Salvataggio...")
            
            returncode = process.wait()
            
            if returncode == 0:
                # LivePortrait salva con nome diverso, cerchiamo il file
                if Path(output_path).exists():
                    if progress_callback:
                        progress_callback(100, "✅ LivePortrait completato!")
                    return True
                else:
                    # Cerca file più recente nella output_dir
                    mp4_files = list(output_dir.glob('**/*.mp4'))
                    if mp4_files:
                        latest_file = max(mp4_files, key=lambda p: p.stat().st_mtime)
                        shutil.move(str(latest_file), output_path)
                        if progress_callback:
                            progress_callback(100, "✅ LivePortrait completato!")
                        return True
                    else:
                        logger.error(f"LivePortrait completato ma nessun file trovato in {output_dir}")
                        return False
            else:
                logger.error(f"LivePortrait fallito con exit code {returncode}")
                logger.error(f"STDERR: {''.join(stderr_lines[-20:])}")
                return False
                
        except Exception as e:
            logger.error(f"Eccezione in _run_liveportrait_inference: {e}", exc_info=True)
            return False
    
    def prepare_audio_with_countdown(self, audio_file, countdown_seconds=5):
        """
        Add countdown beeps to audio for webcam sync
        Returns: (countdown_audio_path, countdown_audio_path_for_preview)
        """
        if audio_file is None:
            return None, None
        
        try:
            # Generate audio with countdown
            output_path = self.temp_manager.get_temp_file_path("audio_with_countdown.wav")
            countdown_audio, duration = add_countdown_to_audio(
                audio_file, 
                output_path, 
                countdown_seconds=countdown_seconds
            )
            
            logger.info(f"✅ Audio with {countdown_seconds}s countdown generated: {countdown_audio}")
            
            # Return same path for both outputs (hidden textbox and audio player)
            return countdown_audio, countdown_audio
            
        except Exception as e:
            logger.error(f"Error generating countdown audio: {e}", exc_info=True)
            return None, None
    
    def process_lipsync(
        self,
        input_file,
        audio_file,
        model_name,
        device,
        resize_factor,
        nosmooth,
        use_webcam=False,
        driving_video_file=None,
        webcam_recording=None,
        audio_with_countdown_path=None,
        progress=gr.Progress()
    ):
        """Process lip sync with selected model"""
        if input_file is None:
            return None, "❌ Per favore carica un'immagine o video"
        
        if audio_file is None:
            return None, "❌ Per favore carica un file audio"
        
        # Check driving video for TheGargantuas LipSync model
        if model_name == "thegargantuas_lipsync":
            driving_video = webcam_recording if use_webcam else driving_video_file
            if driving_video is None:
                return None, "❌ TheGargantuas LipSync richiede un driving video. Per favore carica un video o registra dalla webcam."
            
            # If using webcam with countdown, sync and trim files
            if use_webcam and audio_with_countdown_path:
                progress(0, desc="🎬 Sincronizzazione audio e video...")
                try:
                    # Trim countdown from both audio and video
                    synced_video, synced_audio = sync_audio_video_with_countdown(
                        audio_with_countdown=audio_with_countdown_path,
                        webcam_video=driving_video,
                        output_dir=self.temp_manager.temp_dir,
                        countdown_duration=5
                    )
                    # Use synced files for processing
                    driving_video = synced_video
                    audio_file = synced_audio
                    progress(5, desc="✅ Sincronizzazione completata!")
                except Exception as e:
                    logger.error(f"Errore nella sincronizzazione: {e}", exc_info=True)
                    return None, f"❌ Errore sincronizzazione: {str(e)}"
        
        try:
            # Select device
            self.device_manager.set_device(device)
            
            # Check if TheGargantuas LipSync (LivePortrait + Wav2Lip pipeline)
            if model_name == "thegargantuas_lipsync":
                return self._process_audio_video_pipeline(
                    input_file=input_file,
                    audio_file=audio_file,
                    driving_video=webcam_recording if use_webcam else driving_video_file,
                    device=device,
                    resize_factor=resize_factor,
                    nosmooth=nosmooth,
                    progress=progress
                )
            
            # Standard audio-only processing
            # Initialize processor if model changed
            if self.processor is None or self.current_model != model_name:
                progress(0, desc=f"Inizializzazione {model_name}...")
                models_dir = Path(__file__).parent.parent / "models" / "lipsync"
                self.processor = LipSyncProcessor(
                    model_name=model_name,
                    device=device,
                    models_dir=models_dir
                )
                self.current_model = model_name
            
            # Download model if not present
            if not self.processor.is_model_downloaded():
                progress(0, desc="Download modello...")
                
                download_log = []
                def log_download(msg):
                    download_log.append(msg)
                    progress(0, desc=msg)
                
                success = self.processor.download_model(progress_callback=log_download)
                if not success:
                    return None, "❌ Errore durante il download del modello:\n" + "\n".join(download_log[-10:])
            
            # Process
            output_path = self.temp_manager.get_temp_file_path("lipsync_output.mp4")
            
            def progress_callback(percent, msg):
                progress(percent / 100, desc=msg)
            
            success = self.processor.process(
                image_or_video_path=input_file,
                audio_path=audio_file,
                output_path=str(output_path),
                progress_callback=progress_callback,
                resize_factor=resize_factor,
                nosmooth=nosmooth
            )
            
            if success:
                return str(output_path), f"✅ Lip sync completato con successo usando {LIPSYNC_MODELS[model_name]['name']}!"
            else:
                error_details = ""
                if self.processor and self.processor.last_error:
                    error_details = f"\n\nDettagli:\n{self.processor.last_error}"
                return None, f"❌ Errore durante il processing{error_details}"
            
        except Exception as e:
            logger.error(f"Error in lip sync: {e}", exc_info=True)
            return None, f"❌ Errore: {str(e)}"
    
    def create_tab(self):
        """Create and return the Lip Sync tab interface"""
        with gr.Tab(self._t('tabs.lipsync')):
            gr.Markdown(f"""
            # {self._t('lipsync.title')}
            {self._t('lipsync.description')}
            """)
            
            with gr.Row():
                with gr.Column():
                    # Model selection
                    model_dropdown = gr.Dropdown(
                        choices=list(LIPSYNC_MODELS.keys()),
                        value='wav2lip_gan',
                        label=self._t('lipsync.model'),
                        info=self._t('lipsync.select_model')
                    )
                    
                    # Model info display
                    model_info = gr.HTML(
                        value=self._get_model_info_html('wav2lip_gan'),
                        label=self._t('lipsync.model_info')
                    )
                    
                    # Update model info when selection changes
                    model_dropdown.change(
                        fn=self._get_model_info_html,
                        inputs=[model_dropdown],
                        outputs=[model_info]
                    )
                
                with gr.Column():
                    # Input files
                    input_file = gr.File(
                        label=self._t('lipsync.input_file'),
                        file_types=["image", "video"]
                    )
                    
                    audio_file = gr.File(
                        label=self._t('lipsync.audio_file'),
                        file_types=["audio"]
                    )
                    
                    # Driving video section (visible only in audio_video mode)
                    driving_video_section = gr.Group(visible=False)
                    with driving_video_section:
                        gr.Markdown("### 🎥 Driving Video (for body animation)")
                        
                        use_webcam = gr.Checkbox(
                            value=False,
                            label=self._t('lipsync.use_webcam'),
                            info=self._t('lipsync.use_webcam_info')
                        )
                        
                        driving_video_file = gr.File(
                            label=self._t('lipsync.driving_video_file'),
                            file_types=["video"],
                            visible=True
                        )
                        
                        # Webcam recording section
                        # NOTE: Accordion ALWAYS visible=True (always rendered in DOM)
                        # We use open=False to collapse it, NOT visible=False (which breaks webcam)
                        with gr.Accordion("📹 Webcam Recording", open=False) as webcam_accordion:
                            gr.Markdown("""
                            # 🎬 SYNCHRONIZED RECORDING WITH COUNTDOWN
                            
                            **How it works (AUTOMATIC SYNC!):**
                            1. When you upload audio, we add **5 BEEPS** at the start (countdown)
                            2. Click **▶️ PLAY** on audio below → You'll hear: BEEP...BEEP...BEEP...BEEP...BEEP...[YOUR AUDIO]
                            3. **Within 5 seconds**, click **🔴 RECORD** on webcam
                            4. Lip-sync and move with your audio!
                            5. Click **STOP** when done
                            6. When you process, we **automatically trim** both audio and video from the exact start point
                            
                            ✅ **Perfect sync guaranteed!** No need for perfect timing, just start recording within 5 seconds!
                            
                            ⚠️ **Webcam Permissions:**
                            - Browser will ask for camera → Click **ALLOW**
                            - **Safari**: Settings → Websites → Camera → Allow
                            - **Chrome/Firefox**: Click 🔒 → Camera → Allow
                            """)
                            
                            # Side by side layout
                            with gr.Row():
                                # Audio player with countdown - LEFT
                                audio_preview = gr.Audio(
                                    label="🎵 STEP 1: Click ▶️ PLAY (listen to countdown beeps)",
                                    type="filepath",
                                    interactive=True,
                                    scale=1
                                )
                                
                                # Webcam - RIGHT
                                webcam_recording = gr.Video(
                                    sources=["webcam"],
                                    label="📹 STEP 2: Click 🔴 RECORD within 5 seconds",
                                    include_audio=False,
                                    scale=1
                                )
                            
                            # Hidden: stores path to audio with countdown
                            audio_with_countdown_path = gr.Textbox(visible=False)
                    
                    # Settings
                    with gr.Accordion(self._t('lipsync.advanced_settings'), open=False):
                        device_radio = gr.Radio(
                            choices=self.device_manager.get_available_devices(),
                            value=self.device_manager.current_device,
                            label=self._t('lipsync.device'),
                            info=self._t('lipsync.device_info')
                        )
                        
                        resize_factor = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                            label=self._t('lipsync.resize_factor'),
                            info=self._t('lipsync.resize_info')
                        )
                        
                        nosmooth = gr.Checkbox(
                            value=False,
                            label=self._t('lipsync.nosmooth'),
                            info=self._t('lipsync.nosmooth_info')
                        )
                    
                    # Process button
                    process_btn = gr.Button(
                        self._t('lipsync.process_button'),
                        variant="primary",
                        size="lg"
                    )
                    
                    # Output
                    output_video = gr.Video(label=self._t('lipsync.output_video'))
                    output_info = gr.Textbox(
                        label=self._t('lipsync.info'),
                        lines=10,
                        max_lines=20
                    )
            
            # Callbacks for conditional visibility
            def toggle_driving_video_section(model_name):
                """Show/hide driving video section based on model selection"""
                # TheGargantuas LipSync requires driving video
                return gr.Group(visible=(model_name == "thegargantuas_lipsync"))
            
            def toggle_video_input(use_webcam_flag):
                """Toggle between file upload and webcam recording"""
                if use_webcam_flag:
                    # Show webcam accordion OPEN
                    return (
                        gr.File(visible=False),  # hide file upload
                        gr.Accordion(open=True)  # open webcam accordion
                    )
                else:
                    # Show file upload, close webcam accordion
                    return (
                        gr.File(visible=True),   # show file upload
                        gr.Accordion(open=False) # close webcam accordion (still in DOM!)
                    )
            
            # Update driving video section visibility when model changes
            model_dropdown.change(
                fn=toggle_driving_video_section,
                inputs=[model_dropdown],
                outputs=[driving_video_section]
            )
            
            use_webcam.change(
                fn=toggle_video_input,
                inputs=[use_webcam],
                outputs=[driving_video_file, webcam_accordion]  # changed from webcam_group
            )
            
            # Update audio preview when audio file is uploaded
            # Generate audio with countdown beeps for webcam sync
            audio_file.change(
                fn=self.prepare_audio_with_countdown,
                inputs=[audio_file],
                outputs=[audio_with_countdown_path, audio_preview]
            )
            
            # Wire up the process button
            process_btn.click(
                fn=self.process_lipsync,
                inputs=[
                    input_file,
                    audio_file,
                    model_dropdown,
                    device_radio,
                    resize_factor,
                    nosmooth,
                    use_webcam,
                    driving_video_file,
                    webcam_recording,
                    audio_with_countdown_path  # Added for countdown sync
                ],
                outputs=[output_video, output_info]
            )
            
            # Examples Gallery Section
            gr.Markdown("---")
            
            with gr.Accordion("📊 Compare Models - Example Results", open=False):
                gr.Markdown("""
                ### Side-by-Side Lip Sync Comparison
                Compare the original source with different lip sync models. Videos are synchronized and maintain their original aspect ratio.
                """)
                
                # Origin selector (img or video)
                example_origin_radio = gr.Radio(
                    choices=["img", "video"],
                    value="img",
                    label="Select Source Type",
                    info="Choose whether to see image-based or video-based results"
                )
                
                # Model selector (initially for img)
                example_model_radio = gr.Radio(
                    choices=["wav2lip", "wav2lipGAN", "sadtalker", "video_retalking"],
                    value="wav2lip",
                    label="Select Model to Compare",
                    info="Choose which lip sync model result to compare with the original"
                )
                
                gr.Markdown("""
                **ℹ️ Note:** Use the native video player controls below to play/pause the videos. 
                They will automatically synchronize when you interact with either video.
                """)
                
                # JavaScript for automatic video synchronization
                gr.HTML("""
                <script>
                (function() {
                    function setupLipSyncVideoSync() {
                        // Find video elements in lipsync section
                        const allVideos = document.querySelectorAll('video');
                        const videos = Array.from(allVideos).filter(v => {
                            const src = v.src || v.querySelector('source')?.src || '';
                            return src.includes('example/lipsync/');
                        });
                        
                        if (videos.length < 2) {
                            console.log('LipSync videos not ready yet, found:', videos.length);
                            return false;
                        }
                        
                        const [video1, video2] = videos;
                        
                        if (video1._lipsyncSyncSetup) {
                            return true; // Already set up
                        }
                        
                        // Mark as set up
                        video1._lipsyncSyncSetup = true;
                        video2._lipsyncSyncSetup = true;
                        
                        console.log('Setting up LipSync video synchronization...');
                        
                        // Mute both videos
                        video1.muted = true;
                        video2.muted = true;
                        
                        // Sync play events
                        video1.addEventListener('play', () => {
                            if (video2.paused) {
                                video2.play().catch(e => console.log('Sync play error:', e));
                            }
                        });
                        
                        video2.addEventListener('play', () => {
                            if (video1.paused) {
                                video1.play().catch(e => console.log('Sync play error:', e));
                            }
                        });
                        
                        // Sync pause events
                        video1.addEventListener('pause', () => {
                            if (!video2.paused) {
                                video2.pause();
                            }
                        });
                        
                        video2.addEventListener('pause', () => {
                            if (!video1.paused) {
                                video1.pause();
                            }
                        });
                        
                        // Sync seek events
                        video1.addEventListener('seeked', () => {
                            if (Math.abs(video1.currentTime - video2.currentTime) > 0.1) {
                                video2.currentTime = video1.currentTime;
                            }
                        });
                        
                        video2.addEventListener('seeked', () => {
                            if (Math.abs(video1.currentTime - video2.currentTime) > 0.1) {
                                video1.currentTime = video2.currentTime;
                            }
                        });
                        
                        console.log('✓ LipSync video synchronization active');
                        return true;
                    }
                    
                    // Try to setup multiple times
                    let attempts = 0;
                    const maxAttempts = 30;
                    const interval = setInterval(() => {
                        if (setupLipSyncVideoSync() || attempts >= maxAttempts) {
                            clearInterval(interval);
                            if (attempts >= maxAttempts) {
                                console.log('LipSync video sync setup timeout');
                            }
                        }
                        attempts++;
                    }, 500);
                })();
                </script>
                """)
                
                # Videos side by side
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎬 Original Source")
                        origin_video = gr.Video(
                            value="example/lipsync/img.jpg",
                            label="",
                            autoplay=False,
                            show_label=False,
                            height=500
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ✨ Lip Sync Result")
                        result_video = gr.Video(
                            value="example/lipsync/results/img_wav2lip.mp4",
                            label="",
                            autoplay=False,
                            show_label=False,
                            height=500
                        )
                
                # Info text
                gr.Markdown("""
                **💡 How to use:**
                1. Select **Source Type** (img or video) to change the original source
                2. Select a **Model** to see different lip sync results
                3. Click the **play button** on either video - both will start automatically
                4. Use the **seek bar** to jump to any point - both videos will sync
                5. Videos are muted and maintain their original aspect ratio
                
                **Note:** The videos are automatically synchronized. When you play, pause, or seek one video, the other will follow.
                """)
                
                def update_available_models(origin):
                    """Update available models based on origin"""
                    # Check which files exist for this origin
                    available_models = []
                    models_to_check = ["wav2lip", "wav2lipGAN", "sadtalker", "video_retalking"]
                    
                    for model in models_to_check:
                        result_path = Path(f"example/lipsync/results/{origin}_{model}.mp4")
                        if result_path.exists():
                            available_models.append(model)
                    
                    # Default to first available model
                    default_value = available_models[0] if available_models else "wav2lip"
                    
                    return gr.Radio(choices=available_models, value=default_value)
                
                def update_lipsync_example_videos(origin, model):
                    """Update example videos based on selected origin and model"""
                    # Original source
                    if origin == "img":
                        origin_path = "example/lipsync/img.jpg"
                    else:
                        origin_path = "example/lipsync/video.mp4"
                    
                    # Result video path
                    result_path = f"example/lipsync/results/{origin}_{model}.mp4"
                    
                    # Check if result file exists
                    if not Path(result_path).exists():
                        # Return origin and a placeholder message
                        return origin_path, None
                    
                    return origin_path, result_path
                
                # Update available models when origin changes
                example_origin_radio.change(
                    fn=update_available_models,
                    inputs=[example_origin_radio],
                    outputs=[example_model_radio]
                )
                
                # Update videos when origin or model changes
                example_origin_radio.change(
                    fn=update_lipsync_example_videos,
                    inputs=[example_origin_radio, example_model_radio],
                    outputs=[origin_video, result_video]
                )
                
                example_model_radio.change(
                    fn=update_lipsync_example_videos,
                    inputs=[example_origin_radio, example_model_radio],
                    outputs=[origin_video, result_video]
                )
            
            # Tips section
            gr.Markdown(f"""
            ---
            ## {self._t('lipsync.tips_title')}
            
            {self._t('lipsync.tips_content')}
            """)
