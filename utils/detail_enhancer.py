"""Detail enhancement utilities for post-upscaling image refinement."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import types
from urllib.parse import urlparse

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - fallback to PIL
    from PIL import Image, ImageFilter

    class _MockCV2:
        BORDER_REFLECT = "reflect"
        COLOR_BGR2RGB = "bgr2rgb"
        COLOR_RGB2BGR = "rgb2bgr"
        INTER_CUBIC = "cubic"

        @staticmethod
        def cvtColor(arr, mode):
            if mode == _MockCV2.COLOR_BGR2RGB:
                return arr[:, :, ::-1]
            if mode == _MockCV2.COLOR_RGB2BGR:
                return arr[:, :, ::-1]
            return arr

        @staticmethod
        def GaussianBlur(arr, ksize, sigmaX):
            img = Image.fromarray(arr[:, :, ::-1])
            blurred = img.filter(ImageFilter.GaussianBlur(radius=sigmaX))
            return np.array(blurred)[:, :, ::-1]

        @staticmethod
        def addWeighted(src1, alpha, src2, beta, gamma):
            result = np.clip(src1 * alpha + src2 * beta + gamma, 0, 255)
            return result.astype(np.uint8)

        @staticmethod
        def bilateralFilter(arr, d, sigma_color, sigma_space):
            return arr

        @staticmethod
        def detailEnhance(arr, sigma_s=10, sigma_r=0.15):
            return arr

        @staticmethod
        def resize(arr, size, interpolation=None):
            img = Image.fromarray(arr[:, :, ::-1])
            img = img.resize(size, Image.LANCZOS)
            return np.array(img)[:, :, ::-1]

        @staticmethod
        def copyMakeBorder(arr, top, bottom, left, right, borderType):
            return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="reflect")

    cv2 = _MockCV2()  # type: ignore

LOGGER = logging.getLogger(__name__)

# Compatibilità torchvision >=0.17: il modulo functional_tensor è stato spostato
try:  # pragma: no cover - opzionale
    import torchvision.transforms.functional as _tv_functional  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - torchvision assente
    _tv_functional = None  # type: ignore
else:  # pragma: no cover - solo per runtime
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        if hasattr(_tv_functional, "rgb_to_grayscale"):
            shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale  # type: ignore[attr-defined]
            sys.modules["torchvision.transforms.functional_tensor"] = shim

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from gfpgan import GFPGANer
except ImportError:  # pragma: no cover - handled at runtime
    GFPGANer = None  # type: ignore
    GFPGAN_AVAILABLE = False
else:
    GFPGAN_AVAILABLE = True


def _download_file(url: str, destination: Path) -> None:
    """Scarica un file nella destinazione indicata se non esiste."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    if "huggingface.co" in parsed.netloc:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - dipendenza opzionale
            raise ImportError(
                "huggingface_hub non è installato. Esegui 'pip install huggingface_hub' per scaricare i modelli."
            ) from exc

        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "resolve":
            repo_id = f"{parts[0]}/{parts[1]}"
            revision = parts[3]
            filename = "/".join(parts[4:])
        else:
            raise ValueError(f"URL Hugging Face non riconosciuto: {url}")

        LOGGER.info("Scaricamento modello detail enhancer da Hugging Face (%s)", repo_id)
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=None  # usa token attivo (se presente)
        )
        shutil.copy(cached_path, destination)
        return

    import requests

    expected_size = None
    try:
        head = requests.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        expected_size = int(head.headers.get("content-length", 0)) or None
    except Exception:
        expected_size = None

    if destination.exists():
        if expected_size is not None and destination.stat().st_size == expected_size:
            return
        LOGGER.warning(
            "File esistente con dimensione inattesa, riscarico: %s (atteso=%s, attuale=%s)",
            destination,
            "n/d" if expected_size is None else expected_size,
            destination.stat().st_size
        )
        destination.unlink()

    LOGGER.info("Scaricamento modello detail enhancer da %s", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0)) or None
    downloaded = 0
    with open(destination, "wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = (downloaded / total) * 100
                if percent % 10 < 0.1:
                    LOGGER.debug("Scaricamento %.1f%%", percent)

    if expected_size and destination.stat().st_size != expected_size:
        destination.unlink(missing_ok=True)
        raise IOError(
            f"Download incompleto per {destination.name}: atteso {expected_size} byte, ricevuti {downloaded}"
        )


def get_detail_model_choices() -> List[Tuple[str, str, bool]]:
    """Restituisce elenco modelli disponibili (chiave, etichetta, disponibile)."""
    choices = [
        ("none", "Nessuno (solo upscaling)", True),
        ("classic_sharpen", "Classic Sharpen (veloce)", True),
        ("edge_enhance", "Edge Enhancement (contorni)", True),
        ("denoise_medium", "Denoise Medium (riduzione rumore)", True),
        ("denoise_strong", "Denoise Strong (rimozione rumore aggressiva)", True),
    ]
    if GFPGAN_AVAILABLE and torch is not None:
        choices.append(("gfpgan_face_restore", "GFPGAN Face Restore (dettagli volto)", True))
    else:
        choices.append(("gfpgan_face_restore", "GFPGAN Face Restore (richiede torch+gfpgan)", False))
    return choices


class DetailEnhancer:
    """Applica modelli di aumento definizione ai frame."""

    MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
        "classic_sharpen": {
            "label": "Classic Sharpen",
            "requires": [],
        },
        "edge_enhance": {
            "label": "Edge Enhancement",
            "requires": [],
        },
        "denoise_medium": {
            "label": "Denoise Medium",
            "requires": [],
        },
        "denoise_strong": {
            "label": "Denoise Strong",
            "requires": [],
        },
        "gfpgan_face_restore": {
            "label": "GFPGAN Face Restore",
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            "filename": "GFPGANv1.3.pth",
            "upscale": 1,
        },
    }

    def __init__(self, model_name: str, device_preference: Optional[str] = None):
        if model_name == "none":
            raise ValueError("Il modello 'none' non può essere inizializzato")

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Modello detail enhancer non supportato: {model_name}")

        self.model_name = model_name
        self.config = self.MODEL_CONFIGS[model_name]
        self.device_preference = device_preference
        self.logger = LOGGER
        self.device = None
        self.model = None

        if model_name == "classic_sharpen":
            self.available = True
            return

        if model_name in ("edge_enhance", "denoise_medium", "denoise_strong"):
            self.available = True
            return

        if model_name == "gfpgan_face_restore":
            if not GFPGAN_AVAILABLE or torch is None:
                raise ImportError(
                    "GFPGAN non disponibile. Installa torch, gfpgan e dipendenze correlate."
                )
            self._initialize_gfpgan()
            self.available = True
            return

        raise ValueError(f"Modello detail enhancer sconosciuto: {model_name}")

    def enhance(
        self,
        image_bgr: np.ndarray,
        focus_areas: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        if self.model_name == "classic_sharpen":
            return self._enhance_classic(image_bgr)
        if self.model_name == "edge_enhance":
            return self._enhance_edges(image_bgr)
        if self.model_name == "denoise_medium":
            return self._enhance_denoise_medium(image_bgr)
        if self.model_name == "denoise_strong":
            return self._enhance_denoise_strong(image_bgr)
        if self.model_name == "gfpgan_face_restore":
            return self._enhance_gfpgan(image_bgr, focus_areas)
        raise RuntimeError("Modello detail enhancer non supportato")

    @staticmethod
    def _enhance_classic(image: np.ndarray) -> np.ndarray:
        # Bilateral filter + unsharp mask + detail enhance for moderate sharpening
        smooth = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
        unsharp = cv2.addWeighted(image, 1.6, smooth, -0.6, 0)
        detail = cv2.detailEnhance(unsharp, sigma_s=12, sigma_r=0.2)
        blended = cv2.addWeighted(detail, 0.85, image, 0.15, 0)
        return blended

    @staticmethod
    def _enhance_edges(image: np.ndarray) -> np.ndarray:
        """Migliora i contorni mantenendo le aree piatte smooth."""
        # Laplacian edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Dilata i bordi per renderli più evidenti
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(laplacian, kernel, iterations=1)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combina con l'immagine originale
        enhanced = cv2.addWeighted(image, 1.3, edges_bgr, 0.3, 0)
        
        # Unsharp mask per accentuare ulteriormente
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return np.clip(unsharp, 0, 255).astype(np.uint8)

    @staticmethod
    def _enhance_denoise_medium(image: np.ndarray) -> np.ndarray:
        """Riduzione rumore media preservando i dettagli."""
        # Non-local means denoising con parametri bilanciati
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=6,           # Forza filtro per componenti colore
            hColor=6,      # Forza filtro per luminanza
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Leggero sharpen per recuperare dettagli
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.0)
        sharpened = cv2.addWeighted(denoised, 1.3, gaussian, -0.3, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _enhance_denoise_strong(image: np.ndarray) -> np.ndarray:
        """Rimozione aggressiva del rumore, ideale per video molto rumorosi."""
        # Prima passata: bilateral filter per ridurre rumore preservando bordi
        bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Seconda passata: non-local means con parametri più aggressivi
        denoised = cv2.fastNlMeansDenoisingColored(
            bilateral,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Morphological closing per eliminare piccoli artefatti
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Blend con l'immagine originale per evitare perdita eccessiva di dettaglio
        result = cv2.addWeighted(cleaned, 0.8, image, 0.2, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _select_device(self):
        assert torch is not None
        if self.device_preference:
            pref = self.device_preference.lower()
            if pref == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            if pref == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if pref == "cpu":
                return torch.device("cpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _initialize_gfpgan(self) -> None:
        assert torch is not None and GFPGANer is not None
        self.device = self._select_device()
        cfg = self.config
        filename = str(cfg.get("filename", "gfpgan_model.pth"))
        model_path = Path("models/detail") / filename
        _download_file(str(cfg["url"]), model_path)
        # L'upsample rimane a 1: reinseriamo i dettagli sul frame originale
        self.model = GFPGANer(
            model_path=str(model_path),
            upscale=int(cfg.get("upscale", 1)),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )

    def _enhance_gfpgan(
        self,
        image_bgr: np.ndarray,
        focus_areas: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        assert self.model is not None and GFPGANer is not None
        try:
            _cropped, _restored, restored_img = self.model.enhance(
                image_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
        except Exception as exc:
            self.logger.warning("GFPGAN non è riuscito a ripristinare il volto: %s", exc)
            return image_bgr

        if restored_img is None:
            return image_bgr

        # Blend leggero per mantenere texture globali
        blended_full = cv2.addWeighted(restored_img, 0.75, image_bgr, 0.25, 0)

        if not focus_areas:
            return blended_full

        # Applica il blend solo nelle aree selezionate
        height, width = image_bgr.shape[:2]
        output = image_bgr.copy()

        for area in focus_areas:
            if area is None or len(area) != 4:
                continue

            raw_x1, raw_y1, raw_x2, raw_y2 = area
            x1 = max(0, min(int(round(raw_x1)), width - 1))
            y1 = max(0, min(int(round(raw_y1)), height - 1))
            x2 = max(0, min(int(round(raw_x2)), width))
            y2 = max(0, min(int(round(raw_y2)), height))

            if x2 <= x1 or y2 <= y1:
                continue

            src_region = image_bgr[y1:y2, x1:x2]
            enhanced_region = cv2.addWeighted(
                restored_img[y1:y2, x1:x2],
                0.75,
                src_region,
                0.25,
                0
            )
            output[y1:y2, x1:x2] = enhanced_region

        return output

    def release(self):  # pragma: no cover - solo in runtime
        if torch is not None and self.model is not None:
            try:
                if hasattr(self.model, "clean_all"):
                    self.model.clean_all()
            except Exception:
                pass
            self.model = None