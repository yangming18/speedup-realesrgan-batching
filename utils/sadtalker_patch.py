"""
Patch automatico per SadTalker - Fix compatibilità NumPy 2.0+

NumPy 2.0 ha rimosso gli alias deprecati np.float, np.int, np.bool
Questo script applica automaticamente le correzioni necessarie ai file di SadTalker
"""

import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def patch_sadtalker_numpy_compatibility(sadtalker_dir: Path) -> bool:
    """
    Patch automatico per compatibilità NumPy 2.0+ in SadTalker
    
    Sostituisce:
    - np.float -> np.float64
    - np.int -> np.int64
    - np.bool -> np.bool_
    
    Args:
        sadtalker_dir: Path alla cartella di SadTalker
        
    Returns:
        True se il patch è stato applicato con successo
    """
    
    if not sadtalker_dir.exists():
        logger.warning(f"SadTalker directory non trovata: {sadtalker_dir}")
        return False
    
    # File da patchare
    files_to_patch = [
        sadtalker_dir / "src/face3d/util/my_awing_arch.py",
        sadtalker_dir / "src/face3d/util/preprocess.py",
        sadtalker_dir / "src/face3d/models/arcface_torch/torch2onnx.py",
        sadtalker_dir / "src/face3d/models/arcface_torch/onnx_ijbc.py",
        sadtalker_dir / "src/face3d/models/arcface_torch/eval_ijbc.py",
        sadtalker_dir / "src/face3d/models/arcface_torch/utils/plot.py",
    ]
    
    # Pattern di sostituzione
    replacements = [
        (r'\.astype\(np\.float\b', '.astype(np.float64'),  # np.float -> np.float64
        (r'\.astype\(np\.int\b', '.astype(np.int64'),      # np.int -> np.int64
        (r'\.astype\(np\.bool\b', '.astype(np.bool_'),     # np.bool -> np.bool_
        # Fix per np.array con elementi heterogenei (NumPy 2.0+)
        (r'np\.array\(\[w0, h0, s, t\[0\], t\[1\]\]\)', 
         'np.array([w0, h0, float(s), float(t[0]), float(t[1])])'),
    ]
    
    patched_files = []
    
    for file_path in files_to_patch:
        if not file_path.exists():
            continue
        
        try:
            # Leggi contenuto
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Applica sostituzioni
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            # Se ci sono modifiche, scrivi il file
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                patched_files.append(file_path.name)
                logger.info(f"✓ Patch applicato: {file_path.name}")
        
        except Exception as e:
            logger.error(f"Errore nel patch di {file_path.name}: {e}")
            return False
    
    if patched_files:
        logger.info(f"✅ SadTalker patch completato: {len(patched_files)} file aggiornati")
        return True
    else:
        logger.debug("SadTalker già patchato o nessuna modifica necessaria")
        return True


def check_if_patch_needed(sadtalker_dir: Path) -> bool:
    """
    Verifica se il patch è necessario controllando un file chiave
    
    Args:
        sadtalker_dir: Path alla cartella di SadTalker
        
    Returns:
        True se il patch è necessario
    """
    test_file = sadtalker_dir / "src/face3d/util/my_awing_arch.py"
    
    if not test_file.exists():
        return False
    
    try:
        content = test_file.read_text(encoding='utf-8')
        # Se contiene ancora np.float deprecato, serve il patch
        return bool(re.search(r'\.astype\(np\.float\b', content))
    except Exception:
        return False
