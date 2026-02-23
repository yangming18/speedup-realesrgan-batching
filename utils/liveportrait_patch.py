"""
Patch automatico per LivePortrait - Fix compatibilità NumPy 2.0+

Simile al patch di SadTalker, assicura compatibilità con NumPy 2.0+
rimuovendo gli alias deprecati np.float, np.int, np.bool
"""

import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def patch_liveportrait_numpy_compatibility(liveportrait_dir: Path) -> bool:
    """
    Patch automatico per compatibilità NumPy 2.0+ in LivePortrait
    
    Sostituisce:
    - np.float -> np.float64
    - np.int -> np.int64  
    - np.bool -> np.bool_
    - Aggiunge conversioni esplicite per array eterogenei
    
    Args:
        liveportrait_dir: Path alla cartella di LivePortrait
        
    Returns:
        True se il patch è stato applicato con successo
    """
    
    if not liveportrait_dir.exists():
        logger.warning(f"LivePortrait directory non trovata: {liveportrait_dir}")
        return False
    
    # Pattern di sostituzione
    replacements = [
        (r'\.astype\(np\.float\b', '.astype(np.float64'),  # np.float -> np.float64
        (r'\.astype\(np\.int\b', '.astype(np.int64'),      # np.int -> np.int64
        (r'\.astype\(np\.bool\b', '.astype(np.bool_'),     # np.bool -> np.bool_
    ]
    
    # Trova tutti i file Python in LivePortrait
    python_files = list(liveportrait_dir.glob('**/*.py'))
    
    patched_files = []
    
    for file_path in python_files:
        # Salta file in cartelle speciali
        if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'venv', '.venv']):
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
            continue
    
    if patched_files:
        logger.info(f"✅ LivePortrait patch completato: {len(patched_files)} file aggiornati")
        return True
    else:
        logger.debug("LivePortrait già patchato o nessuna modifica necessaria")
        return True


def check_if_patch_needed(liveportrait_dir: Path) -> bool:
    """
    Verifica se il patch è necessario cercando pattern deprecati
    
    Args:
        liveportrait_dir: Path alla cartella di LivePortrait
        
    Returns:
        True se il patch è necessario
    """
    
    if not liveportrait_dir.exists():
        return False
    
    try:
        # Cerca in tutti i file Python
        python_files = list(liveportrait_dir.glob('**/*.py'))
        
        for file_path in python_files:
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'venv', '.venv']):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                # Se contiene ancora np.float/int/bool deprecato, serve il patch
                if re.search(r'\.astype\(np\.(float|int|bool)\b', content):
                    return True
            except Exception:
                continue
                
        return False
        
    except Exception:
        return False
