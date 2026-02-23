"""
Internationalization (i18n) Manager
Gestisce le traduzioni dell'applicazione
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class I18nManager:
    """Gestisce traduzioni e localizzazione"""
    
    AVAILABLE_LANGUAGES = {
        'en': 'English',
        'it': 'Italiano'
    }
    
    def __init__(self, default_lang: str = 'en'):
        self.locales_dir = Path(__file__).parent.parent / "locales"
        self.current_lang = default_lang
        self.translations: Dict[str, Any] = {}
        self.load_language(default_lang)
    
    def load_language(self, lang_code: str) -> bool:
        """Carica file di traduzione per la lingua specificata"""
        if lang_code not in self.AVAILABLE_LANGUAGES:
            logger.warning(f"Lingua non supportata: {lang_code}, uso default (en)")
            lang_code = 'en'
        
        locale_file = self.locales_dir / f"{lang_code}.json"
        
        if not locale_file.exists():
            logger.error(f"File traduzione non trovato: {locale_file}")
            return False
        
        try:
            with open(locale_file, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
            self.current_lang = lang_code
            logger.info(f"✓ Lingua caricata: {self.AVAILABLE_LANGUAGES[lang_code]}")
            return True
        except Exception as e:
            logger.error(f"Errore caricamento traduzione: {e}")
            return False
    
    def t(self, key: str, **kwargs) -> str:
        """
        Traduce una chiave usando la lingua corrente.
        
        Args:
            key: Chiave in formato dot-notation (es: 'app.title')
            **kwargs: Variabili da sostituire nel testo (es: {model})
        
        Returns:
            Stringa tradotta
        """
        keys = key.split('.')
        value = self.translations
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                logger.warning(f"Chiave traduzione non trovata: {key}")
                return key  # Ritorna la chiave stessa se non trovata
        
        # Sostituisci variabili se presenti
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Variabile mancante in traduzione {key}: {e}")
                return value
        
        return str(value)
    
    def get_current_language(self) -> str:
        """Ritorna il codice della lingua corrente"""
        return self.current_lang
    
    def get_current_language_name(self) -> str:
        """Ritorna il nome della lingua corrente"""
        return self.AVAILABLE_LANGUAGES.get(self.current_lang, 'Unknown')
    
    def get_available_languages(self) -> Dict[str, str]:
        """Ritorna dizionario delle lingue disponibili"""
        return self.AVAILABLE_LANGUAGES.copy()
    
    def change_language(self, lang_code: str) -> bool:
        """Cambia lingua dell'applicazione"""
        return self.load_language(lang_code)


# Istanza globale
_i18n_instance: Optional[I18nManager] = None


def get_i18n(lang: Optional[str] = None) -> I18nManager:
    """
    Ottieni istanza globale del manager i18n.
    
    Args:
        lang: Lingua da caricare (opzionale, default: en)
    
    Returns:
        Istanza I18nManager
    """
    global _i18n_instance
    
    if _i18n_instance is None:
        _i18n_instance = I18nManager(default_lang=lang or 'en')
    elif lang and lang != _i18n_instance.current_lang:
        _i18n_instance.change_language(lang)
    
    return _i18n_instance


def t(key: str, **kwargs) -> str:
    """
    Shortcut per traduzione rapida.
    
    Usage:
        from utils.i18n import t
        title = t('app.title')
        msg = t('lipsync.success', model='Wav2Lip')
    """
    return get_i18n().t(key, **kwargs)
