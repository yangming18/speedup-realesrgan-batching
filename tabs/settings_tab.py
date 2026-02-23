"""
Settings Tab
Configurazione e preferenze dell'applicazione
"""
import gradio as gr
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SettingsTab:
    """Gestisce il tab delle impostazioni"""
    
    def __init__(self, i18n_manager, on_language_change=None):
        """
        Inizializza il tab Impostazioni.
        
        Args:
            i18n_manager: Istanza del manager i18n
            on_language_change: Callback quando cambia la lingua
        """
        self.i18n = i18n_manager
        self.on_language_change = on_language_change
        self.config_file = Path(__file__).parent.parent / "config" / "user_settings.json"
        self.settings = self.load_settings()
    
    def load_settings(self) -> dict:
        """Carica impostazioni salvate"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Errore caricamento impostazioni: {e}")
        
        return {'language': 'en'}
    
    def save_settings(self, settings: dict) -> bool:
        """Salva impostazioni su file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Impostazioni salvate: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Errore salvataggio impostazioni: {e}")
            return False
    
    def change_language(self, lang_code: str) -> str:
        """
        Cambia lingua dell'applicazione.
        
        Args:
            lang_code: Codice lingua (en, it, etc.)
        
        Returns:
            Messaggio di conferma
        """
        if self.i18n.change_language(lang_code):
            self.settings['language'] = lang_code
            self.save_settings(self.settings)
            
            # Notifica callback per aggiornare l'interfaccia
            if self.on_language_change:
                self.on_language_change(lang_code)
            
            return self.i18n.t('settings.saved')
        else:
            return f"❌ Errore cambio lingua"
    
    def create_tab(self):
        """Crea e ritorna l'interfaccia del tab Impostazioni"""
        with gr.Tab(self.i18n.t('tabs.settings')):
            gr.Markdown(f"""
            # {self.i18n.t('settings.title')}
            {self.i18n.t('settings.description')}
            """)
            
            with gr.Row():
                with gr.Column():
                    # Selezione lingua
                    language_choices = list(self.i18n.get_available_languages().items())
                    language_dropdown = gr.Dropdown(
                        choices=[(name, code) for code, name in language_choices],
                        value=self.i18n.get_current_language(),
                        label=self.i18n.t('settings.language'),
                        info=self.i18n.t('settings.language_info')
                    )
                    
                    save_button = gr.Button(
                        self.i18n.t('settings.save'),
                        variant="primary",
                        size="lg"
                    )
                    
                    status_message = gr.Textbox(
                        label=self.i18n.t('common.info'),
                        interactive=False,
                        lines=2
                    )
                
                with gr.Column():
                    gr.Markdown("""
                    ### 📋 Current Settings
                    
                    **Available Languages:**
                    - 🇬🇧 English
                    - 🇮🇹 Italiano
                    
                    **Coming Soon:**
                    - 🎨 Theme customization
                    - 🔔 Notifications
                    - 💾 Auto-save preferences
                    """)
            
            # Wire up events
            save_button.click(
                fn=self.change_language,
                inputs=[language_dropdown],
                outputs=[status_message]
            )
            
            gr.Markdown("""
            ---
            ### 💡 Note
            
            When you change the language, **please refresh the page** to see all changes applied throughout the interface.
            
            Quando cambi lingua, **ricarica la pagina** per vedere tutte le modifiche applicate all'interfaccia.
            """)
