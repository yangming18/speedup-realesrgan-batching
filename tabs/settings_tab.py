"""
Settings Tab
Configurazione e preferenze dell'applicazione
"""
import gradio as gr
import logging
from pathlib import Path
import json
from utils import api_key_manager, get_openai_helper

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
    
    def save_api_key(self, api_key: str, provider: str = "openai") -> tuple[str, str]:
        """
        Save and test API key for specified provider.
        
        Args:
            api_key: API key to save
            provider: "openai" or "groq"
        
        Returns:
            Tuple of (status_message, models_json)
        """
        if not api_key or not api_key.strip():
            return "❌ Please enter an API key", "[]"
        
        api_key = api_key.strip()
        key_name = f"{provider.upper()}_API_KEY"
        provider_label = "OpenAI" if provider == "openai" else "Groq"
        
        # Save encrypted
        try:
            api_key_manager.save_api_key(key_name, api_key)
            
            # Test connection
            helper = get_openai_helper(api_key, provider)
            success, message = helper.test_connection()
            
            if success:
                # Get available models
                models = helper.get_available_models()
                models_json = json.dumps(models)
                return f"✓ {provider_label} API Key saved and validated! {message}", models_json
            else:
                return f"⚠️ {provider_label} API Key saved but validation failed: {message}", "[]"
                
        except Exception as e:
            logger.error(f"Error saving {provider} API key: {e}")
            return f"❌ Error: {str(e)}", "[]"
    
    def test_api_key(self, provider: str = "openai") -> tuple[str, str]:
        """
        Test existing API key for specified provider.
        
        Args:
            provider: "openai" or "groq"
        
        Returns:
            Tuple of (status_message, models_json)
        """
        key_name = f"{provider.upper()}_API_KEY"
        api_key = api_key_manager.get_api_key(key_name)
        provider_label = "OpenAI" if provider == "openai" else "Groq"
        
        if not api_key:
            return f"❌ No {provider_label} API key found. Please save one first.", "[]"
        
        try:
            helper = get_openai_helper(api_key, provider)
            success, message = helper.test_connection()
            
            if success:
                models = helper.get_available_models()
                models_json = json.dumps(models)
                return f"✓ {message}", models_json
            else:
                return f"❌ {message}", "[]"
                
        except Exception as e:
            return f"❌ Test failed: {str(e)}", "[]"
    
    def delete_api_key(self, provider: str = "openai") -> str:
        """Delete saved API key for specified provider"""
        key_name = f"{provider.upper()}_API_KEY"
        provider_label = "OpenAI" if provider == "openai" else "Groq"
        try:
            api_key_manager.delete_api_key(key_name)
            return f"✓ {provider_label} API Key deleted successfully"
        except Exception as e:
            return f"❌ Error deleting key: {str(e)}"
    
    def load_existing_api_key(self, key_name: str) -> str:
        """Load existing API key (masked)"""
        api_key = api_key_manager.get_api_key(key_name)
        if api_key:
            # Return masked version
            return f"{api_key[:8]}...{api_key[-4:]}"
        return ""
    
    def create_tab(self):
        """Crea e ritorna l'interfaccia del tab Impostazioni"""
        with gr.Tab(self.i18n.t('tabs.settings')):
            gr.Markdown(f"""
            # {self.i18n.t('settings.title')}
            {self.i18n.t('settings.description')}
            """)
            
            # === OPENAI API KEY SECTION ===
            with gr.Accordion("🔑 OpenAI API Key", open=False):
                gr.Markdown("""
                ### API Key Configuration
                
                Configure your OpenAI API key to enable subtitle generation features.
                Your API key is encrypted and stored securely on your machine.
                
                **Get your API key:** [OpenAI Platform](https://platform.openai.com/api-keys) *(Requires payment)*
                """)
                
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-proj-...",
                        type="password",
                        scale=3,
                        info="Your API key will be encrypted before saving"
                    )
                    
                    saved_key_display = gr.Textbox(
                        label="Current Key (Masked)",
                        value=self.load_existing_api_key("OPENAI_API_KEY"),
                        interactive=False,
                        scale=2
                    )
                
                with gr.Row():
                    save_key_btn = gr.Button("💾 Save & Test API Key", variant="primary")
                    test_key_btn = gr.Button("🔍 Test Existing Key")
                    delete_key_btn = gr.Button("🗑️ Delete Key", variant="stop")
                
                api_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                models_state = gr.State([])
                
                # Wire up API key events
                save_key_btn.click(
                    fn=lambda key: self.save_api_key(key, "openai"),
                    inputs=[api_key_input],
                    outputs=[api_status, models_state]
                ).then(
                    fn=lambda: self.load_existing_api_key("OPENAI_API_KEY"),
                    inputs=[],
                    outputs=[saved_key_display]
                )
                
                test_key_btn.click(
                    fn=lambda: self.test_api_key("openai"),
                    inputs=[],
                    outputs=[api_status, models_state]
                )
                
                delete_key_btn.click(
                    fn=lambda: self.delete_api_key("openai"),
                    inputs=[],
                    outputs=[api_status]
                ).then(
                    fn=lambda: "",
                    inputs=[],
                    outputs=[saved_key_display]
                )
            
            # === GROQ API KEY SECTION (FREE) ===
            with gr.Accordion("🚀 Groq API Key (FREE)", open=False):
                gr.Markdown("""
                ### Groq API - Fast & Free Alternative
                
                Groq offers **FREE API access** with ultra-fast inference speeds!
                Perfect for subtitle generation without any cost.
                
                **Get your FREE API key:** [Groq Console](https://console.groq.com/keys) *(No credit card required)*
                
                Available models: Llama 3.1 (8B/70B), Mixtral 8x7B, Gemma 2 9B
                """)
                
                with gr.Row():
                    groq_key_input = gr.Textbox(
                        label="Groq API Key",
                        placeholder="gsk_...",
                        type="password",
                        scale=3,
                        info="Your API key will be encrypted before saving"
                    )
                    
                    groq_saved_key_display = gr.Textbox(
                        label="Current Key (Masked)",
                        value=self.load_existing_api_key("GROQ_API_KEY"),
                        interactive=False,
                        scale=2
                    )
                
                with gr.Row():
                    groq_save_key_btn = gr.Button("💾 Save & Test API Key", variant="primary")
                    groq_test_key_btn = gr.Button("🔍 Test Existing Key")
                    groq_delete_key_btn = gr.Button("🗑️ Delete Key", variant="stop")
                
                groq_api_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                groq_models_state = gr.State([])
                
                # Wire up Groq API key events
                groq_save_key_btn.click(
                    fn=lambda key: self.save_api_key(key, "groq"),
                    inputs=[groq_key_input],
                    outputs=[groq_api_status, groq_models_state]
                ).then(
                    fn=lambda: self.load_existing_api_key("GROQ_API_KEY"),
                    inputs=[],
                    outputs=[groq_saved_key_display]
                )
                
                groq_test_key_btn.click(
                    fn=lambda: self.test_api_key("groq"),
                    inputs=[],
                    outputs=[groq_api_status, groq_models_state]
                )
                
                groq_delete_key_btn.click(
                    fn=lambda: self.delete_api_key("groq"),
                    inputs=[],
                    outputs=[groq_api_status]
                ).then(
                    fn=lambda: "",
                    inputs=[],
                    outputs=[groq_saved_key_display]
                )
            
            # === GEMINI API KEY SECTION (FREE) ===
            with gr.Accordion("✨ Google Gemini API Key (FREE)", open=False):
                gr.Markdown("""
                ### Google Gemini API - Powerful & Free
                
                Google offers **FREE API access** with extremely generous limits!
                **1 Million tokens per minute** - much higher than other free tiers.
                
                **Get your FREE API key:** [Google AI Studio](https://aistudio.google.com/app/apikey) *(Google account required, no credit card)*
                
                Available models: Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0 Flash
                """)
                
                with gr.Row():
                    gemini_key_input = gr.Textbox(
                        label="Gemini API Key",
                        placeholder="AIza...",
                        type="password",
                        scale=3,
                        info="Your API key will be encrypted before saving"
                    )
                    
                    gemini_saved_key_display = gr.Textbox(
                        label="Current Key (Masked)",
                        value=self.load_existing_api_key("GEMINI_API_KEY"),
                        interactive=False,
                        scale=2
                    )
                
                with gr.Row():
                    gemini_save_key_btn = gr.Button("💾 Save & Test API Key", variant="primary")
                    gemini_test_key_btn = gr.Button("🔍 Test Existing Key")
                    gemini_delete_key_btn = gr.Button("🗑️ Delete Key", variant="stop")
                
                gemini_api_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                gemini_models_state = gr.State([])
                
                # Wire up Gemini API key events
                gemini_save_key_btn.click(
                    fn=lambda key: self.save_api_key(key, "gemini"),
                    inputs=[gemini_key_input],
                    outputs=[gemini_api_status, gemini_models_state]
                ).then(
                    fn=lambda: self.load_existing_api_key("GEMINI_API_KEY"),
                    inputs=[],
                    outputs=[gemini_saved_key_display]
                )
                
                gemini_test_key_btn.click(
                    fn=lambda: self.test_api_key("gemini"),
                    inputs=[],
                    outputs=[gemini_api_status, gemini_models_state]
                )
                
                gemini_delete_key_btn.click(
                    fn=lambda: self.delete_api_key("gemini"),
                    inputs=[],
                    outputs=[gemini_api_status]
                ).then(
                    fn=lambda: "",
                    inputs=[],
                    outputs=[gemini_saved_key_display]
                )
            
            gr.Markdown("---")
            
            # === LANGUAGE SETTINGS ===
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
