#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAI/Groq/Gemini Helper Functions
Handles API calls and model listings for OpenAI, Groq, and Google Gemini APIs
"""

import logging
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIHelper:
    """Helper class for OpenAI/Groq/Gemini API interactions"""
    
    def __init__(self, api_key: str = None, provider: str = "openai"):
        """
        Initialize helper with API key and provider.
        
        Args:
            api_key: API key for the provider
            provider: "openai", "groq", or "gemini"
        """
        self.api_key = api_key
        self.provider = provider
        self.client = None
        if api_key:
            self.initialize_client(api_key, provider)
    
    def initialize_client(self, api_key: str, provider: str = "openai"):
        """Initialize client with API key and provider"""
        try:
            self.api_key = api_key
            self.provider = provider
            
            if provider == "groq":
                # Groq uses OpenAI-compatible API
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                logger.info("Groq client initialized successfully")
            elif provider == "gemini":
                # Google Gemini uses OpenAI-compatible API
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                logger.info("Gemini client initialized successfully")
            else:
                # Default to OpenAI
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {provider} client: {e}")
            return False
    
    def test_connection(self) -> tuple[bool, str]:
        """
        Test API connection.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.client:
            return False, "Client not initialized. Please provide an API key."
        
        try:
            # Test with a simple API call
            models = self.client.models.list()
            return True, f"✓ Connection successful! Found {len(models.data)} models."
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                return False, "❌ Invalid API key. Please check your key."
            elif "quota" in error_msg.lower():
                return False, "❌ API quota exceeded. Check your OpenAI account."
            else:
                return False, f"❌ Connection failed: {error_msg}"
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get list of available models for chat completion.
        
        Returns:
            List of dicts with 'id' and 'name' keys
        """
        if not self.client:
            logger.warning("Cannot get models: client not initialized")
            return self._get_default_models()
        
        # Groq: fetch from API
        if self.provider == "groq":
            try:
                models_response = self.client.models.list()
                
                # Filter for chat-compatible models
                groq_models = []
                for model in models_response.data:
                    model_id = model.id
                    # Include common Groq chat models
                    if any(prefix in model_id for prefix in ['llama', 'mixtral', 'gemma', 'qwen']):
                        # Skip non-chat variants
                        if 'guard' not in model_id and 'vision' not in model_id:
                            name = self._format_model_name(model_id)
                            groq_models.append({
                                'id': model_id,
                                'name': name
                            })
                
                # Sort by capability (larger models first)
                groq_models.sort(key=lambda x: (
                    '405b' not in x['id'],  # 405B first (if available)
                    '70b' not in x['id'],   # 70B second
                    '8b' not in x['id'],    # 8B models last
                    x['id']
                ))
                
                logger.info(f"Found {len(groq_models)} Groq models")
                return groq_models if groq_models else self._get_default_models()
                
            except Exception as e:
                logger.error(f"Failed to retrieve Groq models: {e}")
                return self._get_default_models()
        
        # OpenAI: fetch from API
        try:
            models_response = self.client.models.list()
            
            # Filter for GPT models suitable for chat completion
            gpt_models = []
            for model in models_response.data:
                model_id = model.id
                # Include GPT-4, GPT-3.5, and GPT-4o variants
                if any(prefix in model_id for prefix in ['gpt-4', 'gpt-3.5']):
                    # Exclude fine-tuned, instruct-only, or deprecated models
                    if not any(skip in model_id for skip in ['instruct', 'vision', 'audio']):
                        # Create friendly name
                        name = self._format_model_name(model_id)
                        gpt_models.append({
                            'id': model_id,
                            'name': name
                        })
            
            # Sort by model name (GPT-4 first, then GPT-3.5)
            gpt_models.sort(key=lambda x: (
                '4o' not in x['id'],  # GPT-4o first
                '4-turbo' not in x['id'],  # GPT-4 Turbo second
                '4' not in x['id'],  # GPT-4 base third
                x['id']  # Then alphabetically
            ))
            
            logger.info(f"Found {len(gpt_models)} GPT models")
            return gpt_models if gpt_models else self._get_default_models()
            
        except Exception as e:
            logger.error(f"Failed to retrieve models: {e}")
            return self._get_default_models()
    
    def _format_model_name(self, model_id: str) -> str:
        """Format model ID into friendly name"""
        # Common model names
        name_map = {
            # OpenAI models
            'gpt-4o': 'GPT-4o (Latest, Fastest)',
            'gpt-4o-mini': 'GPT-4o Mini (Fast & Affordable)',
            'gpt-4-turbo-preview': 'GPT-4 Turbo (Preview)',
            'gpt-4-turbo': 'GPT-4 Turbo',
            'gpt-4': 'GPT-4 (Standard)',
            'gpt-4-32k': 'GPT-4 32K (Extended Context)',
            'gpt-3.5-turbo': 'GPT-3.5 Turbo',
            'gpt-3.5-turbo-16k': 'GPT-3.5 Turbo 16K',
            # Groq models (updated Feb 2026)
            'llama-3.3-70b-versatile': 'Llama 3.3 70B Versatile (Recommended, FREE)',
            'llama-3.1-8b-instant': 'Llama 3.1 8B Instant (Fastest, FREE)',
            'llama3-groq-70b-8192-tool-use-preview': 'Llama 3 Groq 70B Tool Use (FREE)',
            'llama3-groq-8b-8192-tool-use-preview': 'Llama 3 Groq 8B Tool Use (FREE)',
            'mixtral-8x7b-32768': 'Mixtral 8x7B (32K Context, FREE)',
            'gemma2-9b-it': 'Gemma 2 9B (Google, FREE)',
            'gemma-7b-it': 'Gemma 7B (Google, FREE)',
        }
        
        # If not in map, create friendly name
        if model_id in name_map:
            return name_map[model_id]
        
        # Auto-format: add FREE tag for Groq
        formatted = model_id.replace('-', ' ').title()
        if self.provider == "groq":
            formatted += " (FREE)"
        
        return formatted
    
    def _get_groq_models(self) -> List[Dict[str, str]]:
        """Return fallback Groq model list (updated Feb 2026)"""
        return [
            {'id': 'llama-3.3-70b-versatile', 'name': 'Llama 3.3 70B Versatile (Recommended, FREE)'},
            {'id': 'llama-3.1-8b-instant', 'name': 'Llama 3.1 8B Instant (Fastest, FREE)'},
            {'id': 'mixtral-8x7b-32768', 'name': 'Mixtral 8x7B (32K Context, FREE)'},
            {'id': 'gemma2-9b-it', 'name': 'Gemma 2 9B (Google, FREE)'},
        ]
    
    def _get_gemini_models(self) -> List[Dict[str, str]]:
        """Return fallback Gemini model list (updated Feb 2026)"""
        return [
            {'id': 'gemini-1.5-flash-latest', 'name': 'Gemini 1.5 Flash Latest (Fastest, FREE - 1M tokens/min)'},
            {'id': 'gemini-1.5-pro-latest', 'name': 'Gemini 1.5 Pro Latest (Best Quality, FREE - 1M tokens/min)'},
            {'id': 'gemini-2.0-flash-exp', 'name': 'Gemini 2.0 Flash Experimental (Latest, FREE)'},
        ]
    
    def _get_default_models(self) -> List[Dict[str, str]]:
        """Return default model list when API call fails"""
        if self.provider == "groq":
            return self._get_groq_models()
        elif self.provider == "gemini":
            return self._get_gemini_models()
        
        return [
            {'id': 'gpt-4o', 'name': 'GPT-4o (Latest, Fastest)'},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini (Fast & Affordable)'},
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo'},
            {'id': 'gpt-4', 'name': 'GPT-4 (Standard)'},
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo'},
        ]
    
    def call_gpt(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> Optional[str]:
        """
        Make a chat completion API call.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Response text or None on error
        """
        if not self.client:
            logger.error(f"Cannot call {self.provider.upper()} API: client not initialized")
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            provider_name = "Gemini" if self.provider == "gemini" else ("Groq" if self.provider == "groq" else "OpenAI")
            error_str = str(e)
            
            # Enhanced error logging with full details
            logger.error(f"{provider_name} API call failed: {e}")
            
            # Check for rate limit error
            if "rate_limit" in error_str.lower() or "429" in error_str:
                logger.warning(f"""\n{'='*80}
⚠️  RATE LIMIT EXCEEDED - {provider_name}
{'='*80}
Model: {model}
Error: {error_str}

SOLUTIONS:
1. Wait for the time specified in the error message
2. Reduce 'Max Agent Iterations' in Advanced Settings (default: 3)
3. Disable Multi-Agent Validation temporarily
4. Switch to a different provider:
   - Gemini: 1M tokens/min (Best free tier) - https://aistudio.google.com/app/apikey
   - Groq: 100k tokens/day - https://console.groq.com/keys
5. For Groq: Upgrade to Dev Tier at https://console.groq.com/settings/billing
6. For OpenAI: Check usage at https://platform.openai.com/usage
{'='*80}""")
            
            # Re-raise with provider-specific message
            raise Exception(f"{provider_name} API Error: {error_str}")


# Global helper instance (initialized when API key is set)
openai_helper = None

def get_openai_helper(api_key: str = None, provider: str = "openai") -> OpenAIHelper:
    """Get or create global helper instance"""
    global openai_helper
    if api_key:
        openai_helper = OpenAIHelper(api_key, provider)
    return openai_helper
