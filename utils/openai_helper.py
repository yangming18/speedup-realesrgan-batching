#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAI/Groq/Gemini Helper Functions
Handles API calls and model listings for OpenAI, Groq, and Google Gemini APIs
"""

import logging
import time
import json
import re
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
        self._models_cache = None  # Cache for model list
        self._cache_timestamp = 0  # When cache was last updated
        self._cache_ttl = 300  # Cache TTL in seconds (5 minutes)
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
        Uses caching to avoid repeated API calls.
        
        Returns:
            List of dicts with 'id' and 'name' keys
        """
        if not self.client:
            logger.warning("Cannot get models: client not initialized")
            return self._get_default_models()
        
        # Check cache
        current_time = time.time()
        if self._models_cache and (current_time - self._cache_timestamp) < self._cache_ttl:
            logger.debug(f"Using cached models for {self.provider}")
            return self._models_cache
        
        # Fetch models based on provider
        models = None
        
        # Gemini: fetch from API
        if self.provider == "gemini":
            try:
                models_response = self.client.models.list()
                
                # Filter for chat-compatible Gemini models only
                gemini_models = []
                for model in models_response.data:
                    model_id = model.id
                    
                    # Include only Gemini models (2.5, 3.x)
                    if 'gemini' not in model_id.lower():
                        continue
                    
                    # Skip specialized/non-chat variants
                    skip_keywords = [
                        'embedding',      # Embedding models
                        'vision',         # Vision-only models
                        'imagen',         # Image generation
                        'tts',            # Text-to-speech
                        'audio',          # Audio-specific (native-audio)
                        'image',          # Image generation/processing
                        'computer-use',   # Computer use models
                        'robotics',       # Robotics models
                        'lite',           # Lite versions (less capable)
                        '-001',           # Specific snapshots when base exists
                        '-latest',        # Latest aliases (prefer versioned)
                        'customtools'     # Custom tools variants
                    ]
                    
                    if any(skip in model_id.lower() for skip in skip_keywords):
                        continue
                    
                    name = self._format_model_name(model_id)
                    gemini_models.append({
                        'id': model_id,
                        'name': name
                    })
                
                # Sort: Pro models first, then Flash, then by version (3.x > 2.x)
                gemini_models.sort(key=lambda x: (
                    'pro' not in x['id'].lower(),  # Pro first
                    'flash' not in x['id'].lower(),  # Flash second
                    '3' not in x['id'],  # Version 3.x before 2.x
                    x['id']
                ))
                
                logger.info(f"Found {len(gemini_models)} Gemini chat models from API (filtered from {len(models_response.data)} total)")
                models = gemini_models if gemini_models else self._get_gemini_models()
                
            except Exception as e:
                logger.warning(f"Failed to retrieve Gemini models from API: {e}")
                logger.info("Using fallback Gemini model list")
                models = self._get_gemini_models()
        
        # Groq: fetch from API
        elif self.provider == "groq":
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
                
                logger.info(f"Found {len(groq_models)} Groq models from API")
                models = groq_models if groq_models else self._get_groq_models()
                
            except Exception as e:
                logger.warning(f"Failed to retrieve Groq models from API: {e}")
                logger.info("Using fallback Groq model list")
                models = self._get_groq_models()
        
        # OpenAI: fetch from API
        else:
            try:
                models_response = self.client.models.list()
                
                # Filter for GPT models suitable for chat completion
                chat_models = []
                for model in models_response.data:
                    model_id = model.id
                    # Include GPT-4, GPT-3.5, and GPT-4o variants
                    if any(prefix in model_id for prefix in ['gpt-4', 'gpt-3.5']):
                        # Exclude fine-tuned, instruct-only, or deprecated models
                        if not any(skip in model_id for skip in ['instruct', 'vision', 'audio']):
                            # Create friendly name
                            name = self._format_model_name(model_id)
                            chat_models.append({
                                'id': model_id,
                                'name': name
                            })
                
                # Sort by model name (GPT-4 first, then GPT-3.5)
                chat_models.sort(key=lambda x: (
                    '4o' not in x['id'],  # GPT-4o first
                    '4-turbo' not in x['id'],  # GPT-4 Turbo second
                    '4' not in x['id'],  # GPT-4 base third
                    x['id']  # Then alphabetically
                ))
                
                logger.info(f"Found {len(chat_models)} OpenAI models from API")
                models = chat_models if chat_models else self._get_default_models()
                
            except Exception as e:
                logger.warning(f"Failed to retrieve OpenAI models from API: {e}")
                logger.info("Using fallback OpenAI model list")
                models = self._get_default_models()
        
        # Cache the results
        if models:
            self._models_cache = models
            self._cache_timestamp = current_time
            logger.debug(f"Cached {len(models)} models for {self.provider}")
        
        return models if models else self._get_default_models()
    
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
            # Gemini models (updated March 2026)
            'gemini-2.5-flash': 'Gemini 2.5 Flash (Fastest, FREE - 1M tokens/min)',
            'gemini-2.5-pro': 'Gemini 2.5 Pro (Best Quality, FREE - 1M tokens/min)',
            'gemini-3-flash-preview': 'Gemini 3 Flash Preview (Latest, FREE)',
            'gemini-3-pro-preview': 'Gemini 3 Pro Preview (Advanced, FREE)',
            'gemini-3.1-flash-preview': 'Gemini 3.1 Flash Preview (Latest, FREE)',
            'gemini-3.1-pro-preview': 'Gemini 3.1 Pro Preview (Advanced, FREE)',
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
        """Return fallback Gemini model list (updated Feb 2026 - OpenAI-compatible endpoint)"""
        return [
            {'id': 'gemini-2.5-flash', 'name': 'Gemini 2.5 Flash (Fastest, FREE - 1M tokens/min)'},
            {'id': 'gemini-2.5-pro', 'name': 'Gemini 2.5 Pro (Best Quality, FREE - 1M tokens/min)'},
            {'id': 'gemini-3-flash-preview', 'name': 'Gemini 3 Flash Preview (Latest, FREE)'},
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
        max_tokens: int = 4000,
        max_retries: int = 3,
        base_delay: float = 1.0,
        progress_callback=None
    ) -> Optional[str]:
        """
        Make a chat completion API call with automatic retry on rate limits.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts for rate limit errors (default: 3)
            base_delay: Base delay in seconds between requests (default: 1.0)
            progress_callback: Optional callback(progress, message) to update UI
        
        Returns:
            Response text or None on error
        """
        if not self.client:
            logger.error(f"Cannot call {self.provider.upper()} API: client not initialized")
            return None
        
        provider_name = "Gemini" if self.provider == "gemini" else ("Groq" if self.provider == "groq" else "OpenAI")
        
        # Apply base delay to avoid hitting rate limits (be nice to the API)
        time.sleep(base_delay)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                
                # Enhanced error logging
                logger.error(f"{provider_name} API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Check QUOTA FIRST (before rate limit) because RESOURCE_EXHAUSTED can mean both
                # If it's daily quota (GenerateRequestsPerDay), don't retry even if RESOURCE_EXHAUSTED
                is_quota_exceeded = (
                    "quota exceeded" in error_str.lower() or 
                    "GenerateRequestsPerDay" in error_str or
                    "GenerateContentInputTokensPerModelPerDay" in error_str or
                    "current quota" in error_str.lower() or
                    ("limit: 0" in error_str and "PerDay" in error_str)  # Exhausted daily limit
                )
                
                # Check for rate limit error (429) - retriable ONLY if not quota exceeded
                is_rate_limit = (
                    not is_quota_exceeded and (  # Only rate limit if NOT quota exceeded
                        "rate_limit" in error_str.lower() or 
                        "429" in error_str or 
                        ("RESOURCE_EXHAUSTED" in error_str and "PerMinute" in error_str)
                    )
                )
                
                if is_quota_exceeded:
                    # Quota exceeded - don't retry, inform user immediately
                    logger.error(f"""
{'='*80}
❌ DAILY QUOTA EXCEEDED - {provider_name}
{'='*80}
You've reached your daily quota limit (not just rate limit).

🔑 IMMEDIATE SOLUTIONS:
1. ⭐ Switch to Groq (RECOMMENDED - 100k tokens/day FREE):
   - Get API key: https://console.groq.com/keys
   - Add in Settings tab
   - Select "Groq" as provider

2. ⚡ Use Single Pass mode (saves API calls):
   - Select "Single Pass" validation mode
   - Avoids 12-16 API calls of Multi-Agent mode

3. Wait for quota reset:
   - Gemini Free Tier Limits:
     * gemini-2.5-flash: 20 requests/day
     * gemini-2.5-pro: 10 requests/day (you hit this)
   - Resets at midnight Pacific Time
   - Check usage: https://ai.dev/rate-limit

4. Upgrade your plan:
   - Gemini: https://ai.google.dev/pricing
   - OpenAI: https://platform.openai.com/account/billing

💡 TIP: With Single Pass + Groq, you can generate hundreds of subtitles/day!
{'='*80}
""")
                    raise Exception(f"{provider_name} API Error: {error_str}")
                
                elif is_rate_limit:
                    # Extract retry delay from error message
                    retry_delay = self._extract_retry_delay(error_str)
                    
                    if attempt < max_retries - 1:  # Don't retry on last attempt
                        wait_message = f"⏰ RATE LIMIT - Waiting {retry_delay}s before retry ({attempt + 2}/{max_retries})..."
                        logger.warning(wait_message)
                        print(f"\n{'='*80}\n⚠️  {wait_message}\n{'='*80}\n")  # Console visibility
                        
                        # Update UI if callback available
                        if progress_callback:
                            progress_callback(None, f"⚠️  {wait_message}")
                        
                        time.sleep(retry_delay)
                        
                        retry_message = f"✓ Wait completed. Retrying ({attempt + 2}/{max_retries})..."
                        logger.info(retry_message)
                        print(f"\n{retry_message}\n")
                        
                        if progress_callback:
                            progress_callback(None, retry_message)
                        
                        continue  # Retry
                    else:
                        # Last attempt failed, show full error message
                        logger.warning(f"""\n{'='*80}
⚠️  RATE LIMIT EXCEEDED - {provider_name} (Max retries reached)
{'='*80}
Model: {model}
All {max_retries} attempts failed.
Last retry delay was: {retry_delay}s

SOLUTIONS:
1. Wait {retry_delay}s and try again manually
2. Reduce 'Max Agent Iterations' in Advanced Settings (default: 3, try 1-2)
3. Disable Multi-Agent Validation temporarily
4. Switch to a different provider:
   - Gemini 2.5 Flash: Faster, lower limits but sufficient for most tasks
   - Groq: 100k tokens/day - https://console.groq.com/keys
   - OpenAI: Paid but reliable - https://platform.openai.com/api-keys
5. Check your quota at:
   - Gemini: https://ai.dev/rate-limit
   - Groq: https://console.groq.com/settings/billing
   - OpenAI: https://platform.openai.com/usage
{'='*80}""")
                
                # Re-raise with provider-specific message
                raise Exception(f"{provider_name} API Error: {error_str}")
        
        return None
    
    def _extract_retry_delay(self, error_str: str) -> float:
        """
        Extract retry delay from error message.
        
        Common formats:
        - "Please retry in 14.624966946s."
        - "'retryDelay': '14s'"
        - "Retry after 10 seconds"
        
        Returns:
            Delay in seconds (default: 15.0 if not found)
        """
        # Try to find "retry in Xs" pattern
        match = re.search(r'retry in ([0-9.]+)s', error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Try to find "retryDelay': 'Xs'" pattern
        match = re.search(r"retryDelay['\"]:\s*['\"]([0-9.]+)s?['\"]", error_str)
        if match:
            return float(match.group(1))
        
        # Try to find "Retry after X" pattern
        match = re.search(r'retry after ([0-9.]+)', error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Default delay if not found
        logger.warning("Could not extract retry delay from error, using default 15s")
        return 15.0


# Global helper instance (initialized when API key is set)
openai_helper = None

def get_openai_helper(api_key: str = None, provider: str = "openai") -> OpenAIHelper:
    """Get or create global helper instance"""
    global openai_helper
    if api_key:
        openai_helper = OpenAIHelper(api_key, provider)
    return openai_helper
