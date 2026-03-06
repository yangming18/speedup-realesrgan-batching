#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Subititles Tab
From Audio to Subtitles - Generate accurate subtitles from audio and lyrics
"""

import gradio as gr
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import tempfile
from faster_whisper import WhisperModel
from utils import api_key_manager, get_openai_helper
from utils.subtitle_agents import SubtitleAgentSystem

logger = logging.getLogger(__name__)


class SubtitlesTab:
    """Manages the subtitles generation tab"""
    
    def __init__(self, i18n_manager):
        self.i18n = i18n_manager
        self.whisper_model = None
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> dict:
        """Load prompt templates"""
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.json"
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return {}
    
    def update_models_for_provider(self, provider: str, task: str = "clean") -> dict:
        """
        Update model choices when provider changes.
        Dynamically loads models from API if possible.
        
        Args:
            provider: "openai", "groq", or "gemini"
            task: "clean" or "gen" (generation)
        
        Returns:
            gr.update() dict for Dropdown
        """
        # Try to get API key for the provider
        api_key = None
        try:
            if provider == "openai":
                api_key = api_key_manager.get_api_key("OPENAI_API_KEY")
            elif provider == "groq":
                api_key = api_key_manager.get_api_key("GROQ_API_KEY")
            elif provider == "gemini":
                api_key = api_key_manager.get_api_key("GEMINI_API_KEY")
        except:
            pass  # If key doesn't exist, use fallback
        
        # If we have API key, try to load models from API
        if api_key:
            try:
                from utils.openai_helper import OpenAIHelper
                helper = OpenAIHelper(api_key=api_key, provider=provider)
                models_list = helper.get_available_models()
                
                if models_list:
                    # Extract just the model IDs
                    choices = [m['id'] for m in models_list]
                    
                    # Select appropriate default based on task
                    if task == "clean":
                        # Prefer fastest/cheapest model for cleaning
                        if provider == "groq":
                            value = next((m for m in choices if '8b' in m.lower() and 'instant' in m.lower()), choices[0])
                        elif provider == "gemini":
                            value = next((m for m in choices if 'flash' in m.lower()), choices[0])
                        else:  # openai
                            value = next((m for m in choices if 'mini' in m.lower()), choices[0])
                    else:  # generation
                        # Prefer best quality model for generation
                        if provider == "groq":
                            value = next((m for m in choices if '70b' in m.lower() or '405b' in m.lower()), choices[0])
                        elif provider == "gemini":
                            value = next((m for m in choices if 'pro' in m.lower()), choices[0])
                        else:  # openai
                            value = next((m for m in choices if 'gpt-4o' in m.lower() and 'mini' not in m.lower()), choices[0])
                    
                    logger.info(f"Loaded {len(choices)} models from {provider} API")
                    return gr.update(choices=choices, value=value)
            except Exception as e:
                logger.warning(f"Failed to load models from {provider} API: {e}")
        
        # Fallback to hardcoded lists
        logger.debug(f"Using fallback model list for {provider}")
        if provider == "groq":
            if task == "clean":
                choices = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
                value = "llama-3.1-8b-instant"
            else:  # generation
                choices = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
                value = "llama-3.3-70b-versatile"
        elif provider == "gemini":
            if task == "clean":
                choices = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"]
                value = "gemini-2.5-flash"
            else:  # generation
                choices = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview"]
                value = "gemini-2.5-pro"
        else:  # openai
            if task == "clean":
                choices = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
                value = "gpt-4o-mini"
            else:  # generation
                choices = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
                value = "gpt-4o"
        
        return gr.update(choices=choices, value=value)
    
    def get_whisper_model(self, model_size: str, device: str):
        """Load or get cached Whisper model"""
        if self.whisper_model is None or self.whisper_model.model_size_or_path != model_size:
            logger.info(f"Loading Whisper model: {model_size} on {device}")
            self.whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type="float32" if device == "cpu" else "float16"
            )
        return self.whisper_model
    
    def clean_lyrics(
        self,
        lyrics: str,
        is_clean: bool,
        model: str,
        provider: str = "openai",
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Clean lyrics using GPT.
        
        Args:
            lyrics: Raw lyrics text
            is_clean: Skip cleaning if True
            model: Model to use
            provider: "openai" or "groq"
        
        Returns:
            Tuple of (cleaned_lyrics, status_message)
        """
        if not lyrics.strip():
            return "", "❌ Please provide lyrics"
        
        if is_clean:
            return lyrics, "✓ Lyrics already clean - no processing needed"
        
        # Get API key for provider
        key_name = f"{provider.upper()}_API_KEY"
        api_key = api_key_manager.get_api_key(key_name)
        provider_label = "Gemini" if provider == "gemini" else ("Groq" if provider == "groq" else "OpenAI")
        
        if not api_key:
            return lyrics, f"❌ No {provider_label} API key found. Please configure in Settings tab."
        
        progress(0.3, f"Cleaning lyrics with {provider_label}...")
        
        try:
            helper = get_openai_helper(api_key, provider)
            
            # Prepare messages
            prompt_config = self.prompts.get('lyrics_cleaner', {})
            system_prompt = prompt_config.get('system_prompt', '')
            user_template = prompt_config.get('user_prompt_template', '')
            user_prompt = user_template.replace('{lyrics}', lyrics)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call GPT
            cleaned = helper.call_gpt(
                messages=messages,
                model=model,
                temperature=0.3,  # Low temperature for consistent cleaning
                max_tokens=4000
            )
            
            if cleaned:
                progress(1.0, "Lyrics cleaned successfully!")
                return cleaned, "✓ Lyrics cleaned successfully! You can edit them if needed."
            else:
                return lyrics, "❌ GPT call failed. Check your API key."
                
        except Exception as e:
            logger.error(f"Error cleaning lyrics: {e}")
            return lyrics, f"❌ Error: {str(e)}"
    
    def transcribe_audio(
        self,
        audio_path: str,
        whisper_model: str,
        device: str,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Transcribe audio with word-level timestamps using Whisper.
        
        Returns:
            Tuple of (transcription_json, status_message)
        """
        if not audio_path:
            return "[]", "❌ Please upload an audio file"
        
        try:
            progress(0.2, f"Loading Whisper model ({whisper_model})...")
            model = self.get_whisper_model(whisper_model, device)
            
            progress(0.4, "Transcribing audio (this may take a while)...")
            
            # Transcribe with word-level timestamps
            segments, info = model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en",  # Can be auto-detected or made configurable
                vad_filter=False,  # DISABLED: VAD can stop prematurely on long songs
                beam_size=5,
                condition_on_previous_text=False  # Avoid context-based stopping
            )
            
            logger.info(f"Audio duration: {info.duration:.2f}s, language: {info.language}")
            
            # Convert generator to list to ensure all segments are processed
            progress(0.6, "Processing segments...")
            segments_list = list(segments)
            logger.info(f"Total segments received: {len(segments_list)}")
            
            # Extract BOTH segment-level (standard subtitles) and word-level data
            segments_data = []  # Standard subtitles
            words_data = []     # Word-by-word
            total_segments = len(segments_list)
            
            for idx, segment in enumerate(segments_list):
                if idx % 10 == 0:  # Update progress every 10 segments
                    progress(0.6 + (0.3 * idx / total_segments), f"Processing segment {idx+1}/{total_segments}...")
                
                # Extract segment-level (standard subtitle grouping)
                segments_data.append({
                    'text': segment.text.strip(),
                    'start': round(segment.start, 3),
                    'end': round(segment.end, 3)
                })
                
                # Extract word-level (precise timing for each word)
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        words_data.append({
                            'word': word.word.strip(),
                            'start': round(word.start, 3),
                            'end': round(word.end, 3)
                        })
            
            # Store both levels in transcript data
            transcript_data = {
                'segments': segments_data,  # For standard mode
                'words': words_data         # For ultra-detailed mode
            }
            
            logger.info(f"Extracted: {len(segments_data)} segments, {len(words_data)} words")
            progress(1.0, f"Transcribed: {len(segments_data)} segments, {len(words_data)} words!")
            
            # Return transcript + audio duration for multi-agent validation
            return (
                json.dumps(transcript_data, indent=2), 
                f"✓ Transcription complete: {len(segments_data)} segments, {len(words_data)} words",
                info.duration  # Audio duration for multi-agent system
            )
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "[]", f"❌ Transcription failed: {str(e)}", 0.0
    
    def generate_subtitles(
        self,
        transcript_json: str,
        cleaned_lyrics: str,
        model: str,
        subtitle_format: str,
        ultra_mode: str,
        pause_threshold: float,
        max_chars_per_line: int,
        max_lines: int,
        provider: str = "openai",
        validation_mode: str = "single_pass",
        audio_duration: float = 0.0,
        max_iterations: int = 3,
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Generate subtitles using GPT based on Whisper transcript + cleaned lyrics.
        
        Args:
            transcript_json: JSON string with Whisper word timestamps
            cleaned_lyrics: Clean reference lyrics
            model: Model to use
            subtitle_format: SRT, VTT, or ASS
            ultra_mode: disabled, basic, or word_by_word
            pause_threshold: Pause duration for new subtitle
            max_chars_per_line: Max characters per line
            max_lines: Max lines per subtitle
            provider: "openai" or "groq"
            validation_mode: 'single_pass' or 'multi_agent'
            audio_duration: Total audio duration in seconds
        
        Returns:
            Tuple of (subtitles_content, status_message, validation_log)
        """
        if not transcript_json or transcript_json == "[]":
            return "", "❌ Please transcribe audio first", ""
        
        if not cleaned_lyrics.strip():
            return "", "❌ Please provide and clean lyrics first", ""
        
        # Get API key for provider
        key_name = f"{provider.upper()}_API_KEY"
        api_key = api_key_manager.get_api_key(key_name)
        provider_label = "Gemini" if provider == "gemini" else ("Groq" if provider == "groq" else "OpenAI")
        
        if not api_key:
            return "", f"❌ No {provider_label} API key found. Please configure in Settings tab.", ""
        
        try:
            transcript_data = json.loads(transcript_json)
            
            progress(0.3, f"Generating subtitles with {provider_label}...")
            
            helper = get_openai_helper(api_key, provider)
            
            # Select data source based on ultra_mode
            # Use segments (standard grouping) for disabled/basic modes
            # Use words (word-by-word) for word_by_word mode
            if ultra_mode == "word_by_word":
                source_data = transcript_data.get('words', transcript_data)  # Fallback to old format
            else:
                source_data = transcript_data.get('segments', transcript_data)  # Fallback to old format
            
            # Build prompt based on mode
            mode_instructions = self._get_mode_instructions(
                ultra_mode,
                pause_threshold,
                max_chars_per_line,
                max_lines
            )
            
            # Format transcript for GPT
            formatted_transcript = self._format_transcript(source_data, ultra_mode)
            
            # Add word count info for word-by-word mode
            word_count_info = ""
            if ultra_mode == "word_by_word":
                expected_words = len(source_data)
                word_count_info = f"\n⚠️  CRITICAL: You MUST generate exactly {expected_words} subtitles (one per word).\n"
            
            # Different system prompts for word-by-word vs standard modes
            if ultra_mode == "word_by_word":
                system_prompt = f"""🎯 KARAOKE SUBTITLE GENERATOR

🚨 ABSOLUTE REQUIREMENT 🚨
Generate ALL {len(source_data)} subtitles. No exceptions.

INPUT: {len(source_data)} words
OUTPUT: {len(source_data)} subtitles

RULES:
1. ONE subtitle per word (no grouping)
2. Use exact Whisper timestamps
3. Fix transcription errors using lyrics
4. NEVER STOP EARLY - complete ALL {len(source_data)} words
5. Format: Concise SRT (no extra spacing)

{mode_instructions}

START: Subtitle #1
END: Subtitle #{len(source_data)}

DO NOT write explanations, DO NOT stop at a round number, DO NOT summarize.
Output the COMPLETE {subtitle_format} file from 1 to {len(source_data)}.
"""
            else:
                system_prompt = f"""You are an expert subtitle generator. You create {subtitle_format} format subtitles.

Your task:
1. Match the Whisper timestamps with the reference lyrics
2. Generate properly formatted {subtitle_format} subtitles
3. Follow subtitle best practices
{word_count_info}
{mode_instructions}

Output ONLY the {subtitle_format} formatted subtitles, nothing else."""

            # Use full lyrics - no truncation
            lyrics_preview = cleaned_lyrics
            
            user_prompt = f"""Reference Lyrics:
{lyrics_preview}

---

Whisper Transcript ({len(source_data)} words):
{formatted_transcript}

---

GENERATE ALL {len(source_data)} SUBTITLES NOW.
Start from word 1, end at word {len(source_data)}.
Do not stop until complete."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call GPT with appropriate max_tokens for word-by-word mode
            if ultra_mode == "word_by_word":
                # Gemini max: 8,192 tokens
                # 266 subtitles ≈ 5,000 tokens (well within limit)
                max_tokens = 8192  # Use Gemini's maximum
            else:
                max_tokens = 8000
            
            progress(0.5, f"Generating ALL {len(source_data)} subtitles with {provider_label}...")
            subtitles = helper.call_gpt(
                messages=messages,
                model=model,
                temperature=0.1,  # VERY low temperature to follow instructions precisely
                max_tokens=max_tokens,
                progress_callback=lambda p, msg: progress(0.4 + p * 0.3 if p is not None else 0.5, msg)
            )
            
            if not subtitles:
                return "", f"❌ {provider_label} API call failed - no response received", ""
            
            # COMPLETENESS CHECK for word-by-word mode (ONLY in multi_agent mode to save quota)
            if ultra_mode == "word_by_word" and validation_mode == "multi_agent":
                subtitle_count = subtitles.count('\n\n') + 1  # Count subtitle blocks
                expected_count = len(source_data)
                
                if subtitle_count < expected_count:
                    progress(0.6, f"⚠️ Incomplete: {subtitle_count}/{expected_count} subtitles. Generating missing parts...")
                    
                    # Calculate missing range
                    missing_count = expected_count - subtitle_count
                    start_idx = subtitle_count  # Start from where it stopped
                    
                    # Extract missing words from source_data
                    missing_words = source_data[start_idx:expected_count]
                    missing_transcript = self._format_transcript(missing_words, ultra_mode)
                    
                    # Generate completion prompt
                    completion_prompt = f"""Continue generating subtitles from #{start_idx + 1} to #{expected_count}.

Already generated: {subtitle_count} subtitles
Missing: {missing_count} subtitles

Continue from here:
{missing_transcript}

Generate subtitles #{start_idx + 1} through #{expected_count} in {subtitle_format} format.
Start with index {start_idx + 1}, use exact Whisper timestamps."""
                    
                    completion = helper.call_gpt(
                        messages=[
                            {"role": "system", "content": f"You are a subtitle generator. Continue the subtitle file."},
                            {"role": "user", "content": completion_prompt}
                        ],
                        model=model,
                        temperature=0.1,
                        max_tokens=max_tokens,
                        progress_callback=lambda p, msg: progress(0.6 + p * 0.1 if p is not None else 0.65, msg)
                    )
                    
                    if completion:
                        # Append missing subtitles
                        subtitles = subtitles.rstrip() + "\n\n" + completion.strip()
                        progress(0.7, f"✓ Completed: Now have {subtitle_count + completion.count(chr(10)+chr(10)) + 1} subtitles")
            elif ultra_mode == "word_by_word" and validation_mode == "single_pass":
                # In single_pass mode, just count and warn if incomplete (don't make extra API calls)
                subtitle_count = subtitles.count('\n\n') + 1  # Count subtitle blocks
                expected_count = len(source_data)
                if subtitle_count < expected_count:
                    logger.warning(f"⚠️ Single Pass: Generated {subtitle_count}/{expected_count} subtitles (incomplete)")
                    progress(0.6, f"⚠️ Generated {subtitle_count}/{expected_count} subtitles. Use Multi-Agent for auto-completion.")
            
            validation_log = ""
            
            # Multi-Agent Validation (if enabled)
            if validation_mode == "multi_agent" and audio_duration > 0:
                progress(0.7, "Running multi-agent validation...")
                
                try:
                    agent_system = SubtitleAgentSystem(helper, model, max_iterations=max_iterations)
                    
                    # Pass full context to agents including UI parameters
                    agent_context = {
                        'audio_duration': audio_duration,
                        'ultra_detailed_mode': ultra_mode,
                        'max_chars_per_line': max_chars_per_line,
                        'max_lines_per_subtitle': max_lines,
                        'content_type': 'song',  # KEY: Tell agents this is a song (long gaps are normal)
                        'subtitle_mode': ultra_mode  # EXPLICIT: Tell agents which mode we're in
                    }
                    
                    # Use words data for agent analysis (most detailed)
                    whisper_words = transcript_data.get('words', transcript_data) if isinstance(transcript_data, dict) else transcript_data
                    
                    corrected_subtitles, log_lines = agent_system.validate_and_correct_subtitles(
                        subtitles=subtitles,
                        lyrics=cleaned_lyrics,
                        whisper_data=whisper_words,
                        audio_duration=audio_duration,
                        subtitle_format=subtitle_format,
                        context=agent_context,
                        progress_callback=lambda p, msg: progress(0.7 + p * 0.3 if p is not None else 0.7, msg)
                    )
                    
                    subtitles = corrected_subtitles
                    validation_log = "\n".join(log_lines)
                    
                    progress(1.0, "Multi-agent validation complete!")
                    
                except Exception as e:
                    logger.error(f"Multi-agent validation error: {e}")
                    error_details = str(e)
                    
                    # Format error message with provider info
                    if "rate_limit" in error_details.lower() or "429" in error_details:
                        validation_log = f"""⚠️  RATE LIMIT EXCEEDED

Provider: {provider_label}
Model: {model}

{error_details}

SOLUTIONS:
1. Reduce 'Max Agent Iterations' (currently: {max_iterations}) to 1-3
2. Disable Multi-Agent Validation temporarily
3. Switch to a different provider:
   - ✨ Gemini: 1M tokens/min (Best free tier)
   - Groq: 100k tokens/day
4. Wait for the cooldown period
5. For Groq: Upgrade at https://console.groq.com/settings/billing
6. For OpenAI: Check usage at https://platform.openai.com/usage

Using original subtitles without validation."""
                    else:
                        validation_log = f"⚠️ Multi-agent validation failed: {error_details}\n\nUsing original subtitles."
            
            status_msg = f"✓ {subtitle_format} subtitles generated successfully!"
            if validation_mode == "multi_agent" and "RATE LIMIT" not in validation_log:
                status_msg += " (Multi-agent validated)"
            
            return subtitles, status_msg, validation_log
                
        except Exception as e:
            logger.error(f"Error generating subtitles: {e}")
            error_msg = str(e)
            
            # Check if it's QUOTA EXCEEDED (daily limit) - more serious than rate limit
            is_quota_exceeded = (
                "quota exceeded" in error_msg.lower() or 
                "GenerateRequestsPerDay" in error_msg or
                "current quota" in error_msg.lower() or
                ("limit: 0" in error_msg and "PerDay" in error_msg)
            )
            
            # Check if it's rate limit (temporary, retriable)
            is_rate_limit = (
                "rate_limit" in error_msg.lower() or 
                "429" in error_msg
            )
            
            if is_quota_exceeded:
                # Daily quota exhausted - need immediate action
                detailed_error = f"""❌ DAILY QUOTA EXHAUSTED

Provider: {provider_label}
Model: {model}

⚠️ You've used all your daily requests. This is NOT a rate limit - you must wait or switch provider.

🔥 IMMEDIATE SOLUTIONS (choose one):

1. ⭐ SWITCH TO GROQ (RECOMMENDED):
   → Get free API key: https://console.groq.com/keys
   → Add in Settings tab
   → Select "Groq" as provider
   → Limits: 100k tokens/day (much higher!)

2. ⚡ Switch to OpenAI:
   → Pay-as-you-go model
   → No daily limits
   → Add key in Settings tab

3. ⏰ Wait for reset:
   → Gemini free tier limits:
      • gemini-2.5-flash: 20 requests/day
      • gemini-2.5-pro: 10 requests/day
   → Resets at midnight Pacific Time
   → Track usage: https://ai.dev/rate-limit

💡 TIP: Use "Single Pass" mode to save API calls (currently uses 1 call vs 12-16 in Multi-Agent)

Error Details:
{error_msg}"""
            
            elif is_rate_limit:
                # Temporary rate limit - can retry
                detailed_error = f"""❌ RATE LIMIT (temporary - can retry)

Provider: {provider_label}
Model: {model}

Error Details:
{error_msg}

SOLUTIONS:
1. Reduce 'Max Agent Iterations' slider (currently: {max_iterations})
2. Switch to "Single Pass" validation mode (saves 12-16 API calls)
3. Switch to a different provider:
   - ✨ Groq: 100k tokens/day - Good alternative
   - ⭐ Gemini: Try gemini-2.5-flash (higher limits than pro)
4. Wait for the cooldown period specified above
5. Provider limits comparison:
   - Gemini Free: 15 RPM, 1M tokens/min
   - Groq Free: 30 RPM, 100k tokens/day"""
            else:
                detailed_error = f"❌ Error from {provider_label} API:\n\n{error_msg}"
            
            return "", detailed_error, ""
    
    def _get_mode_instructions(
        self,
        ultra_mode: str,
        pause_threshold: float,
        max_chars_per_line: int,
        max_lines: int
    ) -> str:
        """Get mode-specific instructions for GPT"""
        
        if ultra_mode == "word_by_word":
            # Word-by-word mode: EVERY word must be included, one subtitle per word
            return """
🎯 WORD-BY-WORD MODE (KARAOKE):

MANDATORY RULES:
1. Generate EXACTLY one subtitle for each word in the transcript
2. Each subtitle contains EXACTLY one word (the word itself)
3. Use the EXACT start/end timestamps from Whisper for each word
4. If Whisper transcribed a word wrong, use the correct word from reference lyrics
5. Generate subtitles for ALL words - do NOT skip any word for any reason

⚠️  CRITICAL: DO NOT STOP EARLY!
- If you have 266 words, generate ALL 266 subtitles
- Do not generate just 10-20 as an example
- Do not stop at the end of a line/verse
- Continue processing until EVERY word has its subtitle

EXAMPLE INPUT (3 words):
[
  {"word": "hello", "start": 0.5, "end": 0.8},
  {"word": "world", "start": 0.9, "end": 1.2},
  {"word": "now", "start": 1.3, "end": 1.6}
]

REQUIRED OUTPUT (3 subtitles):
1
00:00:00,500 --> 00:00:00,800
Hello

2
00:00:00,900 --> 00:00:01,200
World

3
00:00:01,300 --> 00:00:01,600
Now

If you have N words in input, you MUST generate N subtitles in output.
"""
        
        # Standard modes: apply character/line limits
        base_rules = f"""
Subtitle Formatting Rules:
- Maximum {max_chars_per_line} characters per line
- Maximum {max_lines} lines per subtitle
- Use proper timing: don't split words across subtitles
- Maintain natural reading pace (150-180 words per minute)
"""
        
        if ultra_mode == "disabled":
            return base_rules + "\nGenerate standard subtitles grouping words naturally."
        
        elif ultra_mode == "basic":
            return base_rules + f"""
Ultra Detailed Mode: Basic Pause Handling
- When there's a pause of {pause_threshold} seconds or more between words, create a new subtitle
- Group words naturally within each subtitle
- Respect natural phrase and sentence boundaries
"""
        
        return base_rules
    
    def _format_transcript(self, transcript_data: List[Dict], ultra_mode: str = "disabled") -> str:
        """Format transcript data for GPT prompt"""
        lines = []
        for item in transcript_data:
            # Handle both word-level (has 'word' key) and segment-level (has 'text' key)
            content = item.get('word') or item.get('text', '')
            lines.append(f"[{item['start']}s - {item['end']}s] {content}")
        return "\n".join(lines)
    
    def parse_subtitles_to_dataframe(self, subtitles_text: str, format: str) -> List[List]:
        """
        Parse SRT/VTT/ASS subtitles into editable dataframe format.
        
        Returns:
            List of lists: [[index, start_time, end_time, text], ...]
        """
        if not subtitles_text or not subtitles_text.strip():
            return []
        
        data = []
        
        if format == "SRT":
            # Parse SRT format
            blocks = subtitles_text.strip().split('\n\n')
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    index = lines[0].strip()
                    timing = lines[1].strip()
                    text = '\n'.join(lines[2:])
                    
                    # Parse timing: 00:00:10,500 --> 00:00:13,000
                    if ' --> ' in timing:
                        start, end = timing.split(' --> ')
                        data.append([index, start.strip(), end.strip(), text])
        
        elif format == "VTT":
            # Parse WebVTT format
            lines = subtitles_text.strip().split('\n')
            i = 0
            index = 1
            while i < len(lines):
                line = lines[i].strip()
                # Skip WEBVTT header and empty lines
                if line.startswith('WEBVTT') or not line:
                    i += 1
                    continue
                
                # Check if line is a timestamp
                if '-->' in line:
                    timing = line
                    text_lines = []
                    i += 1
                    # Collect text until empty line
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    if ' --> ' in timing:
                        start, end = timing.split(' --> ')
                        text = '\n'.join(text_lines)
                        data.append([str(index), start.strip(), end.strip(), text])
                        index += 1
                i += 1
        
        elif format == "ASS":
            # Parse ASS format (basic parsing of Dialogue lines)
            lines = subtitles_text.strip().split('\n')
            index = 1
            for line in lines:
                if line.startswith('Dialogue:'):
                    # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
                    parts = line.split(',', 9)
                    if len(parts) >= 10:
                        start = parts[1].strip()
                        end = parts[2].strip()
                        text = parts[9].strip()
                        data.append([str(index), start, end, text])
                        index += 1
        
        return data
    
    def dataframe_to_subtitles(self, dataframe_data: List[List], format: str) -> str:
        """
        Convert edited dataframe data back to subtitle format.
        
        Args:
            dataframe_data: List of lists [[index, start, end, text], ...]
            format: SRT, VTT, or ASS
        
        Returns:
            Formatted subtitle string
        """
        if not dataframe_data:
            return ""
        
        lines = []
        
        if format == "SRT":
            for row in dataframe_data:
                if len(row) >= 4:
                    index, start, end, text = row[0], row[1], row[2], row[3]
                    lines.append(f"{index}\n{start} --> {end}\n{text}\n")
            return "\n".join(lines)
        
        elif format == "VTT":
            lines.append("WEBVTT\n")
            for row in dataframe_data:
                if len(row) >= 4:
                    start, end, text = row[1], row[2], row[3]
                    lines.append(f"{start} --> {end}\n{text}\n")
            return "\n".join(lines)
        
        elif format == "ASS":
            # Basic ASS format
            lines.append("[Script Info]\nTitle: Generated Subtitles\nScriptType: v4.00+\n")
            lines.append("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            lines.append("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n")
            lines.append("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            for row in dataframe_data:
                if len(row) >= 4:
                    start, end, text = row[1], row[2], row[3]
                    lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
            return "".join(lines)
        
        return ""
    
    def save_subtitles(self, subtitles: str, format: str) -> Tuple[str, str]:
        """
        Save subtitles to a temporary file for download.
        
        Returns:
            Tuple of (file_path, status_message)
        """
        if not subtitles:
            return None, "❌ No subtitles to save"
        
        try:
            # Create temp file with appropriate extension
            ext_map = {
                "SRT": ".srt",
                "VTT": ".vtt",
                "ASS": ".ass"
            }
            ext = ext_map.get(format, ".txt")
            
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                suffix=ext,
                delete=False
            ) as f:
                f.write(subtitles)
                file_path = f.name
            
            return file_path, f"✓ Subtitles saved: {Path(file_path).name}"
            
        except Exception as e:
            logger.error(f"Error saving subtitles: {e}")
            return None, f"❌ Save failed: {str(e)}"
    
    def create_tab(self):
        """Create the subtitles generation tab UI"""
        with gr.Tab("🎵 Audio to Subtitles"):
            gr.Markdown("""
            # From Audio to Subtitles
            
            Generate accurate subtitles from audio files using Whisper AI + GPT correction.
            
            **Workflow:**
            1. Configure your OpenAI API key in Settings tab
            2. Upload audio file and provide lyrics
            3. Clean lyrics (if needed)
            4. Transcribe audio with Whisper
            5. Generate subtitles with GPT
            """)
            
            # === STEP 1: Upload Files ===
            gr.Markdown("## 📁 Step 1: Upload Files")
            with gr.Row():
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                lyrics_input = gr.Textbox(
                    label="Lyrics",
                    placeholder="Paste your lyrics here...",
                    lines=10
                )
            
            is_clean_checkbox = gr.Checkbox(
                label="Lyrics are already clean (no AI tags like [Verse], [Chorus], etc.)",
                value=False
            )
            
            # === STEP 2: Clean Lyrics ===
            gr.Markdown("## 🧹 Step 2: Clean Lyrics")
            gr.Markdown("Remove AI formatting tags (Suno, Udio, Mureka) to get clean text.")
            
            provider_clean = gr.Radio(
                choices=[
                    ("OpenAI (Paid)", "openai"),
                    ("Groq (FREE)", "groq"),
                    ("Gemini (FREE - Best)", "gemini")
                ],
                value="gemini",
                label="API Provider",
                info="✨ Gemini: 1M tokens/min (Best free tier!)"
            )
            
            # Model selection in collapsible accordion to avoid dropdown positioning issues
            with gr.Accordion("🔧 Select Model", open=False):
                with gr.Row():
                    model_clean_openai = gr.Radio(
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                        value="gpt-4o-mini",
                        label="OpenAI Model",
                        info="Fast model recommended for cleaning",
                        visible=False
                    )
                    
                    model_clean_groq = gr.Radio(
                        choices=["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
                        value="llama-3.1-8b-instant",
                        label="Groq Model",
                        info="Fast model recommended for cleaning",
                        visible=False
                    )
                    
                    model_clean_gemini = gr.Radio(
                        choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"],
                        value="gemini-2.5-flash",
                        label="Gemini Model",
                        info="Fast model recommended for cleaning",
                        visible=True
                    )
            
            clean_btn = gr.Button("🧹 Clean Lyrics", variant="primary")
            
            cleaned_lyrics = gr.Textbox(
                label="Cleaned Lyrics (editable)",
                placeholder="Cleaned lyrics will appear here...",
                lines=10,
                interactive=True
            )
            
            clean_status = gr.Textbox(label="Status", interactive=False, lines=1)
            
            # === STEP 3: Transcribe Audio ===
            gr.Markdown("## 🎤 Step 3: Transcribe with Whisper")
            with gr.Accordion("🔧 Select Model & Device", open=False):
                with gr.Row():
                    whisper_model_select = gr.Radio(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="medium",
                        label="Whisper Model",
                        info="medium gives best accuracy"
                    )
                    
                    device_select = gr.Radio(
                        choices=["cpu", "cuda"],
                        value="cpu",
                        label="Device",
                        info="Use CUDA if you have NVIDIA GPU"
                    )
                
            transcribe_btn = gr.Button("🎤 Transcribe Audio", variant="primary")
            
            transcript_json = gr.Textbox(
                label="Transcript (JSON with word timestamps)",
                lines=8,
                interactive=False
            )
            
            transcribe_status = gr.Textbox(label="Status", interactive=False, lines=1)
            
            # Hidden state for audio duration (used by multi-agent)
            audio_duration_state = gr.State(value=0.0)
            
            # === STEP 4: Generate Subtitles ===
            gr.Markdown("## 📝 Step 4: Generate Subtitles")
            
            provider_gen = gr.Radio(
                choices=[
                    ("OpenAI (Paid)", "openai"),
                    ("Groq (FREE)", "groq"),
                    ("Gemini (FREE - Best)", "gemini")
                ],
                value="gemini",
                label="API Provider",
                info="✨ Gemini: 1M tokens/min (Best free tier!)"
            )
            
            with gr.Accordion("🔧 Select Format & Model", open=False):
                subtitle_format = gr.Radio(
                    choices=["SRT", "VTT", "ASS"],
                    value="SRT",
                    label="Subtitle Format"
                )
                
                with gr.Row():
                    model_gen_openai = gr.Radio(
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                        value="gpt-4o",
                        label="OpenAI Model",
                        info="Pro model recommended for best subtitle quality",
                        visible=False
                    )
                    
                    model_gen_groq = gr.Radio(
                        choices=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                        value="llama-3.3-70b-versatile",
                        label="Groq Model",
                        info="Pro model recommended for best subtitle quality",
                        visible=False
                    )
                    
                    model_gen_gemini = gr.Radio(
                        choices=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview"],
                        value="gemini-2.5-pro",
                        label="Gemini Model",
                        info="Pro model recommended for best subtitle quality",
                        visible=True
                    )
            
            ultra_mode = gr.Radio(
                choices=[
                    ("Disabled", "disabled"),
                    ("Basic Pause Handling", "basic"),
                    ("Word-by-Word", "word_by_word")
                ],
                value="disabled",
                label="Ultra Detailed Mode",
                info="Controls subtitle granularity"
            )
            
            with gr.Row(visible=False) as pause_settings:
                pause_threshold = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.5,
                    label="Pause Threshold (seconds)",
                    info="Create new subtitle after pause of this length"
                )
            
            with gr.Accordion("⚙️ Advanced Settings", open=True):
                with gr.Row():
                    max_chars = gr.Slider(
                        minimum=20,
                        maximum=60,
                        value=42,
                        step=1,
                        label="Max Characters per Line",
                        info="Standard is 42"
                    )
                    
                    max_lines = gr.Slider(
                        minimum=1,
                        maximum=3,
                        value=2,
                        step=1,
                        label="Max Lines per Subtitle",
                        info="Standard is 2"
                    )
                
                with gr.Row():
                    validation_mode = gr.Radio(
                        choices=[
                            ("⚡ Single Pass (1 API call - fast, saves quota)", "single_pass"),
                            ("🤖 Multi-Agent Validation (12-16 calls - better quality)", "multi_agent")
                        ],
                        value="single_pass",
                        label="Validation Mode",
                        info="Single pass recommended for Gemini free tier (20 req/day)"
                    )
                    
                    max_agent_iterations = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Max Agent Iterations",
                        info="⚠️ Only for multi-agent mode",
                        visible=False
                    )
            
            generate_btn = gr.Button("📝 Generate Subtitles", variant="primary", size="lg")
            
            subtitles_output = gr.Textbox(
                label="Generated Subtitles",
                lines=15,
                interactive=True
            )
            
            generate_status = gr.Textbox(label="Status", interactive=False, lines=1)
            
            validation_log = gr.Textbox(
                label="Multi-Agent Validation Log",
                lines=12,
                interactive=False,
                visible=True
            )
            
            with gr.Row():
                download_btn = gr.DownloadButton("💾 Download Subtitles", variant="secondary")
            
            # === STEP 5: Preview & Edit ===
            gr.Markdown("## 🎬 Step 5: Preview & Edit Subtitles")
            gr.Markdown("""
            ### Audio Preview with Subtitle Editor
            
            Listen to your audio and edit subtitles in real-time. Click on a subtitle to jump to that timestamp.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    preview_audio = gr.Audio(
                        label="Audio Player",
                        type="filepath",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("**Edit Subtitles Below:**")
                    
                    subtitle_editor = gr.Dataframe(
                        headers=["#", "Start Time", "End Time", "Text"],
                        datatype=["number", "str", "str", "str"],
                        row_count=20,
                        column_count=(4, "fixed"),
                        interactive=True,
                        wrap=True,
                        label="Subtitle Editor"
                    )
            
            with gr.Row():
                apply_edits_btn = gr.Button("💾 Apply Edits", variant="primary")
                refresh_preview_btn = gr.Button("🔄 Refresh from Generated", variant="secondary")
            
            edit_status = gr.Textbox(label="Status", interactive=False, lines=1)
            
            # === Event Handlers ===
            
            # Update model dropdown visibility when provider changes (avoids gr.update() positioning bug)
            def toggle_clean_models(provider):
                return {
                    model_clean_openai: gr.update(visible=(provider == "openai")),
                    model_clean_groq: gr.update(visible=(provider == "groq")),
                    model_clean_gemini: gr.update(visible=(provider == "gemini"))
                }
            
            provider_clean.change(
                fn=toggle_clean_models,
                inputs=[provider_clean],
                outputs=[model_clean_openai, model_clean_groq, model_clean_gemini]
            )
            
            def toggle_gen_models(provider):
                return {
                    model_gen_openai: gr.update(visible=(provider == "openai")),
                    model_gen_groq: gr.update(visible=(provider == "groq")),
                    model_gen_gemini: gr.update(visible=(provider == "gemini"))
                }
            
            provider_gen.change(
                fn=toggle_gen_models,
                inputs=[provider_gen],
                outputs=[model_gen_openai, model_gen_groq, model_gen_gemini]
            )
            
            # Show/hide pause threshold based on ultra mode
            def update_pause_visibility(mode):
                return gr.update(visible=(mode == "basic"))
            
            ultra_mode.change(
                fn=update_pause_visibility,
                inputs=[ultra_mode],
                outputs=[pause_settings]
            )
            
            # Clean lyrics
            def clean_with_selected_model(lyrics, is_clean, model_openai, model_groq, model_gemini, provider):
                """Wrapper to select correct model based on provider"""
                model_map = {
                    "openai": model_openai,
                    "groq": model_groq,
                    "gemini": model_gemini
                }
                selected_model = model_map.get(provider, model_gemini)
                return self.clean_lyrics(lyrics, is_clean, selected_model, provider)
            
            clean_btn.click(
                fn=clean_with_selected_model,
                inputs=[
                    lyrics_input, 
                    is_clean_checkbox, 
                    model_clean_openai, 
                    model_clean_groq, 
                    model_clean_gemini, 
                    provider_clean
                ],
                outputs=[cleaned_lyrics, clean_status]
            )
            
            # Transcribe audio
            transcribe_btn.click(
                fn=self.transcribe_audio,
                inputs=[audio_input, whisper_model_select, device_select],
                outputs=[transcript_json, transcribe_status, audio_duration_state]
            )
            
            # Show/hide validation log and iterations slider based on validation mode
            validation_mode.change(
                fn=lambda mode: (
                    gr.update(visible=mode == "multi_agent"),
                    gr.update(visible=mode == "multi_agent")
                ),
                inputs=[validation_mode],
                outputs=[validation_log, max_agent_iterations]
            )
            
            # Generate subtitles
            def generate_with_selected_model(
                transcript_json, cleaned_lyrics, 
                model_openai, model_groq, model_gemini,
                subtitle_format, ultra_mode, pause_threshold,
                max_chars, max_lines, provider,
                validation_mode, audio_duration_state, max_agent_iterations
            ):
                """Wrapper to select correct model based on provider"""
                model_map = {
                    "openai": model_openai,
                    "groq": model_groq,
                    "gemini": model_gemini
                }
                selected_model = model_map.get(provider, model_gemini)
                return self.generate_subtitles(
                    transcript_json, cleaned_lyrics, selected_model,
                    subtitle_format, ultra_mode, pause_threshold,
                    max_chars, max_lines, provider,
                    validation_mode, audio_duration_state, max_agent_iterations
                )
            
            generate_btn.click(
                fn=generate_with_selected_model,
                inputs=[
                    transcript_json,
                    cleaned_lyrics,
                    model_gen_openai,
                    model_gen_groq,
                    model_gen_gemini,
                    subtitle_format,
                    ultra_mode,
                    pause_threshold,
                    max_chars,
                    max_lines,
                    provider_gen,
                    validation_mode,
                    audio_duration_state,
                    max_agent_iterations
                ],
                outputs=[subtitles_output, generate_status, validation_log]
            )
            
            # Prepare download
            generate_btn.click(
                fn=lambda subs, fmt: self.save_subtitles(subs, fmt)[0],
                inputs=[subtitles_output, subtitle_format],
                outputs=[download_btn]
            )
            
            # Populate preview when subtitles are generated
            generate_btn.click(
                fn=lambda audio, subs, fmt: (
                    audio,  # Audio passthrough
                    self.parse_subtitles_to_dataframe(subs, fmt)  # Parse to dataframe
                ),
                inputs=[audio_input, subtitles_output, subtitle_format],
                outputs=[preview_audio, subtitle_editor]
            )
            
            # Apply manual edits back to subtitles
            apply_edits_btn.click(
                fn=lambda df_data, fmt: (
                    self.dataframe_to_subtitles(df_data, fmt),
                    "✓ Edits applied! Subtitles updated."
                ),
                inputs=[subtitle_editor, subtitle_format],
                outputs=[subtitles_output, edit_status]
            ).then(
                fn=lambda subs, fmt: self.save_subtitles(subs, fmt)[0],
                inputs=[subtitles_output, subtitle_format],
                outputs=[download_btn]
            )
            
            # Refresh dataframe from generated subtitles
            refresh_preview_btn.click(
                fn=lambda subs, fmt: (
                    self.parse_subtitles_to_dataframe(subs, fmt),
                    "✓ Refreshed from generated subtitles"
                ),
                inputs=[subtitles_output, subtitle_format],
                outputs=[subtitle_editor, edit_status]
            )
