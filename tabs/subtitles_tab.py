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
        
        Args:
            provider: "openai", "groq", or "gemini"
            task: "clean" or "gen" (generation)
        
        Returns:
            gr.Dropdown.update() dict
        """
        if provider == "groq":
            if task == "clean":
                choices = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
                value = "llama-3.1-8b-instant"
            else:  # generation
                choices = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
                value = "llama-3.3-70b-versatile"
        elif provider == "gemini":
            if task == "clean":
                choices = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-2.0-flash-exp"]
                value = "gemini-1.5-flash-latest"
            else:  # generation
                choices = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-2.0-flash-exp"]
                value = "gemini-1.5-pro-latest"
        else:  # openai
            if task == "clean":
                choices = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
                value = "gpt-4o-mini"
            else:  # generation
                choices = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
                value = "gpt-4o"
        
        return gr.Dropdown(choices=choices, value=value)
    
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
        gpt_model: str,
        provider: str = "openai",
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Clean lyrics using GPT.
        
        Args:
            lyrics: Raw lyrics text
            is_clean: Skip cleaning if True
            gpt_model: Model to use
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
                model=gpt_model,
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
        gpt_model: str,
        subtitle_format: str,
        ultra_mode: str,
        pause_threshold: float,
        max_chars_per_line: int,
        max_lines: int,
        provider: str = "openai",
        use_multiagent: bool = False,
        audio_duration: float = 0.0,
        max_iterations: int = 3,
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Generate subtitles using GPT based on Whisper transcript + cleaned lyrics.
        
        Args:
            transcript_json: JSON string with Whisper word timestamps
            cleaned_lyrics: Clean reference lyrics
            gpt_model: Model to use
            subtitle_format: SRT, VTT, or ASS
            ultra_mode: disabled, basic, or word_by_word
            pause_threshold: Pause duration for new subtitle
            max_chars_per_line: Max characters per line
            max_lines: Max lines per subtitle
            provider: "openai" or "groq"
            use_multiagent: Enable multi-agent validation system
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
            
            system_prompt = f"""You are an expert subtitle generator. You create {subtitle_format} format subtitles.

Your task:
1. Match the Whisper timestamps with the reference lyrics
2. Generate properly formatted {subtitle_format} subtitles
3. Follow subtitle best practices

{mode_instructions}

Output ONLY the {subtitle_format} formatted subtitles, nothing else."""

            user_prompt = f"""Reference Lyrics (ground truth text):
{cleaned_lyrics}

---

Whisper Transcription (with word timestamps):
{formatted_transcript}

---

Generate {subtitle_format} subtitles following all rules."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call GPT
            subtitles = helper.call_gpt(
                messages=messages,
                model=gpt_model,
                temperature=0.2,  # Low temperature for precise formatting
                max_tokens=8000
            )
            
            if not subtitles:
                return "", f"❌ {provider_label} API call failed - no response received", ""
            
            validation_log = ""
            
            # Multi-Agent Validation (if enabled)
            if use_multiagent and audio_duration > 0:
                progress(0.7, "Running multi-agent validation...")
                
                try:
                    agent_system = SubtitleAgentSystem(helper, gpt_model, max_iterations=max_iterations)
                    
                    # Pass full context to agents including UI parameters
                    agent_context = {
                        'audio_duration': audio_duration,
                        'ultra_detailed_mode': ultra_mode,
                        'max_chars_per_line': max_chars_per_line,
                        'max_lines_per_subtitle': max_lines,
                        'content_type': 'song'  # KEY: Tell agents this is a song (long gaps are normal)
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
                        progress_callback=lambda p, msg: progress(0.7 + p * 0.3, msg)
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
Model: {gpt_model}

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
            if use_multiagent and "RATE LIMIT" not in validation_log:
                status_msg += " (Multi-agent validated)"
            
            return subtitles, status_msg, validation_log
                
        except Exception as e:
            logger.error(f"Error generating subtitles: {e}")
            error_msg = str(e)
            
            # Enhanced error message with provider info and solutions
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                detailed_error = f"""❌ RATE LIMIT EXCEEDED

Provider: {provider_label}
Model: {gpt_model}

Error Details:
{error_msg}

SOLUTIONS:
1. Reduce 'Max Agent Iterations' slider (currently: {max_iterations})
2. Disable 'Multi-Agent Validation' temporarily
3. Switch to a different provider:
   - ✨ Gemini: 1M tokens/min (Best!) - Settings tab
   - Groq: 100k tokens/day - Good alternative
4. Wait for the cooldown period specified above
5. Provider limits comparison:
   - Gemini Free: 1M tokens/min (15 RPM)
   - Groq Free: 100k tokens/day
   - Each song with 3 iterations: ~30-40k tokens
6. For Groq: Upgrade at https://console.groq.com/settings/billing
7. For OpenAI: Check usage at https://platform.openai.com/usage"""
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
        
        elif ultra_mode == "word_by_word":
            return base_rules + """
Ultra Detailed Mode: Word-by-Word
- Create ONE subtitle for EACH individual word
- Each subtitle shows exactly one word with its precise timing
- Perfect for karaoke-style subtitles or language learning
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
            with gr.Accordion("📁 Step 1: Upload Files", open=True):
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
            with gr.Accordion("🧹 Step 2: Clean Lyrics", open=True):
                gr.Markdown("Remove AI formatting tags (Suno, Udio, Mureka) to get clean text.")
                
                with gr.Row():
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
                
                with gr.Row():
                    gpt_model_clean = gr.Dropdown(
                        choices=["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"],
                        value="gemini-1.5-flash-latest",
                        label="Model for Cleaning",
                        info="Fast model recommended for cleaning"
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
            with gr.Accordion("🎤 Step 3: Transcribe with Whisper", open=True):
                with gr.Row():
                    whisper_model_select = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="medium",
                        label="Whisper Model",
                        info="medium gives best accuracy"
                    )
                    
                    device_select = gr.Dropdown(
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
            with gr.Accordion("📝 Step 4: Generate Subtitles", open=True):
                with gr.Row():
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
                
                with gr.Row():
                    subtitle_format = gr.Dropdown(
                        choices=["SRT", "VTT", "ASS"],
                        value="SRT",
                        label="Subtitle Format"
                    )
                    
                    gpt_model_gen = gr.Dropdown(
                        choices=["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"],
                        value="gemini-1.5-pro-latest",
                        label="Model for Generation",
                        info="Pro model recommended for best subtitle quality"
                    )
                
                with gr.Row():
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
                        enable_multiagent = gr.Checkbox(
                            label="🤖 Enable Multi-Agent Validation",
                            value=True,
                            info="Uses 4 specialized agents to verify and correct subtitles. Fast with Groq!"
                        )
                        
                        max_agent_iterations = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Max Agent Iterations",
                            info="⚠️ Reduce to 3 for Groq free tier (100k tokens/day limit)",
                            visible=True
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
            with gr.Accordion("🎬 Step 5: Preview & Edit Subtitles", open=False):
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
                            col_count=(4, "fixed"),
                            interactive=True,
                            wrap=True,
                            label="Subtitle Editor"
                        )
                
                with gr.Row():
                    apply_edits_btn = gr.Button("💾 Apply Edits", variant="primary")
                    refresh_preview_btn = gr.Button("🔄 Refresh from Generated", variant="secondary")
                
                edit_status = gr.Textbox(label="Status", interactive=False, lines=1)
            
            # === Event Handlers ===
            
            # Update models when provider changes
            provider_clean.change(
                fn=lambda p: self.update_models_for_provider(p, "clean"),
                inputs=[provider_clean],
                outputs=[gpt_model_clean]
            )
            
            provider_gen.change(
                fn=lambda p: self.update_models_for_provider(p, "gen"),
                inputs=[provider_gen],
                outputs=[gpt_model_gen]
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
            clean_btn.click(
                fn=self.clean_lyrics,
                inputs=[lyrics_input, is_clean_checkbox, gpt_model_clean, provider_clean],
                outputs=[cleaned_lyrics, clean_status]
            )
            
            # Transcribe audio
            transcribe_btn.click(
                fn=self.transcribe_audio,
                inputs=[audio_input, whisper_model_select, device_select],
                outputs=[transcript_json, transcribe_status, audio_duration_state]
            )
            
            # Show/hide validation log and iterations slider based on multi-agent checkbox
            enable_multiagent.change(
                fn=lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)),
                inputs=[enable_multiagent],
                outputs=[validation_log, max_agent_iterations]
            )
            
            # Generate subtitles
            generate_btn.click(
                fn=self.generate_subtitles,
                inputs=[
                    transcript_json,
                    cleaned_lyrics,
                    gpt_model_gen,
                    subtitle_format,
                    ultra_mode,
                    pause_threshold,
                    max_chars,
                    max_lines,
                    provider_gen,
                    enable_multiagent,
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
