#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Agent Subtitle Validation System
Uses Groq/OpenAI for fast iterative subtitle validation with multiple specialized agents
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
from utils.openai_helper import OpenAIHelper

logger = logging.getLogger(__name__)


class SubtitleAgentSystem:
    """
    Multi-agent system for subtitle validation and correction.
    
    Agents:
    1. TimingValidator - Verifica timing, sovrapposizioni, pause
    2. LyricsMatcher - Confronta parole con lyrics reference
    3. TextCorrector - Corregge errori usando lyrics come ground truth  
    4. FormatValidator - Verifica formato SRT/VTT/ASS
    5. Coordinator - Coordina e produce output finale
    """
    
    def __init__(self, helper: OpenAIHelper, model: str = "llama-3.3-70b-versatile", max_iterations: int = 3):
        self.helper = helper
        self.model = model
        self.max_iterations = max_iterations
        self.progress_callback = None  # Will be set during validation
        self.conversation_history = []  # Track conversation for summarization
    
    def _summarize_history(self, full_prompt: str, preserve_data: bool = True) -> str:
        """
        Riassume prompt lunghi preservando dati essenziali.
        
        Args:
            full_prompt: Prompt completo
            preserve_data: Se True, preserva subtitle text, lyrics, whisper data
        
        Returns:
            Prompt riassunto (o originale se già corto)
        """
        # Se il prompt è già corto, non riassumere
        if len(full_prompt) < 2000:
            return full_prompt
        
        # Se c'è poco storico, non serve riassumere
        if len(self.conversation_history) < 3:
            return full_prompt
        
        # Se preserve_data=True, identifica e preserva sezioni di dati
        if preserve_data:
            # Pattern per identificare dati da preservare (non riassumere)
            data_markers = [
                "Subtitles (first", "Subtitle Text", "Lyrics Reference", 
                "Whisper", "ORIGINAL SUBTITLES:", "GROUND TRUTH LYRICS:"
            ]
            
            # Controlla se il prompt contiene dati da preservare
            has_critical_data = any(marker in full_prompt for marker in data_markers)
            
            if has_critical_data:
                # NON riassumere prompt con dati critici (lyrics, subtitles, whisper)
                return full_prompt
        
        # Riassumi le istruzioni/regole ma mantieni la struttura
        try:
            summary_prompt = f"""Summarize these instructions keeping only essential information:

{full_prompt[:1500]}

Summary (max 300 chars, preserve key numbers/requirements):"""
            
            summary = self.helper.call_gpt(
                messages=[{"role": "user", "content": summary_prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=300,
                max_retries=1  # Quick summary, don't retry much
            )
            
            if summary and len(summary) < len(full_prompt) * 0.7:
                return summary.strip()
        except Exception as e:
            logger.warning(f"Summary failed, using full prompt: {e}")
        
        # Fallback: restituisci originale
        return full_prompt
    
    def _call_agent_gpt(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 150,
        preserve_data: bool = True
    ) -> Optional[str]:
        """
        Call GPT with automatic history tracking and summarization.
        
        Args:
            prompt: Full prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            preserve_data: If True, don't summarize prompts with subtitle/lyrics data
        
        Returns:
            Model response
        """
        # Track this prompt in history
        self.conversation_history.append(("user", prompt))
        
        # Summarize if needed (but preserve critical data)
        summarized_prompt = self._summarize_history(prompt, preserve_data)
        
        # Call GPT
        response = self.helper.call_gpt(
            messages=[{"role": "user", "content": summarized_prompt}],
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            progress_callback=self.progress_callback
        )
        
        # Track response
        if response:
            self.conversation_history.append(("assistant", response))
        
        return response
        
    def validate_and_correct_subtitles(
        self,
        subtitles: str,
        lyrics: str,
        whisper_data: List[Dict],
        audio_duration: float,
        subtitle_format: str = "SRT",
        context: Dict = None,
        progress_callback=None
    ) -> Tuple[str, List[str]]:
        """
        Run multi-agent validation with up to 10 iterations.
        
        Args:
            subtitles: Generated subtitles
            lyrics: Ground truth lyrics
            whisper_data: Word timestamps from Whisper
            audio_duration: Total audio duration in seconds
            subtitle_format: SRT, VTT, or ASS
            context: Additional context (ultra_mode, max_chars, content_type, etc.)
            progress_callback: Function to report progress
        
        Returns:
            Tuple of (corrected_subtitles, validation_log)
        """
        validation_log = []
        current_subtitles = subtitles
        
        # Store progress callback for use in agent methods  
        self.progress_callback = progress_callback
        
        # Reset conversation history for this validation run
        self.conversation_history = []
        
        # Stats for context
        word_count = len(lyrics.split())
        whisper_words = len(whisper_data)
        
        # Merge provided context with computed stats
        agent_context = {
            "audio_duration": audio_duration,
            "lyrics_word_count": word_count,
            "whisper_word_count": whisper_words,
            "format": subtitle_format
        }
        if context:
            agent_context.update(context)  # Add UI parameters and content type
        
        # Log context including content type and UI settings
        content_type = agent_context.get('content_type', 'audio')
        ultra_mode = agent_context.get('ultra_detailed_mode', 'disabled')
        max_chars = agent_context.get('max_chars_per_line', 42)
        max_lines = agent_context.get('max_lines_per_subtitle', 2)
        
        validation_log.append(f"📊 Context: Audio={audio_duration:.1f}s, Type={content_type}, Mode={ultra_mode}")
        validation_log.append(f"📐 Format: {subtitle_format} | Max {max_chars} chars/line, {max_lines} lines/subtitle")
        validation_log.append(f"📝 Words: Lyrics={word_count}, Whisper={whisper_words}")
        
        for iteration in range(1, self.max_iterations + 1):
            if progress_callback:
                progress_callback(iteration / self.max_iterations, f"Agent iteration {iteration}/{self.max_iterations}")
            
            validation_log.append(f"\n{'='*60}")
            validation_log.append(f"🔄 ITERATION {iteration}/{self.max_iterations}")
            validation_log.append(f"{'='*60}")
            
            # Agent 1: Timing Validator
            validation_log.append(f"\n🤖 Agent 1: Timing Validator")
            validation_log.append(f"{'─'*50}")
            if progress_callback:
                progress_callback((iteration - 0.75) / self.max_iterations, "🔍 Agent 1: Checking timing...")
            timing_issues = self._agent_timing_validator(current_subtitles, agent_context, validation_log)
            validation_log.append(f"📊 Result: {timing_issues}\n")
            
            # Agent 2: Lyrics Matcher
            validation_log.append(f"🤖 Agent 2: Lyrics Matcher")
            validation_log.append(f"{'─'*50}")
            if progress_callback:
                progress_callback((iteration - 0.5) / self.max_iterations, "🎵 Agent 2: Matching lyrics...")
            lyrics_issues = self._agent_lyrics_matcher(current_subtitles, lyrics, whisper_data, agent_context, validation_log)
            validation_log.append(f"📊 Result: {lyrics_issues}\n")
            
            # Agent 3: Format Validator
            validation_log.append(f"🤖 Agent 3: Format Validator")
            validation_log.append(f"{'─'*50}")
            if progress_callback:
                progress_callback((iteration - 0.25) / self.max_iterations, "📝 Agent 3: Validating format...")
            format_issues = self._agent_format_validator(current_subtitles, subtitle_format, agent_context, validation_log)
            validation_log.append(f"📊 Result: {format_issues}\n")
            
            # Coordinator decides if corrections are needed
            validation_log.append(f"🤖 Coordinator: Decision Agent")
            validation_log.append(f"{'─'*50}")
            if progress_callback:
                progress_callback(iteration / self.max_iterations, "🧠 Coordinator: Analyzing results...")
            needs_correction, coordinator_feedback = self._agent_coordinator(
                timing_issues,
                lyrics_issues,
                format_issues,
                iteration
            )
            
            validation_log.append(f"🎯 Decision: {coordinator_feedback}\n")
            
            if not needs_correction:
                validation_log.append(f"\n✅ Validation passed at iteration {iteration}!")
                break
            
            # Agent 4: Text Corrector (only if needed)
            if iteration < self.max_iterations:
                validation_log.append(f"🤖 Agent 4: Text Corrector")
                validation_log.append(f"{'─'*50}")
                if progress_callback:
                    progress_callback(iteration / self.max_iterations, "⚙️ Agent 4: Applying corrections...")
                
                corrected = self._agent_text_corrector(
                    current_subtitles,
                    lyrics,
                    whisper_data,
                    timing_issues,
                    lyrics_issues,
                    format_issues,
                    agent_context,
                    validation_log
                )
                
                if corrected and corrected != current_subtitles:
                    current_subtitles = corrected
                    validation_log.append(f"✅ Corrections applied\n")
                else:
                    validation_log.append(f"⚠️  No changes made - stopping\n")
                    break  # Stop if no changes
        
        return current_subtitles, validation_log
    
    def _agent_timing_validator(self, subtitles: str, context: Dict, log: List[str]) -> str:
        """Agent 1: Validate timing logic"""
        
        # Extract context info
        content_type = context.get('content_type', 'audio')
        audio_duration = context['audio_duration']
        ultra_mode = context.get('ultra_detailed_mode', 'disabled')
        
        # Content-specific guidelines
        if content_type == 'song':
            gap_guidance = """
⚠️  IMPORTANT: This is a SONG - long gaps (10-60s) are NORMAL for instrumental breaks!
- DO NOT flag gaps as suspicious unless they exceed the audio duration
- DO NOT suggest merging subtitles across instrumental sections
- Silence between lyrics is intentional and must be preserved"""
        else:
            gap_guidance = "- Flag gaps >5s between subtitles as suspicious"
        
        prompt = f"""You are a subtitle timing expert. Analyze these subtitles for timing issues.

CONTENT TYPE: {content_type.upper()}
Audio Duration: {audio_duration:.1f}s
Mode: {ultra_mode}

Subtitles (first 2000 chars):
{subtitles[:2000]}

Check for:
1. Overlapping timestamps
2. Timestamps exceeding audio duration ({audio_duration:.1f}s)
3. Negative durations
{gap_guidance}
4. Too-short durations (<0.2s)

Return ONLY a brief summary (max 2 lines):
- "OK - No timing issues" if all good
- Or list specific issues found (be precise with timecodes)"""
        
        log.append(f"💬 Prompt: Analyze timing for {content_type} ({audio_duration:.1f}s, {ultra_mode} mode)")
        
        try:
            response = self._call_agent_gpt(prompt, temperature=0.1, max_tokens=150, preserve_data=False)
            result = response.strip() if response else "Agent error"
            log.append(f"💡 Response: {result}")
            return result
        except Exception as e:
            logger.error(f"Timing validator error: {e}")
            error_msg = f"Agent error: {str(e)}"
            log.append(f"❌ Error: {error_msg}")
            return error_msg
    
    def _agent_lyrics_matcher(self, subtitles: str, lyrics: str, whisper_data: List[Dict], context: Dict, log: List[str]) -> str:
        """Agent 2: Match subtitle text with lyrics"""
        # Extract text from subtitles (parse SRT format)
        subtitle_words = self._extract_subtitle_text(subtitles)
        lyrics_words = lyrics.lower().split()
        whisper_words = [w['word'].lower().strip() for w in whisper_data]
        
        # Check if we're in word-by-word mode from context (explicit flag)
        is_word_by_word = context.get('subtitle_mode') == 'word_by_word' or context.get('ultra_detailed_mode') == 'word_by_word'
        
        if is_word_by_word:
            # Word-by-word mode: EVERY word must be present
            missing = len(whisper_words) - len(subtitle_words)
            
            if len(subtitle_words) == len(whisper_words):
                status = f"OK - All {len(whisper_words)} words present ✓"
            else:
                status = f"CRITICAL: {missing} words missing! Expected {len(whisper_words)}, got {len(subtitle_words)}"
            
            prompt = f"""🎯 WORD-BY-WORD MODE: Critical completeness check!

Expected: {len(whisper_words)} subtitles (one per word)
Generated: {len(subtitle_words)} subtitles
Status: {status}

⚠️  In word-by-word mode, subtitle count MUST equal whisper word count.

Return ONLY:
- "OK - All {len(whisper_words)} words present" if counts match
- "CRITICAL: Missing {missing} words - regenerate all {len(whisper_words)} words" if not matching"""
        else:
            # Standard mode: compare with lyrics
            prompt = f"""You are a lyrics matching expert. Compare subtitle text with ground truth lyrics.

Subtitle Text (first 100 words): {' '.join(subtitle_words[:100])}
Lyrics Reference (first 100 words): {' '.join(lyrics_words[:100])}

Total words: Subtitles={len(subtitle_words)}, Lyrics={len(lyrics_words)}, Whisper={len(whisper_words)}

Identify:
1. Missing words from lyrics
2. Extra words not in lyrics
3. Mismatched words (spelling/recognition errors)
4. Word order issues

Return ONLY a brief summary (max 2 lines):
- "OK - Text matches lyrics" if >95% match
- Or list TOP 3 issues found"""
        
        log.append(f"💬 Prompt: Compare subtitle text ({len(subtitle_words)} words) with lyrics ({len(lyrics_words)} words)")
        
        try:
            response = self._call_agent_gpt(prompt, temperature=0.1, max_tokens=150, preserve_data=True)
            result = response.strip() if response else "Agent error"
            log.append(f"💡 Response: {result}")
            return result
        except Exception as e:
            logger.error(f"Lyrics matcher error: {e}")
            error_msg = f"Agent error: {str(e)}"
            log.append(f"❌ Error: {error_msg}")
            return error_msg
    
    def _agent_format_validator(self, subtitles: str, format_type: str, context: Dict, log: List[str]) -> str:
        """Agent 3: Validate subtitle format"""
        
        # Get UI parameters from context
        max_chars = context.get('max_chars_per_line', 42)
        max_lines = context.get('max_lines_per_subtitle', 2)
        ultra_mode = context.get('ultra_detailed_mode', 'disabled')
        
        # Word-by-word mode: different validation logic
        if ultra_mode == 'word_by_word':
            prompt = f"""🎯 WORD-BY-WORD MODE: Format validation

Subtitles (first 500 chars):
{subtitles[:500]}

Check {format_type} compliance for WORD-BY-WORD mode:
1. Correct structure (index, timing, text, blank line)
2. Valid timestamp format
3. ⚠️  CRITICAL: Each subtitle should contain EXACTLY 1 word (not a phrase)
4. Proper blank line separation

ℹ️  IGNORE character/line limits (they don't apply in word-by-word mode)

Return ONLY:
- "OK - Format valid" if correct for word-by-word
- Or list specific format issues (e.g., "Multi-word subtitles detected")"""
            log.append(f"💬 Prompt: Validate {format_type} format (word-by-word mode - ignore char/line limits)")
        else:
            # Standard modes: validate char/line limits
            prompt = f"""You are a subtitle format expert. Validate this {format_type} format.

USER SETTINGS:
- Max {max_chars} characters per line
- Max {max_lines} lines per subtitle

Subtitles (first 500 chars):
{subtitles[:500]}

Check {format_type} compliance:
- Correct structure (index, timing, text, blank line)
- Valid timestamp format
- Respects max {max_lines} lines per subtitle
- Respects max {max_chars} chars per line
- Proper blank line separation

Return ONLY a brief summary (max 2 lines):
- "OK - Format valid" if correct
- Or list specific format issues"""
            log.append(f"💬 Prompt: Validate {format_type} format (max {max_chars} chars/line, {max_lines} lines, {ultra_mode} mode)")
        
        try:
            response = self._call_agent_gpt(prompt, temperature=0.1, max_tokens=150, preserve_data=True)
            result = response.strip() if response else "Agent error"
            log.append(f"💡 Response: {result}")
            return result
        except Exception as e:
            logger.error(f"Format validator error: {e}")
            error_msg = f"Agent error: {str(e)}"
            log.append(f"❌ Error: {error_msg}")
            return error_msg
    
    def _agent_coordinator(
        self,
        timing: str,
        lyrics: str,
        format_check: str,
        iteration: int
    ) -> Tuple[bool, str]:
        """Coordinator: Decide if corrections are needed"""
        # Simple heuristic
        all_ok = (
            "ok" in timing.lower() and
            "ok" in lyrics.lower() and
            "ok" in format_check.lower()
        )
        
        if all_ok:
            return False, "All agents passed validation ✅"
        
        if iteration >= self.max_iterations:
            return False, f"Max iterations reached ({self.max_iterations}), stopping"
        
        issues = []
        if "ok" not in timing.lower():
            issues.append("timing")
        if "ok" not in lyrics.lower():
            issues.append("lyrics")
        if "ok" not in format_check.lower():
            issues.append("format")
        
        return True, f"Issues detected in: {', '.join(issues)}. Running corrector..."
    
    def _agent_text_corrector(
        self,
        subtitles: str,
        lyrics: str,
        whisper_data: List[Dict],
        timing_issues: str,
        lyrics_issues: str,
        format_issues: str,
        context: Dict,
        log: List[str]
    ) -> str:
        """Agent 4: Apply corrections based on validation feedback"""
        
        # Get context info
        content_type = context.get('content_type', 'audio')
        ultra_mode = context.get('ultra_detailed_mode', 'disabled')
        max_chars = context.get('max_chars_per_line', 42)
        max_lines = context.get('max_lines_per_subtitle', 2)
        is_word_by_word = ultra_mode == 'word_by_word'
        
        # Content-specific instructions
        if content_type == 'song':
            content_note = "\n⚠️  This is a SONG: Preserve long gaps between lyrics (instrumental sections). DO NOT merge subtitles across silence."
        else:
            content_note = ""
        
        # Mode-specific instructions
        if is_word_by_word:
            whisper_word_count = len(whisper_data)
            mode_note = f"""
⚠️  WORD-BY-WORD MODE (KARAOKE):

🚨 CRITICAL REQUIREMENT: Generate EXACTLY {whisper_word_count} subtitles (one per word)

RULES:
1. ONE subtitle = ONE word (no grouping)
2. Use EXACT Whisper timestamps for each word
3. DO NOT skip any word from the Whisper array
4. If text is wrong, fix it using lyrics BUT keep the word
5. Output ALL {whisper_word_count} words as separate subtitles

If feedback says "Missing X words":
→ YOU MUST regenerate ALL {whisper_word_count} subtitles
→ Use the Whisper word array below as your source
→ Every word in array = one subtitle in output"""
        else:
            mode_note = "\n📝 Mode: Standard (group words naturally into readable phrases)"
        
        # User settings section (only for standard modes)
        if is_word_by_word:
            settings_note = ""
            correction_rules = f"""CORRECTION RULES:
1. Use Whisper timestamps exactly as given
2. Fix text to match lyrics (check spelling/transcription errors)
3. Generate ONE subtitle per word
4. Output EXACTLY {len(whisper_data)} subtitles
5. DO NOT skip, merge, or group words"""
        else:
            settings_note = f"""
USER SETTINGS:
- Max {max_chars} characters per line
- Max {max_lines} lines per subtitle"""
            correction_rules = f"""CORRECTION RULES:
1. Keep Whisper timestamps unless clearly wrong
2. Fix text to match lyrics exactly
3. Maintain subtitle format
4. Fix any timing overlaps (but respect intentional gaps for {content_type})
5. Ensure max {max_chars} chars/line, max {max_lines} lines
6. Respect {ultra_mode} mode requirements"""
        
        # Build correction prompt
        prompt = f"""You are an expert subtitle corrector. Fix the subtitles based on validation feedback.

CONTENT TYPE: {content_type.upper()}{content_note}{mode_note}{settings_note}

ORIGINAL SUBTITLES:
{subtitles[:3000]}

GROUND TRUTH LYRICS:
{lyrics[:1000]}

VALIDATION FEEDBACK:
- Timing: {timing_issues}
- Lyrics Match: {lyrics_issues}
- Format: {format_issues}

WHISPER DATA (first 50 words):
{json.dumps(whisper_data[:50], indent=2)}

{correction_rules}

Return the COMPLETE corrected subtitles in proper format. Output ONLY the subtitles, no explanations."""
        
        log.append(f"💬 Prompt: Apply corrections for {content_type} ({ultra_mode} mode, max {max_chars} chars/line, {max_lines} lines)")
        
        try:
            response = self._call_agent_gpt(prompt, temperature=0.2, max_tokens=8000, preserve_data=True)
            result = response.strip() if response else subtitles
            changes = "Applied" if result != subtitles else "No changes"
            log.append(f"💡 Response: {changes} - {len(result)} chars output")
            return result
        except Exception as e:
            logger.error(f"Text corrector error: {e}")
            error_msg = f"Agent error: {str(e)}"
            log.append(f"❌ Error: {error_msg}")
            return subtitles  # Return original if error
    
    def _extract_subtitle_text(self, subtitles: str) -> List[str]:
        """Extract just the text words from SRT/VTT subtitles"""
        words = []
        for line in subtitles.split('\n'):
            line = line.strip()
            # Skip index lines, timestamp lines, empty lines
            if line and not line.isdigit() and '-->' not in line and not line.startswith('WEBVTT'):
                words.extend(line.split())
        return words
