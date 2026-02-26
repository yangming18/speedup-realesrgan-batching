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
            timing_issues = self._agent_timing_validator(current_subtitles, agent_context, validation_log)
            validation_log.append(f"📊 Result: {timing_issues}\n")
            
            # Agent 2: Lyrics Matcher
            validation_log.append(f"🤖 Agent 2: Lyrics Matcher")
            validation_log.append(f"{'─'*50}")
            lyrics_issues = self._agent_lyrics_matcher(current_subtitles, lyrics, whisper_data, validation_log)
            validation_log.append(f"📊 Result: {lyrics_issues}\n")
            
            # Agent 3: Format Validator
            validation_log.append(f"🤖 Agent 3: Format Validator")
            validation_log.append(f"{'─'*50}")
            format_issues = self._agent_format_validator(current_subtitles, subtitle_format, agent_context, validation_log)
            validation_log.append(f"📊 Result: {format_issues}\n")
            
            # Coordinator decides if corrections are needed
            validation_log.append(f"🤖 Coordinator: Decision Agent")
            validation_log.append(f"{'─'*50}")
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
            response = self.helper.call_gpt(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=150
            )
            result = response.strip() if response else "Agent error"
            log.append(f"💡 Response: {result}")
            return result
        except Exception as e:
            logger.error(f"Timing validator error: {e}")
            error_msg = f"Agent error: {str(e)}"
            log.append(f"❌ Error: {error_msg}")
            return error_msg
    
    def _agent_lyrics_matcher(self, subtitles: str, lyrics: str, whisper_data: List[Dict], log: List[str]) -> str:
        """Agent 2: Match subtitle text with lyrics"""
        # Extract text from subtitles (parse SRT format)
        subtitle_words = self._extract_subtitle_text(subtitles)
        lyrics_words = lyrics.lower().split()
        whisper_words = [w['word'].lower().strip() for w in whisper_data]
        
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
            response = self.helper.call_gpt(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=150
            )
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
        
        # Mode-specific format notes
        if ultra_mode == 'word_by_word':
            mode_note = "\n⚠️  Mode: Word-by-word (1 word per subtitle is CORRECT)"
        else:
            mode_note = ""
        
        prompt = f"""You are a subtitle format expert. Validate this {format_type} format.

USER SETTINGS:
- Max {max_chars} characters per line
- Max {max_lines} lines per subtitle{mode_note}

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
            response = self.helper.call_gpt(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=150
            )
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
        
        # Content-specific instructions
        if content_type == 'song':
            content_note = "\n⚠️  This is a SONG: Preserve long gaps between lyrics (instrumental sections). DO NOT merge subtitles across silence."
        else:
            content_note = ""
        
        # Mode-specific instructions
        if ultra_mode == 'word_by_word':
            mode_note = "\n⚠️  Mode: Word-by-word (each subtitle should contain ONE word with precise timing)"
        else:
            mode_note = "\n📝 Mode: Standard (group words naturally into readable phrases)"
        
        # Build correction prompt
        prompt = f"""You are an expert subtitle corrector. Fix the subtitles based on validation feedback.

CONTENT TYPE: {content_type.upper()}{content_note}{mode_note}

USER SETTINGS:
- Max {max_chars} characters per line
- Max {max_lines} lines per subtitle

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

CORRECTION RULES:
1. Keep Whisper timestamps unless clearly wrong
2. Fix text to match lyrics exactly
3. Maintain subtitle format
4. Fix any timing overlaps (but respect intentional gaps for {content_type})
5. Ensure max {max_chars} chars/line, max {max_lines} lines
6. Respect {ultra_mode} mode requirements

Return the COMPLETE corrected subtitles in proper format. Output ONLY the subtitles, no explanations."""
        
        log.append(f"💬 Prompt: Apply corrections for {content_type} ({ultra_mode} mode, max {max_chars} chars/line, {max_lines} lines)")
        
        try:
            response = self.helper.call_gpt(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=8000
            )
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
