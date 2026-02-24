"""
Audio countdown generator - adds beeps before audio starts
"""
import numpy as np
from scipy.io import wavfile
import subprocess
from pathlib import Path
import tempfile


def generate_beep(frequency=800, duration=0.15, sample_rate=44100):
    """Generate a beep tone"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    beep = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume
    return (beep * 32767).astype(np.int16)


def add_countdown_to_audio(input_audio_path, output_path, countdown_seconds=5):
    """
    Add countdown beeps before audio starts
    
    Returns:
        tuple: (output_path, countdown_duration_seconds)
    """
    sample_rate = 44100
    
    # Generate countdown (1 beep per second)
    beep = generate_beep(frequency=800, duration=0.15, sample_rate=sample_rate)
    silence = np.zeros(int(sample_rate * 0.85), dtype=np.int16)  # Rest of the second
    
    # Create countdown audio
    countdown_audio = []
    for i in range(countdown_seconds):
        countdown_audio.extend(beep)
        countdown_audio.extend(silence)
    
    countdown_audio = np.array(countdown_audio, dtype=np.int16)
    
    # Save countdown to temp file
    countdown_wav = tempfile.mktemp(suffix='.wav')
    wavfile.write(countdown_wav, sample_rate, countdown_audio)
    
    # Convert input audio to WAV if needed
    input_wav = tempfile.mktemp(suffix='.wav')
    subprocess.run([
        'ffmpeg', '-y', '-i', str(input_audio_path),
        '-ar', str(sample_rate), '-ac', '1',
        input_wav
    ], capture_output=True)
    
    # Concatenate countdown + original audio
    concat_file = tempfile.mktemp(suffix='.txt')
    with open(concat_file, 'w') as f:
        f.write(f"file '{countdown_wav}'\n")
        f.write(f"file '{input_wav}'\n")
    
    # Merge audio files
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        str(output_path)
    ], capture_output=True)
    
    # Cleanup
    Path(countdown_wav).unlink(missing_ok=True)
    Path(input_wav).unlink(missing_ok=True)
    Path(concat_file).unlink(missing_ok=True)
    
    return str(output_path), countdown_seconds


def trim_video_from_countdown(video_path, output_path, skip_seconds=5):
    """
    Trim first N seconds from video (the countdown part)
    """
    subprocess.run([
        'ffmpeg', '-y',
        '-ss', str(skip_seconds),  # Skip first N seconds
        '-i', str(video_path),
        '-c', 'copy',  # Copy without re-encoding (fast)
        str(output_path)
    ], capture_output=True)
    
    return str(output_path)


def sync_audio_video_with_countdown(audio_with_countdown, webcam_video, output_dir, countdown_duration=5):
    """
    Sync audio and video by trimming countdown from both
    
    Returns:
        tuple: (synced_video_path, original_audio_path)
    """
    output_dir = Path(output_dir)
    
    # FIRST: Convert webcam video to 24fps (optimize for processing)
    video_24fps = output_dir / "webcam_24fps.mp4"
    subprocess.run([
        'ffmpeg', '-y',
        '-i', str(webcam_video),
        '-r', '24',  # Force 24fps
        '-c:v', 'libx264',  # Re-encode to h264
        '-preset', 'fast',
        '-crf', '23',
        str(video_24fps)
    ], capture_output=True, check=True)
    
    # Trim countdown from converted 24fps video
    synced_video = output_dir / "synced_video.mp4"
    trim_video_from_countdown(video_24fps, synced_video, skip_seconds=countdown_duration)
    
    # Extract original audio (without countdown)
    original_audio = output_dir / "original_audio.wav"
    subprocess.run([
        'ffmpeg', '-y',
        '-ss', str(countdown_duration),  # Skip countdown
        '-i', str(audio_with_countdown),
        '-vn',  # No video
        str(original_audio)
    ], capture_output=True, check=True)
    
    return str(synced_video), str(original_audio)
