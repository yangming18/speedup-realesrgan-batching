#!/usr/bin/env python3
"""
Minimal webcam test - isolates webcam component from full interface
Run this to check if webcam permissions work at all
"""

import gradio as gr

def test_webcam(video):
    if video is None:
        return "No video recorded"
    return f"✅ Video recorded successfully! Path: {video}"

# Create minimal interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Webcam Test")
    gr.Markdown("""
    **Testing webcam access:**
    1. Click "Access webcam" button below
    2. Browser will ask permission → Click **ALLOW**
    3. Record a short video (5 seconds)
    4. Click STOP
    5. Click "Test" button
    
    **If you see an error:**
    - **macOS**: System Preferences → Security & Privacy → Camera → Allow Terminal/Python
    - **Browser**: Click 🔒 icon in address bar → Camera → Allow
    - **Safari**: Settings → Websites → Camera → Allow for this site
    """)
    
    video_input = gr.Video(
        sources=["webcam"],
        label="📹 Webcam (record a test video)",
        include_audio=True
    )
    
    output = gr.Textbox(label="Result")
    
    btn = gr.Button("Test Recorded Video")
    btn.click(fn=test_webcam, inputs=[video_input], outputs=[output])

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎥 WEBCAM TEST - Minimal Interface")
    print("="*60)
    print("\n📍 Open: http://127.0.0.1:7860")
    print("\n⚠️  If webcam button fails:")
    print("   1. Check macOS System Preferences → Security & Privacy → Camera")
    print("   2. Check browser permissions (click 🔒 in address bar)")
    print("   3. Try a different browser (Chrome, Firefox, Safari)")
    print("\n" + "="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True  # Enable debug mode to see more errors
    )
