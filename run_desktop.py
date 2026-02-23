"""Desktop entry point for Catalan Lecture Processor.

Run with:  python run_desktop.py
Opens a browser to http://127.0.0.1:7860
"""

import sys
import os
import logging

# Add project root to path so 'core' and 'ui' can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding — prevents UnicodeEncodeError with progress bars
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# Suppress HuggingFace tqdm download bars (they interfere with Gradio progress)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import gradio as gr
from ui.app import create_app

if __name__ == "__main__":
    app = create_app(mode="desktop")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 960px !important; margin: auto; }",
    )
