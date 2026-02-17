# Troubleshooting Guide

## Issue: "Error" badges on UI components + PermissionError in logs

### Symptoms
- Red "Error" badges appear on Status, Raw Transcription, and other fields after clicking "Process Lecture"
- Server logs show: `PermissionError: [Errno 13] Permission denied: 'D:\Temp\gradio\...\AUDIO-...m4a'`
- The pipeline may still be running in the background despite the UI errors

### Root Cause
This is a **Windows file-locking issue**. When faster-whisper opens the audio file for transcription, it locks the file. Simultaneously, Gradio's ASGI server tries to serve the same file back to the browser for the audio player widget. Windows doesn't allow the file to be read by both at the same time, causing a `PermissionError` that cascades to show "Error" on all UI components.

### Fix Options (try in order)

#### Fix 1: Copy audio to a separate path before processing (recommended)
In `core/pipeline.py`, at the start of `process()`, copy the uploaded file so the transcriber and Gradio don't fight over the same file:

```python
import shutil

# Inside process(), right after output_dir = tempfile.mkdtemp(prefix="lecture_")
# Copy audio to output_dir so Gradio and transcriber don't lock the same file
audio_copy = os.path.join(output_dir, os.path.basename(audio_path))
shutil.copy2(audio_path, audio_copy)
audio_path = audio_copy
```

Add this right after line 110 (`output_dir = tempfile.mkdtemp(prefix="lecture_")`), before the validation step.

#### Fix 2: Set Gradio temp directory
Set the `GRADIO_TEMP_DIR` environment variable to a different location than the default. In `run_desktop.py`, add before launching:

```python
os.environ["GRADIO_TEMP_DIR"] = os.path.join(tempfile.gettempdir(), "gradio_uploads")
```

#### Fix 3: Disable Windows file locking for Gradio
This is less reliable, but you can try setting Gradio to not cache the uploaded file:

```python
audio_input = gr.Audio(
    label="Lecture audio (m4a, mp3, wav, ogg, webm, flac)",
    type="filepath",
    sources=["upload"],
    # Try adding:
    streaming=False,
)
```

### Verification
After applying a fix:
1. Restart the app: kill old process, run `python run_desktop.py`
2. Upload audio and click "Process Lecture"
3. Check that:
   - No `PermissionError` in the terminal logs
   - Status shows step progress (e.g., "Step 1/5 — Transcribing")
   - No red "Error" badges on components

---

## Issue: Pipeline takes very long on CPU

### Explanation
Transcribing on CPU (desktop mode) is ~50x slower than GPU. A 58-minute audio takes ~30-40 minutes on CPU with faster-whisper.

### Workaround
- Use Google Colab with GPU for long lectures (the notebook is at `colab/lecture_processor.ipynb`)
- For desktop testing, use a short audio clip (< 5 minutes)

---

## Issue: Per-language summaries require Gemini API key

### Explanation
Step 4 (Summarize) uses Gemini to generate summaries in each translated language. Without a Gemini API key, summaries and slides will be skipped.

### Fix
Set the `GEMINI_API_KEY` environment variable before running:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your-key-here"
python run_desktop.py
```

**Windows (cmd):**
```cmd
set GEMINI_API_KEY=your-key-here
python run_desktop.py
```

**Mac/Linux:**
```bash
export GEMINI_API_KEY="your-key-here"
python run_desktop.py
```

Get a free key at https://ai.google.dev/

---

## Issue: `google.genai` import error

### Symptoms
```
ModuleNotFoundError: No module named 'google.genai'
```

### Fix
The project migrated from `google-generativeai` to `google-genai`. If you have the old package installed, it causes namespace conflicts:

```bash
pip uninstall -y google-generativeai google-ai-generativelanguage
pip install --force-reinstall google-genai
```

---

## Quick Setup Checklist (new machine)

1. Clone the repo
2. Create venv: `python -m venv venv`
3. Install CPU-only PyTorch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
4. Install deps: `pip install -r requirements.txt`
5. Set env vars:
   ```bash
   export GEMINI_API_KEY="your-key"
   export HF_HUB_DISABLE_SYMLINKS_WARNING=1
   export HF_HUB_DISABLE_PROGRESS_BARS=1
   ```
6. Run: `python run_desktop.py`
7. Open http://127.0.0.1:7860
