# Local Hosting Guide — Catalan Lecture Processor

This guide covers how to run the lecture processor on Michael's Windows machine
and share it with students via a public URL.

---

## What was set up (2026-03-27)

| Component | Detail |
|-----------|--------|
| Conda env | `catalan-lecture` at `F:\conda_envs\catalan-lecture` |
| Python | 3.12 |
| PyTorch | 2.11 with CUDA 12.8 |
| GPU | RTX 5090 (32 GB) + RTX 4090 (24 GB) — app uses first available |
| ffmpeg | 8.0.1 (installed inside conda env) |
| All deps | faster-whisper, transformers, gradio, NLLB, etc. |

---

## Step 1 — Launch the app

Open **Git Bash**, **Anaconda Prompt**, or any terminal and run:

```bash
cd "G:/My Drive/PythonCode/TranslationProject"
conda run -n catalan-lecture python run_desktop.py
```

Or from Anaconda Prompt / CMD:

```cmd
cd "G:\My Drive\PythonCode\TranslationProject"
conda activate catalan-lecture
python run_desktop.py
```

### First run only

The first launch downloads ML models (~3-5 GB total). This is a one-time cost:

| Model | Size | Purpose |
|-------|------|---------|
| faster-whisper-large-v3-ca | ~1.5 GB | Catalan speech-to-text |
| NLLB-200 distilled 600M | ~1.2 GB | Translation (ca → es/en/pt/it) |

These are cached in `~/.cache/huggingface/` so subsequent starts are fast.

### What you should see

```
Running on local URL:   http://127.0.0.1:7860
Running on public URL:  https://abc123def.gradio.live    <-- SEND THIS TO STUDENT

This share link expires in 72 hours.
```

The public URL is what you send to the student.

---

## Step 2 — Send the link to the student

Copy the `https://xxxxx.gradio.live` URL and send it (email, Teams, etc.).

The student:
1. Opens the link in any browser
2. Uploads their audio file (.m4a, .mp3, .wav, .ogg, .webm, .flac)
3. Optionally enters a Gemini API key for AI summarization
4. Clicks Process and waits (~5-10 min for a 1-hour lecture)
5. Downloads the results (transcript, translations, summary, slides)

**No setup needed on the student's end — just a browser.**

---

## Step 3 — Stop the app

Press `Ctrl+C` in the terminal, or just close the terminal window.
The public link stops working immediately.

---

## Troubleshooting

### "conda not found" in Git Bash

Conda should be in PATH (added to `~/.bashrc`). If not, run:
```bash
export PATH="/c/Users/mbruy/anaconda3/condabin:/c/Users/mbruy/anaconda3:/c/Users/mbruy/anaconda3/Scripts:$PATH"
```

### "ffmpeg not found" error

ffmpeg is installed inside the conda env. If the app still can't find it,
install system-wide:
```
winget install Gyan.FFmpeg
```
Then restart the terminal.

### GPU not detected / running on CPU

Verify CUDA works:
```bash
conda run -n catalan-lecture python -c "import torch; print(torch.cuda.is_available())"
```
Should print `True`. If `False`:
- Make sure NVIDIA drivers are up to date
- Restart the machine after driver updates

### "No module named X" error

Reinstall deps:
```bash
conda run -n catalan-lecture pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -n catalan-lecture pip install -r requirements.txt
```

### Share link doesn't work / student can't connect

- Gradio share links expire after **72 hours** — restart the app for a new one
- Some corporate/university firewalls block `*.gradio.live` — student should try from home WiFi or mobile data
- If share links are consistently blocked, consider using `ngrok` or Tailscale instead

### App crashes during processing

Check the terminal for error messages. Common causes:
- Audio file too large (try files under 500 MB)
- Out of GPU memory (unlikely with 24-32 GB, but close other GPU apps)

---

## Optional: Watchdog (get notified if app goes down)

Open a **second terminal** and run:

```bash
cd "G:/My Drive/PythonCode/TranslationProject"
conda run -n catalan-lecture python watchdog.py
```

This checks the server every 60 seconds. If it goes down:
- Prints a loud warning in the terminal
- Pops up a **Windows toast notification** on your desktop

You'll see output like:
```
  [OK]  14:30:00 — server responding
  [OK]  14:31:00 — server responding
  [!!]  14:32:00 — no response (fail 1/2)
  [!!]  14:33:00 — no response (fail 2/2)

  *** ALERT: App is DOWN! Restart with: conda run -n catalan-lecture python run_desktop.py ***
```

It waits for 2 consecutive failures before alerting (avoids false alarms from brief hiccups).

---

## Optional: LLM summarization

The app can generate AI summaries and slides. It checks for backends in order:

1. **Ollama** (local) — if Ollama is running at localhost:11434
2. **Gemini API** (cloud) — if `GOOGLE_GENAI_API_KEY` env var is set
3. **HuggingFace** (cloud) — if `HF_TOKEN` env var is set
4. **Skip** — summarization is skipped if none available

To use Gemini (easiest):
1. Get a free API key at https://ai.google.dev/
2. Set it before launching:
   ```bash
   export GOOGLE_GENAI_API_KEY="your-key-here"
   conda run -n catalan-lecture python run_desktop.py
   ```

To install Ollama (fully offline):
1. Download from https://ollama.ai
2. Run `ollama pull llama3.1:8b`
3. Start the app — it auto-detects Ollama

---

## Quick reference

```bash
# Launch (sends public link to terminal)
cd "G:/My Drive/PythonCode/TranslationProject"
conda run -n catalan-lecture python run_desktop.py

# Verify setup
conda run -n catalan-lecture python -c "import torch; print(torch.cuda.is_available())"
conda run -n catalan-lecture ffmpeg -version

# Update deps if code changes
conda run -n catalan-lecture pip install -r requirements.txt
```
