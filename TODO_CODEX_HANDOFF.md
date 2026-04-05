# Codex Handoff — Two Tasks Remaining

## Task 1: Replace NLLB translation with Gemini API

**Problem:** NLLB-200 600M can't handle scientific vocabulary. "Plasmid" → "plastic tenses", "ferritin" → "ferret". Need contextual LLM translation.

**What to do:**
1. In `core/translator.py`, add a `GeminiTranslator` class (or modify existing) that uses `google-genai` SDK (already in requirements.txt)
2. The Gemini API key is passed via env var `GOOGLE_GENAI_API_KEY` or entered in the UI
3. Prompt should be: "Translate the following Catalan university lecture transcript to {language}. Preserve scientific terminology accurately. Return only the translation."
4. Chunk the text (~3000 words per chunk) to stay within Gemini token limits
5. Fall back to NLLB if no API key is set
6. Update `core/pipeline.py` to use Gemini translator when available
7. The summarizer (`core/summarizer.py`) already has Gemini integration — reuse its pattern for API calls

**Key files:**
- `core/translator.py` — current NLLB translator (keep as fallback)
- `core/summarizer.py` — has working Gemini API pattern to copy from
- `core/config.py` — model names, prompts
- `core/pipeline.py` — orchestrator, step 3 is translation
- `ui/app.py` — may want optional API key input field

## Task 2: Firebase signup/login (replace current JSON auth)

**Problem:** Current auth is JSON file (`auth/users.json`) with passwords managed by admin CLI. Need self-service signup.

**What to do:**
1. Integrate Firebase Authentication (email/password)
2. Replace `core/auth.py` verify_user() to check Firebase instead of JSON
3. In `run_desktop.py`, the `auth=verify_user` pattern stays the same — just change the backend
4. Add signup UI — Gradio's built-in `auth` only shows login. Options:
   - Custom Gradio tab with signup form before main app
   - Or use Firebase JS SDK in a custom HTML component
5. Keep `manage_users.py` as admin tool (can use Firebase Admin SDK)
6. User's display name from Firebase maps to output folder: `outputs/{uid}/`

**Key files:**
- `core/auth.py` — current JSON auth (replace internals)
- `run_desktop.py` — `auth=auth_fn` in app.launch()
- `ui/app.py` — needs signup form
- `manage_users.py` — update to use Firebase Admin SDK

**Dependencies to add:** `firebase-admin` (server-side SDK)

## Current state
- Server runs: `conda run -n catalan-lecture python run_desktop.py`
- Conda env: `catalan-lecture` at `F:\conda_envs\catalan-lecture`
- GPU: RTX 4090 + 5090, CUDA working
- Transcription: faster-whisper with VAD (working well)
- Translation: NLLB (working but bad quality — Task 1 fixes this)
- Auth: JSON-based login (working — Task 2 replaces this)
- History tab: working, per-user output folders
- Share link: `share=True` generates gradio.live URL
