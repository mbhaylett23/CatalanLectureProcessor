# Claude Handoff

## Summary
Codex corrected the incomplete Copilot implementation. Gemini-first translation, runtime Gemini API-key support, Firebase-capable auth, and a reachable signup flow are now in place, but they still need live end-to-end verification against real Firebase/Gemini credentials.

## What Changed
- `core/translator.py`
  Gemini-first translation path added with `GeminiTranslator`, ~3000-word chunking, and automatic NLLB fallback.
- `core/genai.py`
  Shared helper added for resolving `GOOGLE_GENAI_API_KEY` and legacy `GEMINI_API_KEY`.
- `core/cleaner.py`, `core/summarizer.py`, `core/pipeline.py`, `ui/app.py`
  Runtime Gemini API key now flows from the UI through the pipeline so one request can use the same key consistently.
- `core/auth.py`
  Hybrid auth layer added: Firebase email/password login via REST when configured, JSON auth preserved as fallback, Firebase UID used for output folders when available.
- `run_desktop.py`
  Login page now shows a signup link when Firebase is configured, and the server exposes a real `/signup` page that creates Firebase users.
- `manage_users.py`
  Updated to use the new auth layer, including Firebase Admin-backed operations when admin credentials are configured.
- `requirements.txt`
  Added `firebase-admin`.

## Validation Already Done
- `py_compile` passed for the touched Python files.
- `create_app()` and `run_desktop` import cleanly with the `catalan-lecture` interpreter.
- Direct assertion scripts passed for:
  JSON auth fallback
  Firebase login/signup request flow (mocked)
  Gemini/NLLB translation routing and fallback
  Cleaner/slides sanity checks

## What Claude Should Verify Next
1. Ensure the `catalan-lecture` env has current dependencies installed, especially `firebase-admin`.
2. Run `conda run -n catalan-lecture python run_desktop.py`.
3. With `FIREBASE_WEB_API_KEY` set, confirm the Gradio login page shows the signup link and `/signup` creates a usable account.
4. Log in with that new Firebase account and verify outputs save under `outputs/{uid}/...`.
5. Check `manage_users.py` against real Firebase Admin credentials.
6. Enter a real Gemini key in the UI and verify translation quality improves on scientific text.
7. Leave the Gemini key blank and confirm translation falls back to NLLB without crashing.
8. Fix any runtime issues or UX rough edges found during the real-service test pass.

## Required Environment For Full Verification
- Firebase login/signup:
  `FIREBASE_WEB_API_KEY`
- Firebase Admin operations and reliable UID lookup:
  `FIREBASE_SERVICE_ACCOUNT_FILE` or `FIREBASE_SERVICE_ACCOUNT_JSON`
- Gemini translation test:
  `GOOGLE_GENAI_API_KEY` or enter the key in the UI

## Known Risks / Limitations
- If Firebase Admin credentials are missing, signup/login still work through Firebase REST, but UID lookup may fall back to a sanitized email when cache/admin lookup is unavailable.
- `pytest` is not installed in `F:\conda_envs\catalan-lecture`.
- The checked-in `venv` points at a missing base Python and is not usable for validation.

## Key Files
- `CLAUDE_HANDOFF.md`
- `core/translator.py`
- `core/auth.py`
- `run_desktop.py`
- `ui/app.py`
