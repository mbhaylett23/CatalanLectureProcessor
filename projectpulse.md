# ProjectPulse - Catalan Lecture Processor

- **Status**: Gemini-first translation and Firebase-capable auth/signup are implemented; live end-to-end verification against real credentials is the main remaining step.
- **Last worked on**: Added Gemini translation/key plumbing, Firebase-capable auth, signup route, admin CLI updates, manual validation scripts, and a fresh `CLAUDE_HANDOFF.md`.
- **Next steps**:
  - Verify `/signup`, Gradio login, UID-based output folders, and `manage_users.py` with real Firebase credentials
  - Verify Gemini translation in the UI with a real API key and confirm NLLB fallback when the key is absent
  - Fix any runtime or UX issues Claude finds during live-service testing
- **Blockers**: Full verification requires `FIREBASE_WEB_API_KEY`, Firebase admin credentials, and a real Gemini key; `pytest` is not installed in the conda env.
- **Key files**:
  - `CLAUDE_HANDOFF.md`
  - `core/translator.py`
  - `core/auth.py`
  - `run_desktop.py`
  - `ui/app.py`
