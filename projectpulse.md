# ProjectPulse - Catalan Lecture Processor

- **Status**: Desktop app working end-to-end. Mac Apple Silicon support added. Ready for student deployment.
- **Last worked on**: Fixed PyTorch install for Apple Silicon Macs (CPU-only index has no ARM64 wheels -- now uses default PyPI on Mac). Improved launch.command with Python check and error handling. Updated student guide with detailed Mac setup steps including Homebrew PATH note. Added setup version marker for forced reinstall on logic changes. Previous session fixed Windows UnicodeEncodeError crash.
- **Next steps**:
  - Have student test on their Apple Silicon Mac end-to-end
  - Test Colab notebook on actual Google Colab after Gradio 6.x updates
  - Consider creating a zip distribution excluding venv/.git/audio files
- **Blockers**: None -- needs real Mac testing by student
- **Key files**:
  - `setup_and_run.py` - Auto-setup script (PyTorch platform fix here)
  - `launch.command` - Mac launcher (improved with Python check)
  - `GUIDE_FOR_STUDENTS.md` - Student instructions (Mac section rewritten)
  - `core/pipeline.py` - Main pipeline orchestrator
  - `ui/app.py` - Gradio UI (desktop + Colab shared)
