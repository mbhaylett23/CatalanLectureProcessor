# ProjectPulse - Catalan Lecture Processor

- **Status**: UI defaults changed to English; summarization/slides require Gemini API key or LLM backend
- **Last worked on**: Changed notebook UI default language from Catalan to English (dropdown, state, tab names, HTML lang, skip-nav, fallbacks). Discussed BSC hosting for permanent student access and Gemini API key as quick fix for summarization.
- **Next steps**:
  - Get a free Gemini API key to enable summarization and slides on Colab
  - Draft and send email to Cristina about BSC hosting
  - Consider adding source language selection (Spanish input support)
- **Blockers**: No LLM backend configured on Colab (summarization/slides skipped without Gemini key)
- **Key files**:
  - `colab/lecture_processor_faster_whisper.ipynb` - Main notebook (now defaults to English UI)
  - `core/pipeline.py` - Desktop pipeline orchestrator
  - `core/config.py` - Central configuration
  - `ui/app.py` - Gradio UI (desktop version)
  - `DEPLOYMENT_OPTIONS.md` - Platform comparison
