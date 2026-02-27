# ProjectPulse - Catalan Lecture Processor

- **Status**: New faster-whisper notebook created to replace transformers-based Whisper pipeline. Pending Colab testing.
- **Last worked on**: Created `lecture_processor_faster_whisper.ipynb` using faster-whisper (CTranslate2) with Silero VAD to eliminate hallucinations (repeated words/phrases during silence). Expected 2-4x speedup and ~50% less VRAM vs original. Original notebook also patched with anti-hallucination generate_kwargs.
- **Next steps**:
  - Test `lecture_processor_faster_whisper.ipynb` on Colab with the same 58-min audio that triggered hallucinations
  - Compare transcription quality between faster-whisper (base large-v3) and original (Catalan fine-tune)
  - If faster-whisper works well, consider making it the default student notebook
- **Blockers**: None
- **Key files**:
  - `colab/lecture_processor_faster_whisper.ipynb` - New faster-whisper notebook (recommended for testing)
  - `colab/lecture_processor_simple.ipynb` - Original transformers-based notebook (anti-hallucination patches applied)
  - `core/pipeline.py` - Main pipeline orchestrator (desktop version)
  - `COLAB_TROUBLESHOOTING.md` - Student-facing troubleshooting guide
  - `ui/app.py` - Gradio UI (desktop version)
