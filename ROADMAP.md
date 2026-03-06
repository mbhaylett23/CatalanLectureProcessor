# Roadmap

## Immediate (do now)

- [ ] Get a free Gemini API key at https://ai.google.dev/ and paste it into the notebook's `GEMINI_API_KEY` field — this unlocks text cleanup, summarization, and slide generation on Colab
- [ ] Test the notebook end-to-end on Colab with a T4 GPU runtime and Gemini key set
- [ ] Send email to Cristina about BSC hosting for IQS students

## Short term

### Source language support (Spanish input)
Currently the pipeline is hardcoded for Catalan input:
- Whisper model: `projecte-aina/faster-whisper-large-v3-ca-3catparla` (Catalan-specific)
- Transcribe call: `language="ca"` hardcoded
- Translation: `source_lang="Catalan"` hardcoded
- Filler word removal: Catalan-only regex patterns

To support Spanish (or other languages) as input:
- [ ] Add a source language dropdown to the UI (Catalan, Spanish, etc.)
- [ ] Use a general multilingual Whisper model (e.g. `Systran/faster-whisper-large-v3`) when source is not Catalan, or auto-detect
- [ ] Pass the selected source language to the translator instead of hardcoding "Catalan"
- [ ] Add Spanish filler word patterns (e.g. "o sea", "pues", "bueno", "eh", "a ver")
- [ ] Skip translation to the source language (don't translate Spanish to Spanish)

### LLM backend improvements
- [ ] Add a Colab UI field for HuggingFace token as an alternative to Gemini
- [ ] Consider running a small model directly on the T4 GPU (e.g. Phi-3-mini or TinyLlama) as a fallback when no API key is provided — would need to manage GPU memory carefully alongside Whisper/NLLB

## Medium term — BSC hosting

Goal: a permanent URL where IQS students can go, upload audio, and get results without setting up Colab.

### What we'd need from BSC
- A small persistent GPU instance (even a single T4 or A10 would work)
- A public-facing URL or one accessible from the IQS network
- Storage for cached models (~4 GB for Whisper + NLLB)

### Architecture on BSC
- [ ] Host with Ollama or vLLM for the LLM backend (replaces Gemini dependency)
- [ ] Good open-source LLM candidates:
  - Llama 3.1 8B (~5 GB, great multilingual quality)
  - Mistral 7B (~4 GB, fast, good at structured output)
  - BSC/AINA's own Catalan models (natural fit, worth asking Cristina about)
- [ ] Run the Gradio app as a persistent service (systemd or Docker)
- [ ] Add basic auth or IQS institutional login to restrict access
- [ ] Set up model caching so cold starts are fast

### Deployment options comparison
See `DEPLOYMENT_OPTIONS.md` for full platform comparison. BSC is ideal because:
- Free for academic/research use
- Aligns with Projecte AINA's Catalan language mission
- No session timeouts like Colab
- Can host local LLM (no external API dependency)

## Long term / nice to have

- [ ] Batch processing — upload multiple lectures, process overnight
- [ ] Speaker diarization — identify different speakers in the lecture
- [ ] Timestamp alignment — link transcript segments to audio timestamps
- [ ] Vocabulary glossary — extract and define domain-specific terms per course
- [ ] Student accounts — save processing history, re-download past results
- [ ] Support more source languages beyond Catalan and Spanish (French, Portuguese, etc.)
