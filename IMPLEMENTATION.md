# Catalan Lecture Processor - Implementation Reference

## Architecture

```
Audio file (.m4a/.mp3/.wav)
        │
        ▼
┌─────────────────┐
│  Transcriber     │  Whisper large-v3 (Catalan fine-tuned)
│  GPU: transformers pipeline, float16, 30s chunks, batch=16
│  CPU: faster-whisper, INT8, VAD-based batching
└────────┬────────┘
         │ raw text
         ▼
┌─────────────────┐
│  TextCleaner     │  Stage 1: Regex filler removal
│                  │  Stage 2: LLM restructuring (optional)
└────────┬────────┘
         │ clean text
         ▼
┌─────────────────┐
│  Translator      │  NLLB-200-distilled-600M
│                  │  cat_Latn → spa/eng/por/ita_Latn
│                  │  Sentence-level batching (<400 tokens)
└────────┬────────┘
         │ translations
         ▼
┌─────────────────┐
│  Summarizer      │  LLM (Ollama / Gemini / HF Inference)
│                  │  Map-reduce for texts >3000 words
└────────┬────────┘
         │ structured summary
         ▼
┌─────────────────┐
│  SlideGenerator  │  python-pptx, 16:9 widescreen
│                  │  Auto-overflow at 6 bullets/slide
└────────┬────────┘
         │
         ▼
   .txt + .md + .pptx output files
```

---

## File Reference

### `core/config.py`
Central constants. Every other module imports from here.

| Constant | Value | Purpose |
|----------|-------|---------|
| `WHISPER_HF_MODEL` | `projecte-aina/whisper-large-v3-ca-3catparla` | GPU transcription model |
| `WHISPER_FASTER_MODEL` | `projecte-aina/faster-whisper-large-v3-ca-3catparla` | CPU transcription model |
| `WHISPER_FASTER_FALLBACK` | `Systran/faster-whisper-large-v3` | Fallback if Catalan model fails |
| `NLLB_MODEL` | `facebook/nllb-200-distilled-600M` | Translation model |
| `LANGUAGE_CODES` | dict | Maps language names → NLLB FLORES-200 codes |
| `CATALAN_FILLERS` | list of regex | 20 filler patterns, sorted longest-first |
| `CLEANUP_PROMPT` | str | LLM prompt for text restructuring |
| `SUMMARY_PROMPT` | str | LLM prompt for structured summarisation |
| `CHUNK_SUMMARY_PROMPT` | str | LLM prompt for chunk-level summaries (map phase) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2:3b` | Default Ollama model |
| `WHISPER_BATCH_SIZE` | 16 | Chunks processed in parallel |
| `NLLB_MAX_LENGTH` | 512 | Max tokens per translation batch |
| `LLM_CHUNK_MAX_WORDS` | 3000 | Text chunk size for LLM calls |
| `SLIDE_MAX_BULLETS` | 6 | Bullets per slide before overflow |

### `core/transcriber.py`
**Class `Transcriber`**

Two code paths selected by device:

**GPU path** (Colab):
```python
# Uses HuggingFace transformers pipeline
pipeline("automatic-speech-recognition",
    model=model,                    # AutoModelForSpeechSeq2Seq, float16
    chunk_length_s=30,              # 30-second processing windows
    batch_size=16,                  # 16 chunks in parallel
    return_timestamps=True)
result = pipe(audio_path, generate_kwargs={"language": "ca", "task": "transcribe"})
```
- Handles chunking automatically with 5s stride overlap
- A 2-hour lecture = ~240 chunks, ~15 batches → ~5-10 minutes

**CPU path** (Desktop):
```python
# Uses faster-whisper with CTranslate2 backend
model = WhisperModel(model_id, device="cpu", compute_type="int8")
batched = BatchedInferencePipeline(model=model)
segments, info = batched.transcribe(audio_path,
    language="ca", batch_size=16, beam_size=5,
    vad_filter=True,                # Silero VAD skips silence
    vad_parameters={"min_silence_duration_ms": 500})
```
- INT8 quantization: 4x less memory, minimal quality loss
- VAD filtering skips silence → saves 15-30% processing time
- A 1-hour lecture → ~30-60 minutes on modern CPU

**Memory usage:**
- GPU (float16): ~3.1 GB VRAM
- CPU (INT8): ~1.6 GB RAM

### `core/cleaner.py`
**Class `TextCleaner`**

**Stage 1 - Regex** (`regex_clean`):
```python
# Fillers sorted longest-first to avoid partial matches
# e.g., "és a dir" matched before "eh" inside words
pattern = r"\b(?:per dir-ho d'alguna manera|diguem-ne|...)\b"
# re.IGNORECASE | re.UNICODE
```
- Word boundary `\b` prevents matching inside words (e.g., "eh" in "vehicle")
- Post-processing: collapse whitespace, fix spacing around punctuation, re-capitalize after sentence boundaries

**Stage 2 - LLM** (`llm_restructure`):
- Chunks text at sentence boundaries into ~3000-word blocks
- Each chunk sent to LLM with `CLEANUP_PROMPT`
- Backend auto-detection order: Ollama → Gemini → HF Inference → skip

**LLM backend implementations:**

| Backend | API Call | Timeout |
|---------|----------|---------|
| Ollama | `POST localhost:11434/api/generate` with `stream: false` | 180s |
| Gemini | `google.genai.Client().models.generate_content(model="gemini-2.0-flash")` | default |
| HF Inference | `InferenceClient.text_generation(model="meta-llama/Llama-3.2-3B-Instruct")` | default |

### `core/translator.py`
**Class `Translator`**

Key implementation detail - NLLB requires `forced_bos_token_id`:
```python
tokenizer.src_lang = "cat_Latn"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
tgt_token_id = tokenizer.convert_tokens_to_ids("spa_Latn")
outputs = model.generate(**inputs, forced_bos_token_id=tgt_token_id, max_length=512)
```

**Sentence batching** for long texts:
- NLLB max context = 512 tokens (~150-200 words)
- Text split at sentence boundaries (regex: `(?<=[.!?;:])\s+`)
- Sentences grouped into batches with total tokens < 400 (headroom for special tokens)
- A 12,000-word transcript → ~60-80 batches

**Memory:** ~2.4 GB on CPU, ~1.2 GB on GPU (float16)

### `core/summarizer.py`
**Class `Summarizer`**

**Single-pass** (text ≤ 3000 words):
- One LLM call with `SUMMARY_PROMPT`

**Map-reduce** (text > 3000 words, typical for 1-2 hour lectures):
1. **Map**: Split into ~3000-word chunks, summarise each independently
2. **Reduce**: Concatenate chunk summaries, run final `SUMMARY_PROMPT` on combined text

**Output parsing:**
- Markdown `## ` headers → section boundaries
- `- ` lines → bullet points
- Returns structured dict: `{raw_summary, main_topics, detailed_summary, key_terms, sections}`

### `core/slides.py`
**Class `SlideGenerator`**

Uses `python-pptx` with 16:9 widescreen (13.333" × 7.5").

**Slide layouts used:**
- `slide_layouts[0]` → Title Slide (title + subtitle)
- `slide_layouts[1]` → Title and Content (title + bullet body)

**Overflow handling:**
```python
for page_idx in range(0, len(bullets), SLIDE_MAX_BULLETS):  # 6 per slide
    page_bullets = bullets[page_idx:page_idx + SLIDE_MAX_BULLETS]
    slide_title = f"{title} (cont.)" if page_idx > 0 else title
```

**Colour palette:** Dark blue titles (`#1A478A`), dark grey body (`#333333`), light grey accents (`#85929E`)

### `core/pipeline.py`
**Class `LectureProcessor`**

Orchestrates all steps with:
- **Lazy loading**: Models loaded on first use, not at startup
- **Generator-based progress**: `process()` is a generator that yields `(status_message, results_dict)` tuples. The UI iterates over these to show real-time updates.
- **Step-numbered progress**: Each step shows `Step X/5 — StepName` with text-based progress bars (`[████████░░░░░░░░░░░░] 40%`) for steps that support sub-progress (transcription, translation, cleanup, summarization).
- **Streaming sub-components**: `Transcriber._transcribe_cpu_streaming()`, `Translator.translate_text_streaming()`, `TextCleaner.clean_streaming()`, and `Summarizer.summarize_streaming()` are all generators that yield per-segment/batch/chunk progress.
- **Fault tolerance**: Each step wrapped in try/except. If translation fails, you still get transcription + cleanup. Only transcription failure is fatal.
- **Output management**: All files saved to temp directory, paths returned in `all_files` dict

### `ui/app.py`
**Function `create_app(mode)`**

Gradio `gr.Blocks` layout:
- `gr.Audio(type="filepath", sources=["upload"])` - file upload widget
- `gr.CheckboxGroup` - language selection (default: Spanish + English)
- `gr.Textbox(lines=3)` - Status box above tabs showing step progress with text-based bars
- `gr.Tabs` with 4 tabs: Transcript, Translations, Summary, Downloads
- `gr.File(file_count="multiple")` - multi-file download
- Dynamic visibility: translation textboxes show/hide based on checkbox selection
- `process_lecture` is a **generator function** that iterates over `pipeline.process()` and yields output tuples for live UI updates

**Gradio 6.x notes:**

- `show_copy_button` removed from `gr.Textbox` (no longer supported)
- `theme` and `css` passed to `app.launch()`, not `gr.Blocks()` constructor
- `gr.Progress()` does not work with generator functions — use text-based progress bar in Status box instead
- `track_tqdm=True` must NOT be used — it captures HuggingFace download bars causing confusing "Downloading 0%" messages

**Mobile responsiveness:** `gr.themes.Soft()` + CSS `max-width: 960px`

### `colab/lecture_processor.ipynb`
Self-contained notebook. All core code is embedded directly in cells (no imports from `core/`). This means the notebook works standalone without needing the project files.

12 cells: instructions → GPU check → install deps → Gemini key → config → transcriber → cleaner → translator → summariser → slides → pipeline (generator with step progress) → Gradio UI + launch with `share=True`

**Colab-specific notes:** The notebook uses the same generator-based `process()` pattern as the desktop app. Pipeline yields `(status_message, results_dict)` tuples with step numbers (`Step X/5`) and text progress bars. Gradio UI uses a generator `process_lecture` function (not `gr.Progress()` callbacks). `theme`/`css` passed to `launch()` for Gradio 6.x compatibility.

---

## Deployment Modes

### Colab (GPU)

| Aspect | Detail |
|--------|--------|
| GPU | NVIDIA T4, 16 GB VRAM |
| Session limit | ~12 hours |
| Model loading | ~2-3 min first time (cached after) |
| 1-hour lecture | ~5-10 min transcription |
| Public URL | `gradio.live` link, valid 72 hours |
| LLM backend | Gemini free tier (if API key provided) |

### Desktop (CPU)

| Aspect | Detail |
|--------|--------|
| Platform | macOS + Windows |
| RAM needed | ~5 GB peak (Whisper + NLLB loaded) |
| 1-hour lecture | ~30-60 min transcription |
| URL | `http://127.0.0.1:7860` (local only) |
| LLM backend | Ollama (if installed) |
| System deps | Python 3.10+, ffmpeg |

---

## Error Handling Strategy

The pipeline uses progressive degradation:

```
Transcription failed?  → STOP (can't continue without text)
Cleanup failed?        → Use raw transcript as fallback
Translation failed?    → Skip failed languages, continue with rest
Summarisation failed?  → Skip summary, user still gets transcripts + translations
Slides failed?         → Skip slides, user still gets text files
```

LLM backend fallback chain:
```
Ollama available?  → Use it
  ↓ no
Gemini key set?    → Use it
  ↓ no
HF token set?      → Use it
  ↓ no
Skip LLM step      → Regex-only cleanup, no summary
```

---

## Key Technical Decisions

### Why NLLB-200 over OPUS-MT?
OPUS-MT requires separate model per language pair (4 models). NLLB-200 is one model for all pairs. Simpler code, less memory, easier to maintain.

### Why faster-whisper over standard Whisper for CPU?
Standard Whisper on CPU takes 4x longer. faster-whisper uses CTranslate2 (optimised C++ inference) and INT8 quantization. Same accuracy, fraction of the time and memory.

### Why Gradio over Streamlit?
Built-in `share=True` for Colab creates public URL without deployment. Native audio upload widget. Simpler API for ML pipelines. Mobile-friendly out of the box.

### Why regex + LLM for cleanup (not just LLM)?
Regex is deterministic, fast, and always available. LLM might not be available (no Ollama, no API key). Regex handles the mechanical work (filler removal), LLM handles the creative work (restructuring into paragraphs). Two-stage approach gives a usable result even without an LLM.

### Why map-reduce for summarisation?
A 2-hour lecture produces ~15,000-20,000 words. Most LLMs degrade with very long inputs even if context window allows it. Map-reduce keeps each call under 3,000 words for consistent quality.

### Why longest-first filler matching?
The filler list contains both `"eh"` and `"ehm"`. If `"eh"` is matched first, it would partially match inside `"ehm"` and leave a stray `"m"`. Sorting longest-first and using alternation (`|`) in the regex ensures `"ehm"` is matched before `"eh"` gets a chance.

---

## Dependencies

### Python packages

| Package | Version | Purpose |
|---------|---------|---------|
| `faster-whisper` | ≥1.1.0 | CPU transcription (CTranslate2 backend) |
| `transformers` | ≥4.40.0 | GPU transcription + NLLB translation |
| `sentencepiece` | ≥0.2.0 | Tokenizer for NLLB |
| `protobuf` | ≥4.25.0 | Required by sentencepiece |
| `google-generativeai` | ≥0.8.0 | Gemini API client (optional) |
| `huggingface-hub` | ≥0.25.0 | HF Inference API client (optional) |
| `requests` | ≥2.31.0 | HTTP client (Ollama API calls) |
| `python-pptx` | ≥1.0.0 | PowerPoint generation |
| `gradio` | ≥4.40.0 | Web UI framework |
| `pydub` | ≥0.25.1 | Audio format handling |
| `tqdm` | ≥4.66.0 | Progress bars |
| `torch` | ≥2.1.0 | Required for NLLB translation model. Desktop: CPU-only (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). Colab: GPU version installed automatically. |
| `accelerate` | ≥0.30.0 | Model loading optimisation (Colab only) |

### System dependencies
- **ffmpeg** - audio format conversion (required)
- **Ollama** - local LLM runtime (optional, desktop only)

---

## Testing

### Tests that run without ML models (local, fast)

```bash
python tests/test_cleaner.py    # 10 tests - regex filler removal
python tests/test_slides.py     # 3 tests - PowerPoint generation
```

### Tests that need ML models (run on Colab or with models downloaded)

The translator and pipeline integration tests require downloading the Whisper and NLLB models (~4 GB total). These should be run on Colab or on a machine where the models are already cached.

### Manual verification checklist
See the plan file at `.claude/plans/tingly-churning-ember.md` for the full 19-item verification checklist covering audio formats, translation accuracy, mobile browser testing, and error handling.

---

## Model Memory Footprint

| Model | GPU (float16) | CPU (INT8/float32) |
|-------|--------------|-------------------|
| Whisper large-v3 | 3.1 GB VRAM | 1.6 GB RAM (INT8) |
| NLLB-200 600M | 1.2 GB VRAM | 2.4 GB RAM |
| **Total peak** | **4.3 GB** | **~5 GB** |

Both models fit simultaneously on a Colab T4 (16 GB VRAM) with headroom.
On desktop, 8 GB RAM is the practical minimum.
