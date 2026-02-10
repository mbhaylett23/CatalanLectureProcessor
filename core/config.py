"""Central configuration for the Catalan Lecture Processor."""

# ── Model identifiers ─────────────────────────────────────────────────────────

# Whisper models
WHISPER_HF_MODEL = "projecte-aina/whisper-large-v3-ca-3catparla"
WHISPER_FASTER_MODEL = "projecte-aina/faster-whisper-large-v3-ca-3catparla"
WHISPER_FASTER_FALLBACK = "Systran/faster-whisper-large-v3"

# Translation
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# ── Language codes (NLLB FLORES-200 format) ────────────────────────────────────

LANGUAGE_CODES = {
    "Catalan": "cat_Latn",
    "Spanish": "spa_Latn",
    "English": "eng_Latn",
    "Portuguese": "por_Latn",
    "Italian": "ita_Latn",
}

TARGET_LANGUAGES = ["Spanish", "English", "Portuguese", "Italian"]

# ── Catalan filler words/phrases (longest-first for regex) ─────────────────────

CATALAN_FILLERS = [
    r"per dir-ho d'alguna manera",
    r"diguem-ne",
    r"a veure",
    r"o sigui",
    r"vull dir",
    r"és a dir",
    r"llavors",
    r"bueno",
    r"doncs",
    r"saps\?",
    r"vale",
    r"clar",
    r"oi\?",
    r"no\?",
    r"bé",
    r"ehm+",
    r"eh+",
    r"mm+",
    r"um+",
    r"ah+",
]

# ── LLM prompts ───────────────────────────────────────────────────────────────

CLEANUP_PROMPT = """You are a text editor. The following is a transcription of a university \
lecture in Catalan. Clean it up by:
1. Organizing into logical paragraphs
2. Fixing any obvious transcription errors
3. Removing remaining verbal fillers or repetitions
4. Do NOT change the language or the meaning
5. Do NOT add any commentary or explanations
6. Return ONLY the cleaned text

Transcription:
{text}"""

SUMMARY_PROMPT = """You are an academic assistant. Summarize the following university lecture \
transcript. The lecture is in {language}. Provide your summary in {language}.

Format your response as:
## Main Topics
- [bullet points of 5-10 key topics covered]

## Detailed Summary
[2-3 paragraphs summarizing the lecture content]

## Key Terms
- [list of important technical terms or concepts mentioned]

Transcript:
{text}"""

CHUNK_SUMMARY_PROMPT = """You are an academic assistant. Summarize this section of a university \
lecture transcript. The lecture is in {language}. Provide a concise summary in {language} \
capturing the key points, concepts, and any important terminology.

Section:
{text}"""

# ── Ollama config ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# ── Audio settings ─────────────────────────────────────────────────────────────

SUPPORTED_AUDIO_FORMATS = [".m4a", ".mp3", ".wav", ".ogg", ".webm", ".flac"]

# ── Whisper transcription settings ─────────────────────────────────────────────

WHISPER_BATCH_SIZE = 16
WHISPER_BEAM_SIZE = 5
WHISPER_CHUNK_LENGTH_S = 30
WHISPER_VAD_PARAMS = {
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
    "threshold": 0.5,
}

# ── Translation settings ──────────────────────────────────────────────────────

NLLB_MAX_LENGTH = 512
NLLB_BATCH_MAX_TOKENS = 400

# ── Summarization settings ─────────────────────────────────────────────────────

LLM_CHUNK_MAX_WORDS = 3000

# ── PPTX settings ─────────────────────────────────────────────────────────────

SLIDE_MAX_BULLETS = 6
