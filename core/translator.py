"""Translation with Gemini-first behavior and NLLB fallback."""

import logging
import re

from core.config import (
    GEMINI_MODEL,
    LANGUAGE_CODES,
    NLLB_BATCH_MAX_TOKENS,
    NLLB_MAX_LENGTH,
    NLLB_MODEL,
    TRANSLATION_CHUNK_MAX_WORDS,
    TRANSLATION_PROMPT,
)
from core.genai import get_genai_client, has_google_genai_api_key

logger = logging.getLogger(__name__)


def _remove_hallucinations(text: str) -> str:
    """Post-processing safety net: collapse repeated words/phrases."""
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+\s+\w+)(?:\s+\1){1,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"  +", " ", text).strip()
    return text


def _split_text_by_words(text: str, max_words: int) -> list[str]:
    """Split text into sentence-aware chunks of roughly max_words words."""
    if not text or not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    if len(sentences) <= 1:
        words = text.split()
        return [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        word_count = len(sentence.split())
        if word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0
            chunks.extend(_split_text_by_words(sentence, max_words))
            continue

        if current_words + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(sentence)
        current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class NLLBTranslator:
    """Translate text between languages using NLLB-200-distilled-600M."""

    def __init__(self, model_id: str | None = None, device: str = "auto"):
        self._model_id = model_id or NLLB_MODEL
        self._device = self._resolve_device(device)
        self._model = None
        self._tokenizer = None

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        """Load NLLB model and tokenizer."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading translation model: %s on %s", self._model_id, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_id)
        if self._device == "cuda":
            self._model = self._model.half().to("cuda")
        else:
            self._model = self._model.to("cpu")
        logger.info("Translation model loaded")

    def _ensure_loaded(self):
        if self._model is None:
            self._load_model()

    def _split_into_sentence_batches(self, text: str) -> list[str]:
        """Split text into batches that fit within token limits."""
        chunks = re.split(r"(?<=[.!?;:])\s+", text)

        if len(chunks) <= 1 and len(text) > 200:
            words = text.split()
            chunks = []
            current = []
            current_tokens = 0
            for word in words:
                word_tokens = len(self._tokenizer.encode(word, add_special_tokens=False))
                if current_tokens + word_tokens > NLLB_BATCH_MAX_TOKENS and current:
                    chunks.append(" ".join(current))
                    current = []
                    current_tokens = 0
                current.append(word)
                current_tokens += word_tokens
            if current:
                chunks.append(" ".join(current))
            return chunks

        batches = []
        current_batch = []
        current_tokens = 0

        for sentence in chunks:
            tokens = len(self._tokenizer.encode(sentence, add_special_tokens=False))
            if current_tokens + tokens > NLLB_BATCH_MAX_TOKENS and current_batch:
                batches.append(" ".join(current_batch))
                current_batch = []
                current_tokens = 0
            current_batch.append(sentence)
            current_tokens += tokens

        if current_batch:
            batches.append(" ".join(current_batch))
        return batches

    def translate_text_streaming(self, text: str, source_lang: str, target_lang: str):
        """Yield progress updates while translating with NLLB."""
        if not text or not text.strip():
            yield {"done": True, "result": "", "backend": "NLLB"}
            return

        self._ensure_loaded()

        src_code = LANGUAGE_CODES[source_lang]
        tgt_code = LANGUAGE_CODES[target_lang]
        tgt_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

        if tgt_token_id == self._tokenizer.unk_token_id:
            raise ValueError(f"Invalid target language code: {tgt_code}")

        self._tokenizer.src_lang = src_code
        batches = self._split_into_sentence_batches(text)
        translated_parts = []

        for i, batch in enumerate(batches):
            yield {
                "progress": i / max(len(batches), 1),
                "batch": i + 1,
                "total": len(batches),
                "done": False,
                "backend": "NLLB",
            }
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=NLLB_MAX_LENGTH,
            )
            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=NLLB_MAX_LENGTH,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
            decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated_parts.extend(decoded)

        yield {
            "done": True,
            "result": _remove_hallucinations(" ".join(translated_parts)),
            "backend": "NLLB",
        }

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source to target language."""
        result = ""
        for update in self.translate_text_streaming(text, source_lang, target_lang):
            if update["done"]:
                result = update["result"]
        return result

    def unload(self):
        """Free model memory."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc

        gc.collect()


class GeminiTranslator:
    """Translate text with Google Gemini."""

    def __init__(self):
        self._client = None
        self._api_key = None

    def _get_client(self, api_key: str | None = None):
        if self._client is None or self._api_key != api_key:
            self._client = get_genai_client(api_key)
            self._api_key = api_key
        return self._client

    def _translate_chunk(
        self,
        chunk: str,
        target_lang: str,
        api_key: str | None = None,
    ) -> str:
        client = self._get_client(api_key)
        prompt = TRANSLATION_PROMPT.format(language=target_lang, text=chunk)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return (response.text or "").strip()

    def translate_text_streaming(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        api_key: str | None = None,
    ):
        """Yield progress updates while translating with Gemini."""
        if source_lang != "Catalan":
            raise ValueError("GeminiTranslator currently expects Catalan source text")
        if not text or not text.strip():
            yield {"done": True, "result": "", "backend": "Gemini"}
            return

        chunks = _split_text_by_words(text, TRANSLATION_CHUNK_MAX_WORDS)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            yield {
                "progress": i / max(len(chunks), 1),
                "batch": i + 1,
                "total": len(chunks),
                "done": False,
                "backend": "Gemini",
            }
            translated_parts.append(
                self._translate_chunk(
                    chunk,
                    target_lang=target_lang,
                    api_key=api_key,
                )
            )

        yield {
            "done": True,
            "result": "\n\n".join(translated_parts).strip(),
            "backend": "Gemini",
        }

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        api_key: str | None = None,
    ) -> str:
        """Translate text from source to target language."""
        result = ""
        for update in self.translate_text_streaming(
            text,
            source_lang,
            target_lang,
            api_key=api_key,
        ):
            if update["done"]:
                result = update["result"]
        return result


class Translator:
    """High-level translator that prefers Gemini and falls back to NLLB."""

    def __init__(self, model_id: str | None = None, device: str = "auto"):
        self._nllb = NLLBTranslator(model_id=model_id, device=device)
        self._gemini = GeminiTranslator()

    def _should_use_gemini(self, api_key: str | None = None) -> bool:
        return has_google_genai_api_key(api_key)

    def _ensure_loaded(self, api_key: str | None = None):
        if not self._should_use_gemini(api_key):
            self._nllb._ensure_loaded()

    def translate_text_streaming(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        api_key: str | None = None,
    ):
        """Yield progress updates while translating with the best backend."""
        if self._should_use_gemini(api_key):
            try:
                yield from self._gemini.translate_text_streaming(
                    text,
                    source_lang,
                    target_lang,
                    api_key=api_key,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Gemini translation failed for %s, falling back to NLLB: %s",
                    target_lang,
                    exc,
                )

        yield from self._nllb.translate_text_streaming(text, source_lang, target_lang)

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        api_key: str | None = None,
    ) -> str:
        """Translate text from source to target language."""
        result = ""
        for update in self.translate_text_streaming(
            text,
            source_lang,
            target_lang,
            api_key=api_key,
        ):
            if update["done"]:
                result = update["result"]
        return result

    def translate_to_languages(
        self,
        text: str,
        target_languages: list[str],
        progress_callback=None,
        api_key: str | None = None,
    ) -> dict[str, str]:
        """Translate text to multiple target languages."""
        translations = {}
        for i, lang in enumerate(target_languages):
            if progress_callback:
                frac = 0.60 + 0.20 * (i / max(len(target_languages), 1))
                progress_callback(frac, desc=f"Translating to {lang}...")

            try:
                translations[lang] = self.translate_text(
                    text,
                    "Catalan",
                    lang,
                    api_key=api_key,
                )
            except Exception as e:
                logger.error("Translation to %s failed: %s", lang, e)
                translations[lang] = f"[Translation to {lang} failed: {e}]"

        if progress_callback:
            progress_callback(0.80, desc="Translation complete")

        return translations

    def unload(self):
        """Free local model memory."""
        self._nllb.unload()
