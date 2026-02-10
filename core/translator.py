"""Translation using NLLB-200 (supports Catalan → Spanish/English/Portuguese/Italian)."""

import re
import logging

from core.config import NLLB_MODEL, LANGUAGE_CODES, NLLB_MAX_LENGTH, NLLB_BATCH_MAX_TOKENS

logger = logging.getLogger(__name__)


class Translator:
    """Translate text between languages using NLLB-200-distilled-600M."""

    def __init__(self, model_id: str = None, device: str = "auto"):
        """
        Args:
            model_id: Override default NLLB model.
            device: "auto", "cuda", or "cpu".
        """
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
        """Split text into batches of sentences that fit within token limits."""
        sentences = re.split(r"(?<=[.!?;:])\s+", text)
        batches = []
        current_batch = []
        current_tokens = 0

        for sentence in sentences:
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
        """Generator that yields progress during translation of a single language.

        Yields dicts: {progress, batch, total, done} during processing,
        then {done: True, result: str} when complete.
        """
        if not text or not text.strip():
            yield {"done": True, "result": ""}
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
            )
            decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated_parts.extend(decoded)

        yield {"done": True, "result": " ".join(translated_parts)}

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source to target language.

        Args:
            text: Text to translate.
            source_lang: Key from LANGUAGE_CODES (e.g., "Catalan").
            target_lang: Key from LANGUAGE_CODES (e.g., "Spanish").

        Returns:
            Translated text.
        """
        if not text or not text.strip():
            return ""

        self._ensure_loaded()

        src_code = LANGUAGE_CODES[source_lang]
        tgt_code = LANGUAGE_CODES[target_lang]
        tgt_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

        if tgt_token_id == self._tokenizer.unk_token_id:
            raise ValueError(f"Invalid target language code: {tgt_code}")

        self._tokenizer.src_lang = src_code
        batches = self._split_into_sentence_batches(text)
        translated_parts = []

        for batch in batches:
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
            )
            decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated_parts.extend(decoded)

        return " ".join(translated_parts)

    def translate_to_languages(
        self,
        text: str,
        target_languages: list[str],
        progress_callback=None,
    ) -> dict[str, str]:
        """Translate text to multiple target languages.

        Args:
            text: Source text in Catalan.
            target_languages: List of language names (e.g., ["Spanish", "English"]).
            progress_callback: Optional callable(fraction, desc=str).

        Returns:
            Dict mapping language name to translated text.
        """
        translations = {}
        for i, lang in enumerate(target_languages):
            if progress_callback:
                frac = 0.60 + 0.20 * (i / len(target_languages))
                progress_callback(frac, desc=f"Translating to {lang}...")

            try:
                translations[lang] = self.translate_text(text, "Catalan", lang)
            except Exception as e:
                logger.error("Translation to %s failed: %s", lang, e)
                translations[lang] = f"[Translation to {lang} failed: {e}]"

        if progress_callback:
            progress_callback(0.80, desc="Translation complete")

        return translations

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
