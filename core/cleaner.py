"""Text cleanup: regex filler removal + optional LLM restructuring."""

import re
import os
import logging

import requests

from core.config import (
    CATALAN_FILLERS,
    CLEANUP_PROMPT,
    OLLAMA_URL,
    OLLAMA_MODEL,
    LLM_CHUNK_MAX_WORDS,
)

logger = logging.getLogger(__name__)


def _build_filler_pattern(fillers: list[str]) -> re.Pattern:
    """Build a compiled regex that matches any filler as a whole word."""
    sorted_fillers = sorted(fillers, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(sorted_fillers) + r")\b"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces and fix spacing around punctuation."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([.,;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_capitalization(text: str) -> str:
    """Capitalize first letter after sentence-ending punctuation."""
    def upper_match(m):
        return m.group(1) + m.group(2).upper()
    text = re.sub(r"([.!?]\s+)([a-záàéèíïóòúüç])", upper_match, text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _chunk_text(text: str, max_words: int = LLM_CHUNK_MAX_WORDS) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk, current_count = [], [], 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_count = [], 0
        current_chunk.append(sentence)
        current_count += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


class TextCleaner:
    """Clean transcribed text using regex and optional LLM restructuring."""

    def __init__(self, llm_backend: str = "auto"):
        """
        Args:
            llm_backend: "ollama", "gemini", "hf_inference", "none", or "auto".
                         "auto" probes available backends in priority order.
        """
        self._filler_pattern = _build_filler_pattern(CATALAN_FILLERS)
        self._backend = llm_backend
        self._resolved_backend = None

    def _detect_available_backend(self) -> str:
        """Probe which LLM backends are available."""
        # Try Ollama
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                logger.info("LLM backend: Ollama detected")
                return "ollama"
        except (requests.ConnectionError, requests.Timeout):
            pass

        # Try Gemini
        if os.environ.get("GEMINI_API_KEY"):
            logger.info("LLM backend: Gemini API key found")
            return "gemini"

        # Try HuggingFace Inference
        if os.environ.get("HF_TOKEN"):
            logger.info("LLM backend: HF Inference token found")
            return "hf_inference"

        logger.info("LLM backend: none available, regex-only mode")
        return "none"

    def _get_backend(self) -> str:
        if self._backend != "auto":
            return self._backend
        if self._resolved_backend is None:
            self._resolved_backend = self._detect_available_backend()
        return self._resolved_backend

    def regex_clean(self, text: str) -> str:
        """Remove Catalan fillers using regex, then clean up whitespace."""
        if not text:
            return ""
        cleaned = self._filler_pattern.sub("", text)
        cleaned = _collapse_whitespace(cleaned)
        cleaned = _fix_capitalization(cleaned)
        return cleaned

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama local API."""
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini free API."""
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    def _call_hf_inference(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.environ.get("HF_TOKEN"))
        response = client.text_generation(
            prompt,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_new_tokens=4096,
        )
        return response.strip()

    def _call_llm(self, prompt: str) -> str | None:
        """Call the selected LLM backend. Returns None on failure."""
        backend = self._get_backend()
        if backend == "none":
            return None

        callers = {
            "ollama": self._call_ollama,
            "gemini": self._call_gemini,
            "hf_inference": self._call_hf_inference,
        }
        caller = callers.get(backend)
        if not caller:
            return None

        try:
            return caller(prompt)
        except Exception as e:
            logger.warning("LLM call failed (%s): %s", backend, e)
            return None

    def llm_restructure(self, text: str, progress_callback=None) -> str | None:
        """Send text to LLM for paragraph restructuring.

        Chunks long texts to stay within context limits.
        Returns restructured text or None if LLM is unavailable.
        """
        if self._get_backend() == "none":
            return None

        chunks = _chunk_text(text)
        restructured_parts = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                frac = 0.52 + 0.08 * ((i + 1) / len(chunks))
                progress_callback(frac, desc=f"Restructuring text... chunk {i+1}/{len(chunks)}")

            prompt = CLEANUP_PROMPT.format(text=chunk)
            result = self._call_llm(prompt)
            if result:
                restructured_parts.append(result)
            else:
                restructured_parts.append(chunk)

        return "\n\n".join(restructured_parts)

    def clean_streaming(self, text: str):
        """Generator version of clean() that yields progress updates.

        Yields dicts: {desc, progress (optional), done: False} during processing,
        then {done: True, result: dict} when complete.
        """
        yield {"desc": "Removing filler words...", "done": False}
        regex_cleaned = self.regex_clean(text)

        backend = self._get_backend()
        if backend == "none":
            yield {"done": True, "result": {
                "regex_cleaned": regex_cleaned,
                "llm_cleaned": None,
                "best": regex_cleaned,
                "llm_backend_used": "none",
            }}
            return

        chunks = _chunk_text(regex_cleaned)
        restructured_parts = []
        for i, chunk in enumerate(chunks):
            yield {
                "desc": f"LLM restructuring (chunk {i + 1}/{len(chunks)})...",
                "progress": i / max(len(chunks), 1),
                "done": False,
            }
            prompt = CLEANUP_PROMPT.format(text=chunk)
            result = self._call_llm(prompt)
            restructured_parts.append(result if result else chunk)

        llm_cleaned = "\n\n".join(restructured_parts)
        yield {"done": True, "result": {
            "regex_cleaned": regex_cleaned,
            "llm_cleaned": llm_cleaned,
            "best": llm_cleaned,
            "llm_backend_used": backend,
        }}

    def clean(self, text: str, progress_callback=None) -> dict:
        """Full cleaning pipeline.

        Returns:
            dict with keys: regex_cleaned, llm_cleaned, best, llm_backend_used
        """
        if progress_callback:
            progress_callback(0.50, desc="Removing filler words...")

        regex_cleaned = self.regex_clean(text)

        if progress_callback:
            progress_callback(0.52, desc="Restructuring text with LLM...")

        llm_cleaned = self.llm_restructure(regex_cleaned, progress_callback)
        backend_used = self._get_backend()

        return {
            "regex_cleaned": regex_cleaned,
            "llm_cleaned": llm_cleaned,
            "best": llm_cleaned if llm_cleaned else regex_cleaned,
            "llm_backend_used": backend_used,
        }
