"""LLM-based lecture summarization with map-reduce for long texts."""

import re
import os
import logging

import requests

from core.config import (
    SUMMARY_PROMPT,
    CHUNK_SUMMARY_PROMPT,
    OLLAMA_URL,
    OLLAMA_MODEL,
    LLM_CHUNK_MAX_WORDS,
)

logger = logging.getLogger(__name__)


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


class Summarizer:
    """Summarize lecture transcripts using available LLM backends."""

    def __init__(self, llm_backend: str = "auto"):
        self._backend = llm_backend
        self._resolved_backend = None

    def _detect_available_backend(self) -> str:
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                return "ollama"
        except (requests.ConnectionError, requests.Timeout):
            pass

        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("HF_TOKEN"):
            return "hf_inference"

        return "none"

    def _get_backend(self) -> str:
        if self._backend != "auto":
            return self._backend
        if self._resolved_backend is None:
            self._resolved_backend = self._detect_available_backend()
        return self._resolved_backend

    def _call_llm(self, prompt: str) -> str | None:
        backend = self._get_backend()
        if backend == "none":
            return None

        try:
            if backend == "ollama":
                return self._call_ollama(prompt)
            elif backend == "gemini":
                return self._call_gemini(prompt)
            elif backend == "hf_inference":
                return self._call_hf_inference(prompt)
        except Exception as e:
            logger.warning("LLM call failed (%s): %s", backend, e)
            return None

    def _call_ollama(self, prompt: str) -> str:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _call_gemini(self, prompt: str) -> str:
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text.strip()

    def _call_hf_inference(self, prompt: str) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.environ.get("HF_TOKEN"))
        response = client.text_generation(
            prompt,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_new_tokens=4096,
        )
        return response.strip()

    def _map_reduce_summarize(self, text: str, language: str, progress_callback=None) -> str | None:
        """Summarize long texts using map-reduce approach."""
        chunks = _chunk_text(text)
        chunk_summaries = []

        # Map phase: summarize each chunk
        for i, chunk in enumerate(chunks):
            if progress_callback:
                frac = 0.82 + 0.06 * ((i + 1) / len(chunks))
                progress_callback(frac, desc=f"Summarizing section {i+1}/{len(chunks)}...")

            prompt = CHUNK_SUMMARY_PROMPT.format(language=language, text=chunk)
            result = self._call_llm(prompt)
            if result:
                chunk_summaries.append(result)
            else:
                return None

        # Reduce phase: final summary from combined chunk summaries
        combined = "\n\n".join(chunk_summaries)
        if progress_callback:
            progress_callback(0.88, desc="Generating final summary...")

        prompt = SUMMARY_PROMPT.format(language=language, text=combined)
        return self._call_llm(prompt)

    def summarize_streaming(self, text: str, language: str = "Catalan"):
        """Generator version of summarize() that yields progress updates.

        Yields dicts: {desc, progress (optional), done: False} during processing,
        then {done: True, result: dict} when complete.
        """
        backend = self._get_backend()
        if backend == "none":
            logger.info("No LLM backend available, skipping summarization")
            yield {"done": True, "result": {
                "raw_summary": None, "main_topics": [],
                "detailed_summary": "", "key_terms": [], "sections": [],
            }}
            return

        word_count = len(text.split())
        raw_summary = None

        if word_count > LLM_CHUNK_MAX_WORDS:
            chunks = _chunk_text(text)
            chunk_summaries = []
            total_ops = len(chunks) + 1  # chunks + final merge

            for i, chunk in enumerate(chunks):
                yield {
                    "desc": f"Summarizing section {i + 1}/{len(chunks)}...",
                    "progress": i / total_ops,
                    "done": False,
                }
                prompt = CHUNK_SUMMARY_PROMPT.format(language=language, text=chunk)
                result = self._call_llm(prompt)
                if result:
                    chunk_summaries.append(result)

            yield {
                "desc": "Generating final summary...",
                "progress": len(chunks) / total_ops,
                "done": False,
            }
            combined = "\n\n".join(chunk_summaries)
            prompt = SUMMARY_PROMPT.format(language=language, text=combined)
            raw_summary = self._call_llm(prompt)
        else:
            yield {"desc": "Generating summary...", "progress": 0.5, "done": False}
            prompt = SUMMARY_PROMPT.format(language=language, text=text)
            raw_summary = self._call_llm(prompt)

        if not raw_summary:
            yield {"done": True, "result": {
                "raw_summary": None, "main_topics": [],
                "detailed_summary": "", "key_terms": [], "sections": [],
            }}
            return

        sections = self._parse_summary_sections(raw_summary)
        yield {"done": True, "result": {
            "raw_summary": raw_summary,
            "main_topics": self._extract_list_items(raw_summary, "Main Topics"),
            "detailed_summary": self._extract_section_text(raw_summary, "Detailed Summary"),
            "key_terms": self._extract_list_items(raw_summary, "Key Terms"),
            "sections": sections,
        }}

    def summarize(self, text: str, language: str = "Catalan", progress_callback=None) -> dict:
        """Generate a structured summary of the lecture transcript.

        Returns:
            dict with keys: raw_summary, main_topics, detailed_summary,
                           key_terms, sections
        """
        if progress_callback:
            progress_callback(0.80, desc="Generating summary...")

        word_count = len(text.split())
        raw_summary = None

        if self._get_backend() == "none":
            logger.info("No LLM backend available, skipping summarization")
        elif word_count > LLM_CHUNK_MAX_WORDS:
            raw_summary = self._map_reduce_summarize(text, language, progress_callback)
        else:
            prompt = SUMMARY_PROMPT.format(language=language, text=text)
            raw_summary = self._call_llm(prompt)

        if not raw_summary:
            return {
                "raw_summary": None,
                "main_topics": [],
                "detailed_summary": "",
                "key_terms": [],
                "sections": [],
            }

        sections = self._parse_summary_sections(raw_summary)

        if progress_callback:
            progress_callback(0.90, desc="Summary complete")

        return {
            "raw_summary": raw_summary,
            "main_topics": self._extract_list_items(raw_summary, "Main Topics"),
            "detailed_summary": self._extract_section_text(raw_summary, "Detailed Summary"),
            "key_terms": self._extract_list_items(raw_summary, "Key Terms"),
            "sections": sections,
        }

    def _parse_summary_sections(self, raw_summary: str) -> list[dict]:
        """Parse markdown-formatted summary into structured sections."""
        sections = []
        current_title = None
        current_bullets = []

        for line in raw_summary.split("\n"):
            line = line.strip()
            if line.startswith("## "):
                if current_title:
                    sections.append({"title": current_title, "bullets": current_bullets})
                current_title = line[3:].strip()
                current_bullets = []
            elif line.startswith("- ") or line.startswith("* "):
                current_bullets.append(line[2:].strip())
            elif line and current_title and not current_bullets:
                current_bullets.append(line)
            elif line and current_title:
                current_bullets.append(line)

        if current_title:
            sections.append({"title": current_title, "bullets": current_bullets})

        return sections

    def _extract_list_items(self, text: str, section_header: str) -> list[str]:
        """Extract bullet points from a specific markdown section."""
        in_section = False
        items = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## ") and section_header.lower() in stripped.lower():
                in_section = True
                continue
            elif stripped.startswith("## "):
                if in_section:
                    break
            elif in_section and (stripped.startswith("- ") or stripped.startswith("* ")):
                items.append(stripped[2:].strip())
        return items

    def _extract_section_text(self, text: str, section_header: str) -> str:
        """Extract paragraph text from a specific markdown section."""
        in_section = False
        paragraphs = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## ") and section_header.lower() in stripped.lower():
                in_section = True
                continue
            elif stripped.startswith("## "):
                if in_section:
                    break
            elif in_section and stripped:
                paragraphs.append(stripped)
        return "\n\n".join(paragraphs)
