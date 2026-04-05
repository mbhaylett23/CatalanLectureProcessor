"""Tests for Gemini/NLLB translation routing."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import translator as translator_module


class _FakeGeminiTranslator:
    def __init__(self, result="gemini-output", exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def translate_text_streaming(self, text, source_lang, target_lang, api_key=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        yield {
            "progress": 0.0,
            "batch": 1,
            "total": 1,
            "done": False,
            "backend": "Gemini",
        }
        yield {"done": True, "result": self.result, "backend": "Gemini"}


class _FakeNLLBTranslator:
    def __init__(self, result="nllb-output"):
        self.result = result
        self.calls = 0

    def translate_text_streaming(self, text, source_lang, target_lang):
        self.calls += 1
        yield {
            "progress": 0.0,
            "batch": 1,
            "total": 1,
            "done": False,
            "backend": "NLLB",
        }
        yield {"done": True, "result": self.result, "backend": "NLLB"}

    def unload(self):
        return None


def _collect_updates(generator):
    return list(generator)


def test_prefers_gemini_when_key_available(monkeypatch):
    monkeypatch.setattr(translator_module, "has_google_genai_api_key", lambda api_key=None: True)
    translator = translator_module.Translator(device="cpu")
    translator._gemini = _FakeGeminiTranslator(result="gemini wins")
    translator._nllb = _FakeNLLBTranslator(result="nllb fallback")

    updates = _collect_updates(
        translator.translate_text_streaming(
            "Bon dia.",
            "Catalan",
            "English",
            api_key="abc123",
        )
    )

    assert updates[-1]["result"] == "gemini wins"
    assert updates[-1]["backend"] == "Gemini"
    assert translator._gemini.calls == 1
    assert translator._nllb.calls == 0


def test_falls_back_to_nllb_when_gemini_errors(monkeypatch):
    monkeypatch.setattr(translator_module, "has_google_genai_api_key", lambda api_key=None: True)
    translator = translator_module.Translator(device="cpu")
    translator._gemini = _FakeGeminiTranslator(exc=RuntimeError("boom"))
    translator._nllb = _FakeNLLBTranslator(result="nllb wins")

    updates = _collect_updates(
        translator.translate_text_streaming(
            "Bon dia.",
            "Catalan",
            "English",
            api_key="abc123",
        )
    )

    assert updates[-1]["result"] == "nllb wins"
    assert updates[-1]["backend"] == "NLLB"
    assert translator._gemini.calls == 1
    assert translator._nllb.calls == 1


def test_chunks_long_text_for_gemini():
    text = " ".join(f"word{i}" for i in range(6505))
    chunks = translator_module._split_text_by_words(text, 3000)

    assert len(chunks) == 3
    assert sum(len(chunk.split()) for chunk in chunks) == 6505
