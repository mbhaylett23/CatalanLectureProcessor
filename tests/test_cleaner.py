"""Tests for the text cleaner module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cleaner import TextCleaner


def test_removes_single_filler():
    c = TextCleaner(llm_backend="none")
    result = c.regex_clean("Doncs avui parlarem de biologia.")
    assert "doncs" not in result.lower()
    assert "avui" in result.lower()


def test_removes_multiple_fillers():
    c = TextCleaner(llm_backend="none")
    result = c.regex_clean("Eh, doncs, o sigui, la resposta és correcta.")
    assert "eh" not in result.lower().split()
    assert "doncs" not in result.lower().split()
    assert "o sigui" not in result.lower()
    assert "resposta" in result.lower()


def test_preserves_filler_inside_words():
    c = TextCleaner(llm_backend="none")
    result = c.regex_clean("El vehicle és molt gran.")
    assert "vehicle" in result.lower()


def test_handles_case_insensitive():
    c = TextCleaner(llm_backend="none")
    result = c.regex_clean("DONCS és així.")
    assert "doncs" not in result.lower().split()


def test_collapses_spaces():
    c = TextCleaner(llm_backend="none")
    result = c.regex_clean("La  eh  resposta  eh  és correcta.")
    assert "  " not in result


def test_empty_input():
    c = TextCleaner(llm_backend="none")
    assert c.regex_clean("") == ""


def test_no_fillers_unchanged():
    c = TextCleaner(llm_backend="none")
    text = "Avui parlarem de biologia molecular."
    assert c.regex_clean(text) == text


def test_fixes_capitalization():
    c = TextCleaner(llm_backend="none")
    result = c.regex_clean("eh avui és un bon dia.")
    assert result[0].isupper()


def test_clean_returns_dict():
    c = TextCleaner(llm_backend="none")
    result = c.clean("Eh, doncs, avui parlarem de biologia.")
    assert "regex_cleaned" in result
    assert "llm_cleaned" in result
    assert "best" in result
    assert result["llm_cleaned"] is None  # no LLM backend
    assert result["best"] == result["regex_cleaned"]


def test_removes_long_filler_phrase():
    c = TextCleaner(llm_backend="none")
    result = c.regex_clean("Per dir-ho d'alguna manera, la teoria és correcta.")
    assert "per dir-ho d'alguna manera" not in result.lower()
    assert "teoria" in result.lower()


if __name__ == "__main__":
    test_removes_single_filler()
    test_removes_multiple_fillers()
    test_preserves_filler_inside_words()
    test_handles_case_insensitive()
    test_collapses_spaces()
    test_empty_input()
    test_no_fillers_unchanged()
    test_fixes_capitalization()
    test_clean_returns_dict()
    test_removes_long_filler_phrase()
    print("All cleaner tests passed!")
