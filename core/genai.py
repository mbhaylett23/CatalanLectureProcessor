"""Helpers for resolving Google GenAI configuration."""

import os

GOOGLE_GENAI_API_KEY_ENV_VARS = (
    "GOOGLE_GENAI_API_KEY",
    "GEMINI_API_KEY",
)


def resolve_google_genai_api_key(explicit_api_key: str | None = None) -> str | None:
    """Return the first non-empty Google GenAI API key."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    for env_var in GOOGLE_GENAI_API_KEY_ENV_VARS:
        value = os.environ.get(env_var)
        if value and value.strip():
            return value.strip()
    return None


def has_google_genai_api_key(explicit_api_key: str | None = None) -> bool:
    """Return True when a Google GenAI API key is available."""
    return resolve_google_genai_api_key(explicit_api_key) is not None


def get_genai_client(explicit_api_key: str | None = None):
    """Build a Google GenAI client for the resolved key."""
    api_key = resolve_google_genai_api_key(explicit_api_key)
    if not api_key:
        raise ValueError("No Google GenAI API key is configured")

    from google import genai

    return genai.Client(api_key=api_key)
