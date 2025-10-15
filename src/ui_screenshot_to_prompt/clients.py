"""Utilities for lazily initializing external service clients."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

from .config import load_and_initialize_clients

logger = logging.getLogger(__name__)

_OPENAI_CLIENT = None
_SUPER_PROMPT_FUNCTION: Optional[Callable[[str], str]] = None


def _ensure_clients() -> Tuple[object, Optional[Callable[[str], str]]]:
    global _OPENAI_CLIENT, _SUPER_PROMPT_FUNCTION
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT, _SUPER_PROMPT_FUNCTION = load_and_initialize_clients()
    return _OPENAI_CLIENT, _SUPER_PROMPT_FUNCTION


def ensure_openai_client():
    """Return a configured OpenAI client, reinitializing if necessary."""
    client, _ = _ensure_clients()
    if client is None:
        raise RuntimeError("OpenAI client is not configured")
    return client


def ensure_super_prompt_function() -> Callable[[str], str]:
    """Return a callable that can generate super prompts."""
    _, prompt_fn = _ensure_clients()
    if prompt_fn is None:
        raise ValueError("No API client available for super prompt generation")
    return prompt_fn
