"""Lightweight llms package init.

Avoid importing provider backends at import time to prevent optional dependency
errors during unrelated imports (e.g., llms.types).
"""

from importlib import import_module
from typing import Any

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_with_api",
    "generate_from_huggingface_completion",
    "generate_from_gemini_completion",
    "call_llm",
    "lm_config",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name in {"generate_from_openai_completion", "generate_from_openai_chat_completion"}:
        mod = import_module("llms.providers.openai_utils")
        return getattr(mod, name)
    if name == "generate_with_api":
        mod = import_module("llms.providers.api_utils")
        return getattr(mod, name)
    if name == "call_llm":
        mod = import_module("llms.utils")
        return getattr(mod, name)
    if name == "lm_config":
        mod = import_module("llms.lm_config")
        return mod
    if name == "generate_from_huggingface_completion":
        try:
            mod = import_module("llms.providers.hf_utils")
            return getattr(mod, name)
        except Exception as e:
            raise ImportError("HuggingFace provider unavailable. Install required dependencies.") from e
    if name == "generate_from_gemini_completion":
        try:
            mod = import_module("llms.providers.gemini_utils")
            return getattr(mod, name)
        except Exception as e:
            raise ImportError("Gemini provider unavailable. Configure Google Cloud to enable it.") from e
    raise AttributeError(f"module 'llms' has no attribute {name!r}")
