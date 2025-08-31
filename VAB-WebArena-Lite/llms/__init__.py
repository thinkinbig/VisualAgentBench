"""This module is adapt from https://github.com/zeno-ml/zeno-build"""
try:
    from .providers.gemini_utils import generate_from_gemini_completion
except Exception:
    print('Google Cloud not set up, skipping import of providers.gemini_utils.generate_from_gemini_completion')

# Guard optional Hugging Face dependency to avoid import-time failure
try:
    from .providers.hf_utils import generate_from_huggingface_completion
    _HAS_HF = True
except Exception:
    print('Hugging Face text-generation not set up, skipping import of providers.hf_utils.generate_from_huggingface_completion')
    _HAS_HF = False
    def generate_from_huggingface_completion(*args, **kwargs):  # type: ignore[override]
        raise ImportError("HuggingFace provider unavailable. Install 'text-generation' or configure HF backend.")

from .providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from .providers.api_utils import (
    generate_with_api,
)
from .utils import call_llm

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "call_llm",
]
if _HAS_HF:
    __all__.append("generate_from_huggingface_completion")
if 'generate_from_gemini_completion' in globals():
    __all__.append("generate_from_gemini_completion")
