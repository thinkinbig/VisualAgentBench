from importlib import import_module
from typing import Any

# Lazy export heavy submodules to avoid importing optional deps (e.g., tiktoken)
__all__ = [
    "Agent",
    "TeacherForcingAgent",
    "PromptAgent",
    "RewardGuidedAgent",
    "construct_agent",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name in {"Agent", "TeacherForcingAgent", "PromptAgent", "construct_agent"}:
        mod = import_module("agent.agent")
        return getattr(mod, name)
    if name == "RewardGuidedAgent":
        mod = import_module("agent.reward_guided_agent")
        return getattr(mod, name)
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
