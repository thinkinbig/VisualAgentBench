"""
Thought-generation prompt utilities for dataset expansion.

This module provides:
- PROMPT_INSTRUCTIONS: the core instructions string for generating a single "thought".
- get_prompt_instructions(): returns the instructions string.
- build_thought_prompt(...): returns a complete prompt with inputs embedded.

The prompt is inferred from existing thought_history samples to match their tone and structure.
"""

from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    # OpenAI SDK (requires OPENAI_API_KEY)
    from openai import OpenAI  # type: ignore
    _openai_available = True
except Exception:
    OpenAI = None  # type: ignore
    _openai_available = False

# Core instructions that define style, constraints, and output format
PROMPT_INSTRUCTIONS: str = (
    "You are a web navigation reasoning assistant. Given the user’s high-level goal, the "
    "interaction history, and the current webpage state, write ONE concise ‘thought’ that "
    "explains the next micro-step to take and why.\n\n"
    "Instructions:\n"
    "- Write in English, 1–4 sentences.\n"
    "- Begin by briefly stating the current state relative to the goal (e.g., ‘The user has…’, ‘The current webpage shows…’).\n"
    "- Clearly state the single next UI interaction needed to make progress toward the goal.\n"
    "- Justify why this step is necessary and how it aligns with constraints (e.g., city, cuisine, rating, price, transit).\n"
    "- Identify the exact target element by visible name/label and, if helpful, its position or appearance (e.g., ‘the blue “SHOW 17 RESULTS” button at the bottom’).\n"
    "- Do not give imperative commands or click instructions; explain rationale only.\n"
    "- Do not propose alternatives; choose the single best next step.\n"
    "- Do not invent elements. If the needed control is off-screen or hidden, note that scrolling or expanding a menu is required.\n"
    "- Maintain the tone and structure used in previous thoughts (e.g., ‘The user has… To …, the next step is… This action is necessary because… The target element is … and is clearly visible.’).\n\n"
    "Action Space (naming alignment only; do NOT output actions in the thought):\n"
    "- Page Operation Actions: click [id]; type [id] [content] [press_enter_after?0/1]; hover [id]; press [key_comb]; scroll [down]|[up]\n"
    "- Tab Management Actions: new_tab; tab_focus [tab_index]; close_tab\n"
    "- URL Navigation Actions: goto [url]; go_back; go_forward\n"
    "- Completion Action: stop [answer]\n"
    "- Progress Reporting Action: send_msg_to_user(\"message\") — concise, task-relevant updates only (no templates/labels).\n\n"
    "Output:\n"
    "- A single paragraph ‘thought’ following the style and constraints above.\n"
)


def get_prompt_instructions() -> str:
    """Return the core instructions for generating a single thought."""
    return PROMPT_INSTRUCTIONS


def build_thought_prompt(
    goal: str,
    history: Optional[str] = None,
    page_state: Optional[str] = None,
    visible_elements: Optional[str] = None,
) -> str:
    """Build a complete prompt with inputs embedded.

    Args:
        goal: User’s high-level goal.
        history: Prior interactions/filters already set (compact summary is fine).
        page_state: What is currently visible/relevant on the page.
        visible_elements: Key visible UI elements with labels/affordances.

    Returns:
        A string prompt ready to send to the LLM to generate one ‘thought’.
    """
    history = history or ""
    page_state = page_state or ""
    visible_elements = visible_elements or ""

    inputs_block = (
        "Inputs:\n"
        f"- GOAL: {goal}\n"
        f"- HISTORY: {history}\n"
        f"- PAGE_STATE: {page_state}\n"
        f"- VISIBLE_ELEMENTS: {visible_elements}\n\n"
    )

    return PROMPT_INSTRUCTIONS + "\n" + inputs_block


__all__ = [
    "PROMPT_INSTRUCTIONS",
    "get_prompt_instructions",
    "build_thought_prompt",
]


def generate_thought_with_gpt4o(
    goal: str,
    history: Optional[str] = None,
    page_state: Optional[str] = None,
    visible_elements: Optional[str] = None,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 200,
    client: Optional[object] = None,
    chosen_action: Optional[str] = None,
    chosen_element_context: Optional[str] = None,
    full_observation: Optional[str] = None,
) -> str:
    """Generate a single-paragraph thought using GPT-4o family models.

    Requires the OpenAI SDK and an OPENAI_API_KEY environment variable if a client
    is not provided.
    """
    inputs = (
        "Inputs:\n"
        f"- GOAL: {goal}\n"
        f"- HISTORY: {history or ''}\n"
        f"- PAGE_STATE: {page_state or ''}\n"
        f"- VISIBLE_ELEMENTS: {visible_elements or ''}\n"
        + (f"- CHOSEN_ACTION: {chosen_action}\n" if chosen_action else "")
        + (f"- CHOSEN_ELEMENT_CONTEXT:\n{chosen_element_context}\n\n" if chosen_element_context else "\n")
        + (f"- FULL_OBSERVATION:\n{full_observation}\n\n" if full_observation else "")
        + "Output Requirement:\n"
        + (
            "- Return only the single-paragraph ‘thought’ that JUSTIFIES the CHOSEN_ACTION as the best next step, "
            "referencing the visible target element and why it progresses the goal. Do not include the action itself.\n"
            if chosen_action
            else "- Return only the single-paragraph ‘thought’ text, without labels or quotes.\n"
        )
    )

    system_content = (
        PROMPT_INSTRUCTIONS
        + (
            "\nReturn only the single-paragraph thought. Do not include any preface or labels."
            if not chosen_action
            else "\nReturn only the single-paragraph thought that clearly justifies the provided CHOSEN_ACTION."
        )
    )

    if client is None:
        if not _openai_available:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OpenAI SDK not available. Install `openai` and set OPENAI_API_KEY."
            )
        client = OpenAI()  # type: ignore

    resp = client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": inputs},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = resp.choices[0].message.content.strip() if getattr(resp, "choices", None) else ""
    return text



