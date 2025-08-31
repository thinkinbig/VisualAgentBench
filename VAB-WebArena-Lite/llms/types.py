"""
LLM-layer types and re-exports.

Note: Policy/Reward request-response schemas live in agent/types.py.
This module re-exports them to avoid breaking existing imports during migration.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
from agent.types import (
    PolicyResponse,
)

class ActionType(str, Enum):
    """Valid action types (exactly one per turn)."""
    CLICK = "click"
    TYPE = "type"
    HOVER = "hover"
    PRESS = "press"
    SCROLL = "scroll"
    NEW_TAB = "new_tab"
    TAB_FOCUS = "tab_focus"
    CLOSE_TAB = "close_tab"
    GOTO = "goto"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    SEND_MSG_TO_USER = "send_msg_to_user"  # terminal answer only


class ParsedAction(BaseModel):
    """Parsed action with validation (executor-facing)."""
    action_type: ActionType = Field(
        description="Action verb from the ActionType enum."
    )
    element_id: Optional[str] = Field(
        None,
        description="Target element id for CLICK/TYPE/HOVER when applicable (AXTREE id)."
    )
    content: Optional[str] = Field(
        None,
        description="Typed text for TYPE, or the final answer for SEND_MSG_TO_USER."
    )
    url: Optional[str] = Field(
        None,
        description="Destination URL for GOTO."
    )
    key_combination: Optional[str] = Field(
        None,
        description="Key combo for PRESS (e.g., 'Ctrl+v')."
    )
    direction: Optional[str] = Field(
        None,
        description="Scroll direction for SCROLL: 'up' or 'down'."
    )
    tab_index: Optional[int] = Field(
        None,
        description="Tab index for TAB_FOCUS."
    )
    press_enter_after: Optional[bool] = Field(
        None,
        description="Whether to press Enter after TYPE (defaults to True if omitted)."
    )

    @validator('direction')
    def validate_direction(cls, v):
        if v is not None and v not in ['up', 'down']:
            raise ValueError('Direction must be "up" or "down"')
        return v

class ThoughtActionPair(BaseModel):
    """Extracted thought + action for debugging/analytics."""
    thought: str = Field(description="The BLOCK.thought text (≤25 words).")
    action: str = Field(description="Same as BLOCK.action (bracket action string).")
    parsed_action: Optional[ParsedAction] = Field(
        None, description="Executor-ready parsed action object derived from `action`."
    )



class LLMResponse(BaseModel):
    """Full LLM turn with parse/validation status."""
    raw_response: str = Field(
        description="Original raw string returned by the LLM."
    )
    agent_response: Optional[PolicyResponse] = Field(
        None,
        description="Parsed CAB response (CHECKPOINT/AGGREGATE/BLOCK)."
    )
    thought_action: Optional[ThoughtActionPair] = Field(
        None,
        description="Convenience view over BLOCK.thought/action."
    )
    is_valid: bool = Field(
        default=False,
        description="Whether the response conforms to schema and action constraints."
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Schema/semantic validation errors (empty if is_valid=True)."
    )