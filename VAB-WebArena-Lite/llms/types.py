"""
LLM-layer types and re-exports.

Note: Policy/Reward request-response schemas live in agent/types.py.
This module re-exports them to avoid breaking existing imports during migration.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
from agent.types import (
    PlanResponse,
)

class ActionType(str, Enum):
    """Valid action types (exactly one per turn)."""
    CLICK = "click"
    TYPE = "type"
    HOVER = "hover"
    PRESS = "press"
    SCROLL = "scroll"
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
    press_enter_after: Optional[bool] = Field(
        None,
        description="Whether to press Enter after TYPE (defaults to True if omitted)."
    )
    page_number: Optional[int] = Field(
        None,
        description="Target tab index for TAB_FOCUS (0-based or env-specific)."
    )

    @validator('direction')
    def validate_direction(cls, v):
        if v is not None and v not in ['up', 'down']:
            raise ValueError('Direction must be "up" or "down"')
        return v

    def to_id_action_str(self) -> str:
        """Render a canonical id_accessibility_tree bracket action string.

        Examples:
        - click [577]
        - type [312] [hello] [1]
        - hover [1749]
        - press [Ctrl+v]
        - scroll [down]
        - goto [https://example.com]
        - tab_focus [1]
        - close_tab
        - new_tab
        - go_back
        - go_forward
        - send_msg_to_user [final answer]
        """
        at = self.action_type

        if at == ActionType.CLICK:
            if not self.element_id:
                raise ValueError("CLICK requires element_id for id_accessibility_tree format")
            return f"click [{self.element_id}]"

        if at == ActionType.TYPE:
            if not self.element_id:
                raise ValueError("TYPE requires element_id for id_accessibility_tree format")
            if self.content is None:
                raise ValueError("TYPE requires content (text)")
            press_enter = self.press_enter_after if self.press_enter_after is not None else True
            enter_flag = 1 if press_enter else 0
            return f"type [{self.element_id}] [{self.content}] [{enter_flag}]"

        if at == ActionType.HOVER:
            if not self.element_id:
                raise ValueError("HOVER requires element_id for id_accessibility_tree format")
            return f"hover [{self.element_id}]"

        if at == ActionType.PRESS:
            if not self.key_combination:
                raise ValueError("PRESS requires key_combination, e.g., 'Ctrl+v'")
            return f"press [{self.key_combination}]"

        if at == ActionType.SCROLL:
            if not self.direction:
                raise ValueError("SCROLL requires direction: 'up' or 'down'")
            return f"scroll [{self.direction}]"

        if at == ActionType.GOTO:
            if not self.url:
                raise ValueError("GOTO requires url")
            return f"goto [{self.url}]"

        if at == ActionType.TAB_FOCUS:
            if self.page_number is None:
                raise ValueError("TAB_FOCUS requires page_number")
            return f"tab_focus [{self.page_number}]"

        if at == ActionType.CLOSE_TAB:
            return "close_tab"

        if at == ActionType.NEW_TAB:
            return "new_tab"

        if at == ActionType.GO_BACK:
            return "go_back"

        if at == ActionType.GO_FORWARD:
            return "go_forward"

        if at == ActionType.SEND_MSG_TO_USER:
            # Some tasks allow empty answer
            return (
                f"send_msg_to_user [{self.content}]" if self.content else "send_msg_to_user"
            )

        # Fallback: make it explicit for easier debugging
        raise ValueError(f"Unsupported ActionType for id_accessibility_tree: {at}")

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
    agent_response: Optional[PlanResponse] = Field(
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