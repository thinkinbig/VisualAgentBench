"""
Response type definitions for LLM outputs (CAB: Checkpoint / Aggregate / Block)
"""
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class CheckpointInfo(BaseModel):
    """Checkpoint information from agent response (logging/memory for policy; PRM does not use it)."""
    step: int = Field(
        description="Monotonic turn counter (1-based). Increment every agent turn."
    )
    url: str = Field(
        description="Canonical current page URL (normalized by the environment)."
    )
    tab: Dict[str, Any] = Field(
        description="Tab state snapshot, e.g. {'id': int, 'stack': List[str]} for navigation history."
    )
    objective: str = Field(
        description="Echo of the task's OBJECTIVE. Do not rewrite or expand."
    )
    env_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Environment flags observed this turn (e.g., {'login': False, 'captcha': False})."
    )
    state_hash: str = Field(
        description="Opaque fingerprint of the visible/interactive state (AXTREE + URL, etc.), computed by the environment. "
                    "Used for loop detection, caching, and recovery. The LLM may emit a placeholder like '...'."
    )


class AggregateInfo(BaseModel):
    """Aggregate (rolling working memory) from agent response. Aids policy planning; PRM does not evaluate it."""
    facts: List[str] = Field(
        default_factory=list,
        description="≤8 short KV-like strings for global context (e.g., 'users_done=2', 'query=fortnite pc'). Keep stable across turns."
    )
    entities: List[str] = Field(
        default_factory=list,
        description="≤8 compact handles of processed/target entities. Prefer IDs present in AXTREE (e.g., '#577 tacocat_yay')."
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="≤8 evidence lines; EACH must reference an AXTREE id and a brief label, e.g., '#1749 price=$279.49 (visible)'."
    )
    plan_next_1to3: List[str] = Field(
        default_factory=list,
        description="1–3 upcoming steps written as action_text strings (e.g., 'click [577]'). "
                    "plan_next_1to3[0] MUST match the next turn's BLOCK.action_text."
    )
    risks: List[str] = Field(
        default_factory=list,
        description="≤3 short risk tags (e.g., 'captcha', 'auth', 'empty_results', 'ambiguous_target', 'unstable_dom')."
    )
    stop_condition: str = Field(
        description="Short predicate for termination (e.g., 'count_entities=5', 'qa_anchor=price_seen', 'url_contains=success')."
    )


class BlockInfo(BaseModel):
    """Executable decision for this turn. PRM evaluates ONLY this block."""
    thought: str = Field(
        description="≤25 words explaining why this single action advances the goal. No fluff; no repetition."
    )
    action: str = Field(
        description="One-line bracket action the executor runs, e.g., 'click [577]', 'goto [http://…]', 'send_msg_to_user [$279.49]'."
    )



class AgentResponse(BaseModel):
    """Complete agent response structure (CAB)."""
    checkpoint: Optional[CheckpointInfo] = Field(
        None, alias="CHECKPOINT",
        description="Checkpoint snapshot for sliding-window memory and recovery."
    )
    aggregate: Optional[AggregateInfo] = Field(
        None, alias="AGGREGATE",
        description="Rolling aggregation: facts/entities/evidence/short plan/risks/stop_condition."
    )
    block: Optional[BlockInfo] = Field(
        None, alias="BLOCK",
        description="The only part that is executed/evaluated this turn."
    )
    class Config:
        allow_population_by_field_name = True


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
    agent_response: Optional[AgentResponse] = Field(
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
