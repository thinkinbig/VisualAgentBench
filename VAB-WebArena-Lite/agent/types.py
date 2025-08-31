"""
Agent-layer type definitions for staged policy and reward evaluation.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, validator
from enum import Enum
from browser_env.trajectory import Trajectory

# Avoid circular import with llms.types by importing ParsedAction only for type checking
if TYPE_CHECKING:
    from llms.types import ParsedAction  # type: ignore
else:
    ParsedAction = Any  # type: ignore



class CheckpointInfo(BaseModel):
    """Checkpoint information from agent response (logging/memory for policy)."""
    step: int = Field(description="Monotonic turn counter (1-based). Increment every agent turn.")
    url: str = Field(description="Canonical current page URL (normalized by the environment).")
    tab: Dict[str, object] = Field(description="Tab state snapshot, e.g. {'id': int, 'stack': List[str]}.")
    objective: str = Field(description="Echo of the task's OBJECTIVE. Do not rewrite or expand.")
    env_flags: Dict[str, bool] = Field(default_factory=dict, description="Observed environment flags.")
    state_hash: str = Field(description="Opaque fingerprint of visible/interactive state (AXTREE + URL, etc.).")


class AggregateInfo(BaseModel):
    """Aggregate (rolling working memory) from agent response."""
    facts: List[str] = Field(default_factory=list, description="≤8 short KV-like strings for global context.")
    entities: List[str] = Field(default_factory=list, description="≤8 compact handles of processed/target entities.")
    evidence: List[str] = Field(default_factory=list, description="≤8 evidence lines, each referencing an AXTREE id.")
    plan_next_1to3: List[str] = Field(default_factory=list, description="1–3 upcoming steps as action_text strings.")
    risks: List[str] = Field(default_factory=list, description="≤3 short risk tags (e.g., 'captcha', 'auth').")
    stop_condition: str = Field(description="Short predicate for termination (e.g., 'url_contains=success').")


class BlockInfo(BaseModel):
    """Executable decision for this turn. Executors evaluate ONLY this block."""
    thought: str = Field(description="≤25 words explaining why this action advances the goal.")
    action: str = Field(description="One-line bracket action, e.g., 'click [577]' or 'goto [http://…]'.")


class PolicyRequest(BaseModel):
    """Stage One request payload for policy action generation."""
    intent: str = Field(description="Task intent/objective provided to the agent.")
    observation: str = Field(description="AXTREE text of the current page. Include bids and labels.")
    current_url: str = Field(description="Current page URL at decision time.")
    previous_action: str = Field(description="The previous action string, or 'None'.")
    start_url: Optional[str] = Field(None, description="Start URL of the task/episode.")

    @validator('intent', 'observation', 'current_url', 'previous_action', 'start_url', pre=True, always=True)
    def _strip_strings(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip()
        return v


class PolicyResponse(BaseModel):
    """Complete policy agent response structure (CAB)."""
    checkpoint: Optional[CheckpointInfo] = Field(None, alias="CHECKPOINT")
    aggregate: Optional[AggregateInfo] = Field(None, alias="AGGREGATE")
    block: Optional[BlockInfo] = Field(None, alias="BLOCK")

    class Config:
        allow_population_by_field_name = True


class PairwiseDecision(str, Enum):
    """Pairwise comparison decision outcomes."""
    RESPONSE_1 = "response_1"
    RESPONSE_2 = "response_2"
    UNDECIDED = "undecided"


class RewardRequest(BaseModel):
    """Stage Two reward request payload for pairwise reward evaluation."""
    intent: str = Field(description="Task intent/objective.")
    observation: str = Field(description="AXTREE text of the current page.")
    trajectory: str = Field(description="Recent steps as '{THOUGHT: ..., ACTION: ...}' lines.")
    start_url: str = Field(description="Start URL of the session.")
    current_url: str = Field(description="Current URL.")
    thought1: str = Field(description="Candidate 1 THOUGHT.")
    action1: str = Field(description="Candidate 1 ACTION string, e.g., 'click [577]'.")
    thought2: str = Field(description="Candidate 2 THOUGHT.")
    action2: str = Field(description="Candidate 2 ACTION string, e.g., 'click [1749]'.")

    @validator('intent', 'observation', 'trajectory', 'start_url', 'current_url', 'thought1', 'action1', 'thought2', 'action2')
    def _strip_all(cls, v: str) -> str:
        return v.strip()


class RewardResponse(BaseModel):
    """Stage Two reward response payload from pairwise reward evaluation."""
    raw_response: str = Field(description="Original string returned by the reward LLM.")
    decision: PairwiseDecision = Field(description="Parsed decision: response_1, response_2, or undecided.")
    winner: Optional[int] = Field(None, description="1 if Response 1 chosen, 2 if Response 2 chosen; None if undecided.")
    think: Optional[str] = Field(None, description="Extracted <think> content if present.")
    criteria: Optional[str] = Field(None, description="Extracted <Criteria> content if present.")
    analysis: Optional[str] = Field(None, description="Extracted <Analysis> content if present.")
    is_valid: bool = Field(default=False, description="True if a valid <Answer> tag was parsed.")
    parse_errors: List[str] = Field(default_factory=list, description="Parsing issues encountered while extracting fields.")

    @validator('raw_response')
    def _strip_raw(cls, v: str) -> str:
        return v.strip()


class Meta(BaseModel):
    """Minimal runtime metadata bridging Stage 1 (Policy) and Stage 2 (Reward).

    Keep only what RewardRequest construction cannot derive from function args:
    - intent: task intent/objective
    - start_url/current_url: prompt context
    - obs_nodes_info: structured node metadata from browser_env (ids, bounds, text)
    - trajectory: list of recent THOUGHT/ACTION pairs
    """
    intent: Optional[str] = Field(None, description="Task intent/objective.")
    start_url: Optional[str] = Field(None, description="Episode start URL.")
    current_url: Optional[str] = Field(None, description="Current page URL.")
    obs_nodes_info: Optional[Dict[str, Any]] = Field(None, description="Structured node metadata from browser_env (ids, bounds, text).")
    trajectory: Trajectory = Field(default_factory=list, description="Recent THOUGHT/ACTION list.")

    @validator('intent', 'start_url', 'current_url', pre=True, always=True)
    def _strip_optional(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip()
        return v


class PairwiseMatch(BaseModel):
    """One pairwise comparison in the knockout tournament."""
    round_index: int = Field(description="0-based round number in the tournament.")
    index_a: int = Field(description="Index in candidate list for Response 1.")
    index_b: int = Field(description="Index in candidate list for Response 2.")
    reward_request: RewardRequest = Field(description="Constructed input for the reward model.")
    reward_response: RewardResponse = Field(description="Parsed output from the reward model.")


class AgentRuntime(BaseModel):
    """Dynamic state for RewardGuidedAgent across steps.

    - step: monotonic turn counter
    - meta: shared context (intent/urls/observation/trajectory)
    - policy_request: last Stage 1 request
    - policy_candidates: last Stage 1 candidates
    - selected_policy: current chosen candidate after knockout
    - tournament_history: record of pairwise comparisons
    """
    step: int = Field(default=0, description="Monotonic turn counter (1-based preferred externally).")
    meta: Meta = Field(default_factory=Meta, description="Shared runtime context.")
    policy_request: Optional[PolicyRequest] = Field(None, description="Last policy request sent.")
    policy_candidates: List[PolicyResponse] = Field(default_factory=list, description="Stage 1 candidates.")
    selected_policy: Optional[PolicyResponse] = Field(None, description="Winner after knockout.")
    tournament_history: List[PairwiseMatch] = Field(default_factory=list, description="Pairwise comparison records.")
    previous_action: Optional[ParsedAction] = Field(None, description="Most recently executed parsed action.")
    last_checkpoint: Optional[CheckpointInfo] = Field(None, description="Latest CHECKPOINT snapshot parsed from policy output.")
    last_aggregate: Optional[AggregateInfo] = Field(None, description="Latest AGGREGATE working memory parsed from policy output.")


