"""
Agent-layer type definitions for staged policy and reward evaluation.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from browser_env.trajectory import Trajectory
else:
    Trajectory = list  # type: ignore[assignment]

# Avoid circular import with llms.types by importing ParsedAction only for type checking
if TYPE_CHECKING:
    from llms.types import ParsedAction  # type: ignore
else:
    ParsedAction = Any  # type: ignore



class CheckpointInfo(BaseModel):
    """Checkpoint information from agent response (logging/memory for policy)."""
    step: int = Field(description="Monotonic turn counter (1-based). Increment every agent turn.")
    url: str = Field(description="Canonical current page URL (normalized by the environment).")
    action: Optional[str] = Field(None, description="The bracket action string the agent just executed, or 'None'.")
    objective: str = Field(description="Echo of the task's OBJECTIVE. Do not rewrite or expand.")
    observation: str = Field(description="AXTREE text of the current page. Include bids and labels.")


class AggregateInfo(BaseModel):
    note: List[str] = Field(default_factory=list, description="notes with key-value pairs; no AX ids.")
    evidence: List[str] = Field(default_factory=list, description="anchors with AX ids, e.g., '#1749 $279.49 (price)'.")
    plan_next: str = Field(default="", description="Intent of next action in text form.")
    answer_ready: bool = Field(default=False, description="Whether the answer is ready.")


class BlockInfo(BaseModel):
    """Executable decision for this turn. Executors evaluate ONLY this block."""
    thought: str = Field(description="Why this action advances the goal.")
    action: str = Field(description="Action text WITHOUT backticks, e.g., 'click [577]' or 'goto [http://…]'.")


class PolicyRequest(BaseModel):
    """Stage One request payload for policy action generation."""
    intent: str = Field(description="Task intent/objective provided to the agent.")
    observation: str = Field(description="AXTREE text of the current page. Include bids and labels.")
    current_url: Optional[str] = Field(None, description="Current page URL at decision time.")
    action: Optional[str] = Field(None, description="The action string, or 'None'.")
    start_url: str = Field(description="Start URL of the task/episode.")

    @validator('intent', 'observation', 'action', 'start_url', pre=True, always=True)
    def _strip_strings(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip()
        return v


class PlanRequest(BaseModel):
    aggregate: Optional[AggregateInfo] = Field(default=None, alias="AGGREGATE")
    observation: str
    action: Optional[ParsedAction] = Field(default=None)

class PlanResponse(BaseModel):
    """Complete policy agent response structure (CAB)."""
    checkpoint: Optional[CheckpointInfo] = Field(None, alias="CHECKPOINT")
    aggregate: Optional[AggregateInfo] = Field(None, alias="AGGREGATE")

    class Config:
        population_by_field_name = True
        populate_by_name = True



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
    - block_candidates: last Stage 1 candidates
    - selected_policy: current chosen candidate after knockout
    - tournament_history: record of pairwise comparisons
    """
    step: int = Field(default=0, description="Monotonic turn counter (1-based preferred externally).")
    meta: Meta = Field(default_factory=Meta, description="Shared runtime context.")
    block_candidates: List[BlockInfo] = Field(default_factory=list, description="Stage 1 candidates.")
    selected_block: Optional[BlockInfo] = Field(None, description="Winner after knockout.")
    tournament_history: List[PairwiseMatch] = Field(default_factory=list, description="Pairwise comparison records.")
    checkpoint: Optional[CheckpointInfo] = Field(None, description="Latest CHECKPOINT snapshot parsed from policy output.")
    aggregate: Optional[AggregateInfo] = Field(default_factory=AggregateInfo, description="Latest AGGREGATE working memory parsed from policy output.")


