"""
Agent-layer type definitions for staged policy and reward evaluation.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, validator
from enum import Enum
import re
if TYPE_CHECKING:
    from browser_env.trajectory import Trajectory
    from .trajectory_tree import TrajectoryTree
else:
    Trajectory = list  # type: ignore[assignment]
    TrajectoryTree = Any  # type: ignore[assignment]


class CheckpointInfo(BaseModel):
    """Checkpoint information from agent response (logging/memory for policy)."""
    step: int = Field(description="Monotonic turn counter (1-based). Increment every agent turn.")
    url: str = Field(description="Canonical current page URL (normalized by the environment).")
    block: Optional["BlockInfo"] = Field(
        None,
        description="The full BLOCK (thought + action) that was executed last turn.",
    )
    objective: str = Field(description="Echo of the task's OBJECTIVE. Do not rewrite or expand.")
    observation: Dict[str, Any] = Field(description="Structured observation with keys: text, nodes_info, url.")
    screenshot_path: Optional[str] = Field(None, description="Path to the screenshot image for this checkpoint.")

class BlockInfo(BaseModel):
    """Executable decision for this turn. Executors evaluate ONLY this block."""
    thought: str = Field(description="Why this action advances the goal.")
    action: str = Field(description="Action text WITHOUT backticks, e.g., 'click [577]' or 'goto [http://…]'.")


class PolicyRequest(BaseModel):
    """Stage One request payload for policy action generation."""
    intent: str = Field(description="Task intent/objective provided to the agent.")
    observation: str = Field(description="AXTREE text of the current page. Include bids and labels.")
    current_url: Optional[str] = Field(None, description="Current page URL at decision time.")

    @validator('intent', 'observation', 'current_url', pre=True, always=True)
    def _strip_strings(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip()
        return v


class PolicyResponse(BaseModel):
    """Stage One response payload from policy action generation."""
    candidates: List[BlockInfo] = Field(default_factory=list, description="Generated candidate actions.")
    total_generated: int = Field(default=0, description="Total candidates generated before filtering.")
    unique_actions: int = Field(default=0, description="Number of unique actions after deduplication.")
    is_valid: bool = Field(default=False, description="True if policy generation was successful.")



class PairwiseDecision(str, Enum):
    """Pairwise comparison decision outcomes."""
    RESPONSE_1 = "response_1"
    RESPONSE_2 = "response_2"



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
    is_valid: bool = Field(default=False, description="True if a valid <Answer> tag was parsed.")

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
    - selected_policy: current chosen candidate after knockout
    - tournament_history: record of pairwise comparisons
    - current_round_samples: actions sampled in current round to avoid duplicates
    """
    step: int = Field(default=0, description="Monotonic turn counter (1-based preferred externally).")
    meta: Meta = Field(default_factory=Meta, description="Shared runtime context.")
    selected_block: Optional[BlockInfo] = Field(None, description="Winner after knockout.")
    tournament_history: List[PairwiseMatch] = Field(default_factory=list, description="Pairwise comparison records.")
    checkpoint: Optional[CheckpointInfo] = Field(None, description="Latest CHECKPOINT snapshot parsed from policy output.")
    current_round_samples: List[str] = Field(default_factory=list, description="Actions sampled in current round to avoid duplicates.")
    trajectory_tree: Optional[TrajectoryTree] = Field(default=None, description="Full trajectory tree with root/nodes/edges for this run.")
