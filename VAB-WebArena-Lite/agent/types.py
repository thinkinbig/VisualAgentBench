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

# Import TrajectoryTree at runtime to avoid circular imports
try:
    from .trajectory_tree import TrajectoryTree
except ImportError:
    TrajectoryTree = None

# Avoid circular import with llms.types by importing ParsedAction only for type checking
if TYPE_CHECKING:
    from llms.types import ParsedAction  # type: ignore
else:
    ParsedAction = Any  # type: ignore


class NodeStatus(str, Enum):
    """Node status: simplified to two states."""
    CANDIDATE = "candidate"     # Candidate state: node contains candidate actions
    SELECTED = "selected"       # Selected: node has been selected and executed


class ObservationData(BaseModel):
    """Observation data: contains all observation-related information."""
    text: str = Field(description="AXTREE text of the current page. Include bids and labels.")
    nodes_info: Optional[Dict[str, Any]] = Field(
        None,
        description="AXTREE/SoM nodes mapping (ids -> bounds/centers/text) for clickable overlays",
    )
    screenshot_path: Optional[str] = Field(None, description="Path to the screenshot image for this observation.")
    hash_value: Optional[str] = Field(None, description="Computed hash for deduplication and debugging.")
    
    def model_post_init(self, __context) -> None:
        """Automatically compute hash value after creation."""
        if not self.hash_value:
            import hashlib
            try:
                content = self.text or ""
                if self.nodes_info:
                    # Convert nodes_info to string for hash calculation
                    content += str(sorted(self.nodes_info.items()))
                self.hash_value = hashlib.md5(content.encode('utf-8')).hexdigest()
            except Exception:
                self.hash_value = ""
    
    @staticmethod
    def compose_observation_from_nodes(nodes: Optional[Dict[str, Any]]) -> str:
        """Compose observation text from accessibility tree nodes."""
        if not isinstance(nodes, dict) or not nodes:
            return ""
        lines: List[str] = []
        try:
            for _, node in nodes.items():
                t = str(node.get("text", ""))
                if t:
                    lines.append(t)
        except Exception:
            pass
        return "\n".join(lines)


class CheckpointInfo(BaseModel):
    """Checkpoint information from agent response (logging/memory for policy)."""
    step: int = Field(description="Monotonic turn counter (1-based). Increment every agent turn.")
    url: str = Field(description="Canonical current page URL (normalized by the environment).")
    block: Optional["BlockInfo"] = Field(
        None,
        description="The full BLOCK (thought + action) that was executed last turn.",
    )
    objective: str = Field(description="Echo of the task's OBJECTIVE. Do not rewrite or expand.")
    observation: ObservationData = Field(description="Complete observation data including text, nodes, screenshot, and hash.")


class AggregateInfo(BaseModel):
    note: List[str] = Field(default_factory=list, description="notes with key-value pairs; no AX ids.")
    evidence: List[str] = Field(default_factory=list, description="anchors with AX ids, e.g., '#1749 $279.49 (price)'.")
    stuck: bool = Field(default=False, description="Whether the agent is currently stuck and should escape.")
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
    - observation_hash_table: hash table for observation deduplication
    """
    step: int = Field(default=0, description="Monotonic turn counter (1-based preferred externally).")
    meta: Meta = Field(default_factory=Meta, description="Shared runtime context.")
    selected_block: Optional[BlockInfo] = Field(None, description="Winner after knockout.")
    tournament_history: List[PairwiseMatch] = Field(default_factory=list, description="Pairwise comparison records.")
    checkpoint: Optional[CheckpointInfo] = Field(None, description="Latest CHECKPOINT snapshot parsed from policy output.")
    aggregate: Optional[AggregateInfo] = Field(default_factory=AggregateInfo, description="Latest AGGREGATE working memory parsed from policy output.")
    current_round_samples: List[str] = Field(default_factory=list, description="Actions sampled in current round to avoid duplicates.")
    trajectory_tree: Optional["TrajectoryTree"] = Field(default=None, description="Full trajectory tree with root/nodes/edges for this run.")
    observation_hash_table: Dict[str, ObservationData] = Field(default_factory=dict, description="Hash table for observation deduplication and lookup.")


# ========= Trajectory Tree Data Structures =========
class TrajNode(BaseModel):
    """Trajectory tree node base class: common attributes for all nodes (Root and non-Root)."""
    node_id: str = Field(..., description="Unique id within the trajectory tree")
    parent_id: Optional[str] = Field(None, description="Parent node id; None for root")
    step: int = Field(..., description="1-based step index along the EXECUTED main path (root=0)")
    url: Optional[str] = Field(None, description="Current page URL at this node")
    checkpoint: Optional[CheckpointInfo] = Field(None, description="Optional checkpoint snapshot for this node")
    labels: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary tags for filtering/searching")
    status: NodeStatus = Field(default=NodeStatus.CANDIDATE, description="Current status of this node")
    candidates: List[BlockInfo] = Field(default_factory=list, description="Candidate actions available at this node")

    @validator("node_id", "parent_id", "url", pre=True, always=True)
    def _strip_opt(cls, v):
        return v.strip() if isinstance(v, str) else v

    def is_root(self) -> bool:
        """Check if the node is a root node."""
        return self.parent_id is None

    def is_candidate(self) -> bool:
        """Check if the node is in candidate state."""
        return self.status == NodeStatus.CANDIDATE

    def is_selected(self) -> bool:
        """Check if the node has been selected."""
        return self.status == NodeStatus.SELECTED


class TrajRoot(TrajNode):
    """Root node: inherits from TrajNode, adds root-specific metadata."""
    run_id: str = Field(..., description="Unique id for this episode/run")
    intent: str = Field(..., description="Task intent/objective")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional extra metadata (seed, model tags, etc.)")

    @validator("run_id", "intent", pre=True, always=True)
    def _strip_basic(cls, v):
        return v.strip() if isinstance(v, str) else v

    def __init__(self, **data):
        # Ensure root node's parent_id is None
        if 'parent_id' not in data:
            data['parent_id'] = None
        super().__init__(**data)



class TrajEdge(BaseModel):
    """Edge: represents 'how to reach' from parent -> child (action/thought)."""
    edge_id: str = Field(..., description="Unique id for this edge")
    parent_id: str = Field(..., description="From node id")
    child_id: str = Field(..., description="To node id")
    thought: Optional[str] = Field(None, description="Why this action is chosen")
    action: Optional[str] = Field(None, description="Raw action string: e.g., 'click [577]' or 'goto [http://…]'")
    meaning: Optional[str] = Field(None, description="Human-readable action meaning")
    reward: Optional[float] = Field(None, description="Optional local reward if computed")
    notes: Dict[str, Any] = Field(default_factory=dict, description="Extra annotations (e.g., KO round/pair)")

    @validator("edge_id", "parent_id", "child_id", "thought", "action", "meaning", pre=True, always=True)
    def _strip_edge(cls, v):
        return v.strip() if isinstance(v, str) else v


class TrajectoryTreeStats(BaseModel):
    """Statistics about the trajectory tree."""
    total_nodes: int = Field(description="Total number of nodes in the tree")
    total_edges: int = Field(description="Total number of edges in the tree")
    selected_nodes: int = Field(description="Number of selected nodes")
    candidate_nodes: int = Field(description="Number of candidate nodes")
    current_step: int = Field(description="Current step number")
    tree_depth: int = Field(description="Maximum depth of the tree")
    has_root: bool = Field(description="Whether the tree has a root node")
    error: Optional[str] = Field(None, description="Error message if tree is not initialized")


# Rebuild models after all classes are defined to resolve forward references
if TrajectoryTree is not None:
    AgentRuntime.model_rebuild()


