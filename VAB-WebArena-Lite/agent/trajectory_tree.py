from typing import List, Optional, Dict, Any
import json
import base64
import mimetypes
import textwrap
import time
import random
from pathlib import Path
from abc import abstractmethod
from pydantic import BaseModel, Field
from enum import Enum


class CandidateNodeStatus(str, Enum):
    CANDIDATE = "candidate"
    SELECTED = "selected"


class TrajNode(BaseModel):
    """Base node: common fields for all nodes."""
    node_id: str = Field(..., description="Unique id within the trajectory tree")
    parent_id: Optional[str] = Field(None, description="Parent node id; None for root")
    url: Optional[str] = Field(None, description="Current page URL at this node")

    def is_root(self) -> bool:
        """Return True if this node is the root."""
        return self.parent_id is None

    @abstractmethod
    def is_state(self) -> bool:
        """Return True if this node is a state node."""
        pass

    @abstractmethod
    def is_candidate(self) -> bool:
        """Return True if this node is a candidate node (unselected)."""
        pass

    @abstractmethod
    def is_selected(self) -> bool:
        """Return True if this node is a selected candidate node."""
        pass


class TrajState(TrajNode):
    """State node: represents the browser state after executing an action."""
    # Note: We allow step to start from 0 (initial state). First state is determined by the smallest step.
    step: int = Field(..., description="Step index along the EXECUTED path (root can be step=0)")
    observation_hash: Optional[str] = Field(None, description="Fingerprint for dedup/debug")
    obs_nodes_info: Optional[Dict[str, Any]] = Field(
        None,
        description="AXTREE/SoM mapping (ids -> bounds/centers/text) for clickable overlays",
    )
    screenshot_path: Optional[str] = Field(None, description="Filesystem path to the screenshot image for this state")
    candidates: List[str] = Field(default_factory=list, description="Child node IDs representing candidate actions")

    def is_state(self) -> bool:
        return True

    def is_candidate(self) -> bool:
        return False

    def is_selected(self) -> bool:
        return False


class TrajCandidate(TrajNode):
    """Candidate node: represents a possible action from a state."""
    thought: Optional[str] = Field(None, description="Reasoning for choosing this action")
    action: Optional[str] = Field(None, description="Raw action string, e.g., 'click [577]' or 'goto [http://…]'")
    meaning: Optional[str] = Field(None, description="Human-readable action meaning")
    status: CandidateNodeStatus = Field(default=CandidateNodeStatus.CANDIDATE, description="Current status of this candidate")

    def is_state(self) -> bool:
        return False

    def is_candidate(self) -> bool:
        return self.status == CandidateNodeStatus.CANDIDATE

    def is_selected(self) -> bool:
        return self.status == CandidateNodeStatus.SELECTED


class TrajRoot(TrajNode):
    """Root node: run metadata and intent."""
    run_id: str = Field(default="", description="Unique run identifier")
    intent: str = Field(default="", description="Task intent/objective")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    screenshot_path: Optional[str] = Field(None, description="Filesystem path to the screenshot image for the root state")

    def is_state(self) -> bool:
        return False

    def is_candidate(self) -> bool:
        return False

    def is_selected(self) -> bool:
        return False


class TrajectoryTree:
    """Complete trajectory tree: one root + multiple nodes with parent-child relationships."""

    def __init__(self, root: TrajRoot):
        self.root = root
        self.nodes: List[TrajNode] = [root]  # Include root in nodes list
        
        # Ensure root has a unique run_id
        if not self.root.run_id or self.root.run_id == "default_run":
            self.root.run_id = self._generate_run_id()

    def _generate_run_id(self) -> str:
        """Generate a unique run_id with timestamp and random component."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        random_suffix = random.randint(1000, 9999)
        return f"run_{timestamp}_{random_suffix}"

    def get_run_id(self) -> str:
        """Get the run_id from the root node."""
        return self.root.run_id

    def set_run_id(self, run_id: str) -> None:
        """Set the run_id in the root node."""
        self.root.run_id = run_id

    # -------------------- Serialization --------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"nodes": [node.model_dump() for node in self.nodes]}

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "TrajectoryTree":
        """Create TrajectoryTree from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryTree":
        """Create TrajectoryTree from a dict."""
        nodes: List[TrajNode] = []
        root: Optional[TrajRoot] = None

        for node_data in data.get("nodes", []):
            if node_data.get("parent_id") is None:
                # Root
                root = TrajRoot(
                    node_id=node_data["node_id"],
                    parent_id=node_data.get("parent_id"),
                    url=node_data.get("url"),
                    run_id=node_data.get("run_id", ""),
                    intent=node_data.get("intent", ""),
                    meta=node_data.get("meta", {}),
                    screenshot_path=node_data.get("screenshot_path"),
                )
            elif "action" in node_data and node_data.get("action") is not None:
                # Candidate
                node = TrajCandidate(
                    node_id=node_data["node_id"],
                    parent_id=node_data.get("parent_id"),
                    url=node_data.get("url"),
                    thought=node_data.get("thought"),
                    action=node_data.get("action"),
                    meaning=node_data.get("meaning"),
                    status=CandidateNodeStatus(node_data.get("status", "candidate")),
                )
                nodes.append(node)
            elif "step" in node_data:
                # State
                node = TrajState(
                    node_id=node_data["node_id"],
                    parent_id=node_data.get("parent_id"),
                    step=node_data["step"],
                    url=node_data.get("url"),
                    observation_hash=node_data.get("observation_hash"),
                    obs_nodes_info=node_data.get("obs_nodes_info"),
                    screenshot_path=node_data.get("screenshot_path"),
                    candidates=node_data.get("candidates", []),
                )
                nodes.append(node)
            else:
                raise ValueError(f"Unknown node type for node_id: {node_data.get('node_id', 'unknown')}")

        if root is None:
            raise ValueError("No root node found in JSON data")

        tree = cls(root)
        for node in nodes:
            tree.nodes.append(node)
        return tree

    # -------------------- Lookup helpers --------------------

    def get_node(self, node_id: str) -> Optional[TrajNode]:
        """Return the node with the given id, or None if not found."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def is_state_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        return node is not None and node.is_state()

    def is_candidate_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        return node is not None and node.is_candidate()

    def is_root_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        return node is not None and node.is_root()

    def is_selected_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        return node is not None and node.is_selected()

    def get_state_nodes(self) -> List[TrajState]:
        return [node for node in self.nodes if node.is_state()]

    def get_candidate_nodes(self) -> List[TrajCandidate]:
        # Only return candidates that are currently in CANDIDATE status
        return [node for node in self.nodes if isinstance(node, TrajCandidate) and node.is_candidate()]

    def get_selected_nodes(self) -> List[TrajCandidate]:
        return [node for node in self.nodes if isinstance(node, TrajCandidate) and node.is_selected()]

    # -------------------- File URI utility (Plan A) --------------------

    def _as_file_uri(self, p: str) -> str:
        """Return a canonical file URI: file:///... with URL encoding (spaces, unicode, etc.)."""
        try:
            return Path(p).resolve().as_uri()
        except Exception:
            # Fallback: coarse join; not guaranteed to be encoded
            return "file://" + str(Path(p))

    # -------------------- Graphviz visualization --------------------

    def _build_graphviz(self, name: str = "trajectory", directory: Optional[str] = None):
        """Build a Graphviz Digraph once; reused by to_graphviz/save_graphviz."""
        from graphviz import Digraph

        G = Digraph(name=name, filename=name, directory=directory)
        G.attr(rankdir="TB")
        G.attr("node", shape="box", style="filled")

        # Root node
        root_label = f"ROOT\nTask: {self.root.intent or 'Unknown'}"
        if self.root.url:
            root_label += f"\nURL: {self.root.url[:50]}..."
        if getattr(self.root, "screenshot_path", None):
            uri = self._as_file_uri(self.root.screenshot_path)
            # Set both URL and href to maximize clickability across backends (pdf/svg)
            G.node(self.root.node_id, root_label, URL=uri, href=uri)
        else:
            G.node(self.root.node_id, root_label, fillcolor="lightblue")

        # Other nodes
        for node in self.nodes:
            if node.is_root():
                continue
            if node.is_state():
                self._add_state_node_to_graphviz(G, node)  # type: ignore[arg-type]
            elif isinstance(node, TrajCandidate):
                self._add_candidate_node_to_graphviz(G, node)

        # Edges
        self._add_edges_to_graphviz_pure(G)

        return G

    def to_graphviz(self, filename: str = "trajectory") -> str:
        """Return DOT source (no rendering)."""
        G = self._build_graphviz(name=filename, directory=None)
        return G.source

    def save_graphviz(self, filename: str = "trajectory", output_dir: str = ".", fmt: str = "pdf") -> str:
        """Render and save the graph; return the output filepath."""
        G = self._build_graphviz(name=filename, directory=output_dir)
        output_path = G.render(format=fmt, cleanup=True)
        return output_path

    def _add_state_node_to_graphviz(self, G, node: TrajState):
        """Add a state node (use canonical file:/// URI when screenshot exists)."""
        label = f"State {node.step}"
        if node.screenshot_path:
            uri = self._as_file_uri(node.screenshot_path)
            G.node(node.node_id, label, URL=uri, href=uri, fillcolor="lightgreen")
        else:
            G.node(node.node_id, label, fillcolor="lightgreen")

    def _add_candidate_node_to_graphviz(self, G, node: TrajCandidate):
        """Add a candidate node with styles reflecting selection status."""
        if node.is_selected():
            fillcolor = "lightgreen"
            style = "filled,bold"
        else:
            fillcolor = "lightyellow"
            style = "filled,dashed"

        candidate_index = self._get_candidate_display_index(node)

        # Build label with thought, meaning, and action
        label_parts = [f"Candidate {candidate_index}"]
        
        # Add thought if available
        if node.thought and node.thought.strip():
            # Truncate thought if too long for display
            thought_text = node.thought.strip()
            if len(thought_text) > 100:
                thought_text = thought_text[:97] + "..."
            label_parts.append(f"Thought: {thought_text}")
        
        # Add meaning if available
        if node.meaning and node.meaning.strip():
            label_parts.append(f"Meaning: {node.meaning}")
        
        # Add action if available
        if node.action and node.action.strip():
            label_parts.append(f"Action: {node.action}")
        
        # If no meaningful content, show unknown
        if len(label_parts) == 1:
            label_parts.append("Unknown action")
        
        label = "\n".join(label_parts)

        G.node(node.node_id, label, fillcolor=fillcolor, style=style)

    def _add_edges_to_graphviz_pure(self, G):
        """Add edges; compute first_state once to avoid redundant scans."""
        first_state = self._get_first_state()
        if first_state:
            G.edge(self.root.node_id, first_state.node_id, label="start", style="bold")

        for node in self.nodes:
            if not node.is_state():
                continue

            # state -> candidate(s)
            for candidate_id in getattr(node, "candidates", []):
                candidate = self.get_node(candidate_id)
                if candidate and isinstance(candidate, TrajCandidate):
                    if candidate.is_selected():
                        G.edge(node.node_id, candidate_id, label="selected", style="bold", color="green")
                    else:
                        G.edge(node.node_id, candidate_id, label="candidate", style="dashed", color="gray")

            # selected candidate(s) -> next state
            selected_candidates = self._get_selected_candidates_for_state(node)  # type: ignore[arg-type]
            if selected_candidates:
                next_state = self._get_next_state(node.step)  # type: ignore[arg-type]
                if next_state:
                    for selected_candidate in selected_candidates:
                        G.edge(selected_candidate.node_id, next_state.node_id, label="execute", style="bold", color="blue")

    def _get_first_state(self) -> Optional[TrajState]:
        """Return the state with the smallest step (supports step starting from 0 or 1)."""
        states = [n for n in self.nodes if n.is_state()]
        if not states:
            return None
        return min(states, key=lambda s: s.step)  # type: ignore[arg-type]

    def _get_selected_candidate_for_state(self, state: TrajState) -> Optional[TrajCandidate]:
        """Return the single selected candidate under this state, if any."""
        for candidate_id in getattr(state, "candidates", []):
            candidate = self.get_node(candidate_id)
            if candidate and isinstance(candidate, TrajCandidate) and candidate.is_selected():
                return candidate
        return None

    def _get_selected_candidates_for_state(self, state: TrajState) -> List[TrajCandidate]:
        """Return all selected candidates under this state."""
        selected_candidates: List[TrajCandidate] = []
        for candidate_id in getattr(state, "candidates", []):
            candidate = self.get_node(candidate_id)
            if candidate and isinstance(candidate, TrajCandidate) and candidate.is_selected():
                selected_candidates.append(candidate)
        return selected_candidates

    def _get_next_state(self, current_step: int) -> Optional[TrajState]:
        """Return the state whose step is current_step + 1, if present."""
        for node in self.nodes:
            if node.is_state() and node.step == current_step + 1:
                return node  # type: ignore[return-value]
        return None

    # -------------------- Runtime tree operations --------------------

    def add_node(self, node: TrajNode) -> None:
        self.nodes.append(node)

    def add_state_node(
        self,
        node_id: str,
        parent_id: str,
        step: int,
        url: Optional[str] = None,
        observation_hash: Optional[str] = None,
        obs_nodes_info: Optional[Dict[str, Any]] = None,
        screenshot_path: Optional[str] = None,
    ) -> TrajState:
        state_node = TrajState(
            node_id=node_id,
            parent_id=parent_id,
            step=step,
            url=url,
            observation_hash=observation_hash,
            obs_nodes_info=obs_nodes_info,
            screenshot_path=screenshot_path,
            candidates=[],
        )
        self.add_node(state_node)
        return state_node

    def add_candidate_node(
        self,
        node_id: str,
        parent_id: str,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        meaning: Optional[str] = None,
        status: CandidateNodeStatus = CandidateNodeStatus.CANDIDATE,
    ) -> TrajCandidate:
        candidate_node = TrajCandidate(
            node_id=node_id,
            parent_id=parent_id,
            thought=thought,
            action=action,
            meaning=meaning,
            status=status,
        )
        self.add_node(candidate_node)
        return candidate_node

    def add_candidate_to_state(self, state_node_id: str, candidate_node_id: str) -> None:
        state_node = self.get_node(state_node_id)
        if state_node and state_node.is_state():  # type: ignore[truthy-function]
            if candidate_node_id not in getattr(state_node, "candidates", []):
                state_node.candidates.append(candidate_node_id)  # type: ignore[union-attr]

    def mark_candidate_as_selected(self, candidate_node_id: str) -> None:
        """Mark a candidate node as SELECTED (idempotent)."""
        candidate_node = self.get_node(candidate_node_id)
        if candidate_node and isinstance(candidate_node, TrajCandidate):
            candidate_node.status = CandidateNodeStatus.SELECTED

    def set_selected_candidate(self, parent_state_id: str, candidate_node_id: str) -> None:
        """Ensure there is at most one SELECTED candidate under a state."""
        state = self.get_node(parent_state_id)
        if state and state.is_state():  # type: ignore[truthy-function]
            for cid in getattr(state, "candidates", []):
                cand = self.get_node(cid)
                if isinstance(cand, TrajCandidate):
                    cand.status = CandidateNodeStatus.CANDIDATE
            self.mark_candidate_as_selected(candidate_node_id)

    def _get_candidate_display_index(self, node: TrajCandidate) -> int:
        """Display index strategy: prefer numeric suffix in node_id (candidate_#), otherwise by encounter order."""
        try:
            import re
            m = re.search(r"candidate_(\d+)$", node.node_id)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        all_candidates: List[TrajCandidate] = [n for n in self.nodes if isinstance(n, TrajCandidate)]
        for i, cand in enumerate(all_candidates):
            if cand.node_id == node.node_id:
                return i + 1
        return 0
