from typing import List, Optional, Dict, Any
import json
import base64
import mimetypes
import textwrap
from pathlib import Path
from abc import abstractmethod
from pydantic import BaseModel, Field
from enum import Enum


class CandidateNodeStatus(str, Enum):
    CANDIDATE = "candidate"
    SELECTED = "selected"


class TrajNode(BaseModel):
    """基础节点：包含所有节点的通用字段。"""
    node_id: str = Field(..., description="Unique id within the trajectory tree")
    parent_id: Optional[str] = Field(None, description="Parent node id; None for root")
    url: Optional[str] = Field(None, description="Current page URL at this node")
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent_id is None
    
    @abstractmethod
    def is_state(self) -> bool:
        """Check if this is a state node."""
        pass
    
    @abstractmethod
    def is_candidate(self) -> bool:
        """Check if this is a candidate node."""
        pass

    @abstractmethod
    def is_selected(self) -> bool:
        """Check if this is a selected node."""
        pass

class TrajState(TrajNode):
    """状态节点：代表执行动作后的浏览器状态。"""
    step: int = Field(..., description="1-based step index along the EXECUTED main path (root=0)")
    observation_hash: Optional[str] = Field(None, description="Fingerprint of AXTREE/screenshot for dedup/debug")
    obs_nodes_info: Optional[Dict[str, Any]] = Field(
        None,
        description="AXTREE/SoM nodes mapping (ids -> bounds/centers/text) for clickable overlays",
    )
    screenshot_path: Optional[str] = Field(None, description="Filesystem path to the screenshot image for this state")
    candidates: List[str] = Field(default_factory=list, description="Child node IDs representing candidate actions")

    def is_state(self) -> bool:
        """State node is always a state."""
        return True

    def is_candidate(self) -> bool:
        """State node is never a candidate."""
        return False

    def is_selected(self) -> bool:
        """State node is never a selected node."""
        return False

class TrajCandidate(TrajNode):
    """候选节点：代表可选的候选动作。"""
    # Action information stored directly in the node
    thought: Optional[str] = Field(None, description="Why this action was chosen")
    action: Optional[str] = Field(None, description="Raw action string: e.g., 'click [577]' or 'goto [http://…]'")
    meaning: Optional[str] = Field(None, description="Human-readable action meaning")
    status: CandidateNodeStatus = Field(default=CandidateNodeStatus.CANDIDATE, description="Current status of this candidate")
    
    def is_state(self) -> bool:
        """State node is never a state."""
        return False

    def is_candidate(self) -> bool:
        """Candidate node is always a candidate."""
        return self.status == CandidateNodeStatus.CANDIDATE


    def is_selected(self) -> bool:
        """Candidate node is always a candidate."""
        return self.status == CandidateNodeStatus.SELECTED


class TrajRoot(TrajNode):
    """根节点：包含任务意图和元数据。"""
    step: int = Field(default=0, description="Root node step (always 0)")
    run_id: str = Field(default="", description="Unique run identifier")
    intent: str = Field(default="", description="Task intent/objective")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    screenshot_path: Optional[str] = Field(None, description="Filesystem path to the screenshot image for the root state")


    def is_state(self) -> bool:
        """Root node is never a state."""
        return False

    def is_candidate(self) -> bool:
        """Root node is never a candidate."""
        return False

    def is_selected(self) -> bool:
        """Root node is never a selected node."""
        return False

class TrajectoryTree:
    """Complete trajectory tree: one root + multiple nodes with parent-child relationships."""
    
    def __init__(self, root: TrajRoot):
        self.root = root
        self.nodes: List[TrajNode] = [root]  # Include root in nodes list
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "nodes": [node.model_dump() for node in self.nodes]
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrajectoryTree":
        """Create TrajectoryTree from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryTree":
        """Create TrajectoryTree from dictionary data."""
        nodes = []
        root = None
        
        for node_data in data.get("nodes", []):
            # Determine node type based on available fields
            if node_data.get("parent_id") is None:
                # This is the root node
                root = TrajRoot(
                    node_id=node_data["node_id"],
                    parent_id=node_data.get("parent_id"),
                    step=node_data.get("step", 0),
                    url=node_data.get("url"),
                    run_id=node_data.get("run_id", ""),
                    intent=node_data.get("intent", ""),
                    meta=node_data.get("meta", {}),
                    screenshot_path=node_data.get("screenshot_path")
                )
            elif "action" in node_data and node_data["action"] is not None:
                # This is a candidate node
                node = TrajCandidate(
                    node_id=node_data["node_id"],
                    parent_id=node_data.get("parent_id"),
                    url=node_data.get("url"),
                    thought=node_data.get("thought"),
                    action=node_data.get("action"),
                    meaning=node_data.get("meaning"),
                    status=CandidateNodeStatus(node_data.get("status", "candidate"))
                )
                nodes.append(node)
            elif "step" in node_data:
                # This is a state node (has step field)
                node = TrajState(
                    node_id=node_data["node_id"],
                    parent_id=node_data.get("parent_id"),
                    step=node_data["step"],
                    url=node_data.get("url"),
                    observation_hash=node_data.get("observation_hash"),
                    obs_nodes_info=node_data.get("obs_nodes_info"),
                    screenshot_path=node_data.get("screenshot_path"),
                    candidates=node_data.get("candidates", [])
                )
                nodes.append(node)
            else:
                # Fallback: try to create a basic node
                raise ValueError(f"Unknown node type for node_id: {node_data.get('node_id', 'unknown')}")
        
        if root is None:
            raise ValueError("No root node found in JSON data")
        
        # Create trajectory tree
        tree = cls(root)
        
        # Add non-root nodes
        for node in nodes:
            tree.nodes.append(node)
        
        return tree
    
    # ---- Node type checking methods ----
    
    def get_node(self, node_id: str) -> Optional[TrajNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def is_state_node(self, node_id: str) -> bool:
        """Check if a node is a state node."""
        node = self.get_node(node_id)
        return node is not None and node.is_state()
    
    def is_candidate_node(self, node_id: str) -> bool:
        """Check if a node is a candidate node."""
        node = self.get_node(node_id)
        return node is not None and node.is_candidate()
    
    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is the root node."""
        node = self.get_node(node_id)
        return node is not None and node.is_root()
    
    def is_selected_node(self, node_id: str) -> bool:
        """Check if a node is a selected node."""
        node = self.get_node(node_id)
        return node is not None and node.is_selected()
    
    def get_state_nodes(self) -> List[TrajState]:
        """Get all state nodes."""
        return [node for node in self.nodes if node.is_state()]
    
    def get_candidate_nodes(self) -> List[TrajCandidate]:
        """Get all candidate nodes."""
        return [node for node in self.nodes if node.is_candidate()]
    
    def get_selected_nodes(self) -> List[TrajCandidate]:
        """Get all selected candidate nodes."""
        return [node for node in self.nodes if node.is_selected()]
    
    # ---- Graphviz visualization methods ----
    
    def to_graphviz(self, filename: str = "trajectory") -> str:
        """Generate Graphviz DOT format trajectory graph."""
        from graphviz import Digraph
        
        # Create Digraph
        G = Digraph(filename, filename)
        G.attr(rankdir="TB")
        G.attr("node", shape="box", style="filled")
        
        # Add root node
        root_label = f"ROOT\nTask: {self.root.intent or 'Unknown'}\nRun ID: {self.root.run_id}"
        if self.root.url:
            root_label += f"\nURL: {self.root.url[:50]}..."
        
        # Add screenshot if available
        if hasattr(self.root, 'screenshot_path') and self.root.screenshot_path:
            G.node(self.root.node_id, root_label, URL=self.root.screenshot_path)
        else:
            G.node(self.root.node_id, root_label, fillcolor="lightblue")
        
        # Add all other nodes
        for node in self.nodes:
            if node.is_root():
                continue  # Skip root, already added
            
            if node.is_state():
                self._add_state_node_to_graphviz(G, node)
            elif node.is_candidate():
                self._add_candidate_node_to_graphviz(G, node)
        
        # Add edges
        self._add_edges_to_graphviz(G)
        
        return G.source
    
    def save_graphviz(self, filename: str = "trajectory", output_dir: str = ".") -> str:
        """Save Graphviz visualization to file."""
        from graphviz import Digraph
        
        # Create Digraph
        G = Digraph(filename, filename, directory=output_dir)
        G.attr(rankdir="TB")
        G.attr("node", shape="box", style="filled")
        
        # Add root node
        root_label = f"ROOT\nTask: {self.root.intent or 'Unknown'}\nRun ID: {self.root.run_id}"
        if self.root.url:
            root_label += f"\nURL: {self.root.url[:50]}..."
        
        # Add screenshot if available
        if hasattr(self.root, 'screenshot_path') and self.root.screenshot_path:
            G.node(self.root.node_id, root_label, URL=self.root.screenshot_path)
        else:
            G.node(self.root.node_id, root_label, fillcolor="lightblue")
        
        # Add all other nodes
        for node in self.nodes:
            if node.is_root():
                continue  # Skip root, already added
            
            if node.is_state():
                self._add_state_node_to_graphviz(G, node)
            elif node.is_candidate():
                self._add_candidate_node_to_graphviz(G, node)
        
        # Add edges
        self._add_edges_to_graphviz(G)
        
        # Render the graph
        output_path = G.render(format="pdf", cleanup=True)
        return output_path
    
    def _add_state_node_to_graphviz(self, G, node: TrajState):
        """Add a state node to Graphviz graph."""
        label = f"State {node.step}"
        if node.url:
            label += f"\nURL: {node.url[:50]}..."
        
        # Add observation info if available
        if node.observation_hash:
            label += f"\nHash: {node.observation_hash[:8]}..."
        
        # Add screenshot if available
        if node.screenshot_path:
            G.node(node.node_id, label, URL=node.screenshot_path, fillcolor="lightgreen")
        else:
            G.node(node.node_id, label, fillcolor="lightgreen")
    
    def _add_candidate_node_to_graphviz(self, G, node: TrajCandidate):
        """Add a candidate node to Graphviz graph."""
        # Determine color based on status
        if node.is_selected():
            fillcolor = "lightgreen"  # Selected candidates - green
            style = "filled,bold"
        else:
            fillcolor = "lightyellow"  # Regular candidates - yellow
            style = "filled,dashed"
        
        # Create simplified node ID
        candidate_index = self._get_candidate_index(node)
        simple_id = f"candidate_{candidate_index}"
        
        # Create label with action and meaning
        label = f"Candidate {candidate_index}\nAction: {node.action or 'Unknown'}"
        
        # Add meaning if available
        if node.meaning:
            label += f"\nMeaning: {node.meaning}"
        elif node.action:
            # Try to extract meaning from action
            meaning = self._extract_action_meaning(node.action)
            if meaning:
                label += f"\nMeaning: {meaning}"
        
        # Add thought if available (truncated)
        if node.thought:
            thought_short = node.thought[:50] + "..." if len(node.thought) > 50 else node.thought
            label += f"\nThought: {thought_short}"
        
        G.node(simple_id, label, fillcolor=fillcolor, style=style)
        
        # Store mapping for edge creation
        if not hasattr(self, '_node_id_mapping'):
            self._node_id_mapping = {}
        self._node_id_mapping[node.node_id] = simple_id
    
    def _add_edges_to_graphviz(self, G):
        """Add edges to Graphviz graph."""
        for node in self.nodes:
            if node.is_root():
                # Add edge from root to first state
                first_state = self._get_first_state()
                if first_state:
                    G.edge(self.root.node_id, first_state.node_id, label="start", style="bold")
            elif node.is_state():
                # Add edges from state to its candidates
                for candidate_id in node.candidates:
                    candidate = self.get_node(candidate_id)
                    if candidate and candidate.is_candidate():
                        # Determine edge style based on selection
                        if candidate.is_selected():
                            G.edge(node.node_id, candidate_id, label="selected", style="bold", color="green")
                        else:
                            G.edge(node.node_id, candidate_id, label="candidate", style="dashed", color="gray")
                
                # Add edge from selected candidate to next state
                selected_candidate = self._get_selected_candidate_for_state(node)
                if selected_candidate:
                    next_state = self._get_next_state(node.step)
                    if next_state:
                        G.edge(selected_candidate.node_id, next_state.node_id, label="execute", style="bold", color="blue")
    
    def _get_first_state(self) -> Optional[TrajState]:
        """Get the first state node (step 1)."""
        for node in self.nodes:
            if node.is_state() and node.step == 1:
                return node
        return None
    
    def _get_selected_candidate_for_state(self, state: TrajState) -> Optional[TrajCandidate]:
        """Get the selected candidate for a given state."""
        for candidate_id in state.candidates:
            candidate = self.get_node(candidate_id)
            if candidate and candidate.is_selected():
                return candidate
        return None
    
    def _get_next_state(self, current_step: int) -> Optional[TrajState]:
        """Get the next state node after the given step."""
        for node in self.nodes:
            if node.is_state() and node.step == current_step + 1:
                return node
        return None
    
    # ---- Tree manipulation methods for RuntimeManager ----
    
    def add_node(self, node: TrajNode) -> None:
        """Add a node to the tree."""
        self.nodes.append(node)
    
    def add_state_node(self, node_id: str, parent_id: str, step: int, url: Optional[str] = None, 
                      observation_hash: Optional[str] = None, obs_nodes_info: Optional[Dict[str, Any]] = None,
                      screenshot_path: Optional[str] = None) -> TrajState:
        """Add a new state node to the tree."""
        state_node = TrajState(
            node_id=node_id,
            parent_id=parent_id,
            step=step,
            url=url,
            observation_hash=observation_hash,
            obs_nodes_info=obs_nodes_info,
            screenshot_path=screenshot_path,
            candidates=[]
        )
        self.add_node(state_node)
        return state_node
    
    def add_candidate_node(self, node_id: str, parent_id: str, thought: Optional[str] = None,
                          action: Optional[str] = None, meaning: Optional[str] = None,
                          status: CandidateNodeStatus = CandidateNodeStatus.CANDIDATE) -> TrajCandidate:
        """Add a new candidate node to the tree."""
        candidate_node = TrajCandidate(
            node_id=node_id,
            parent_id=parent_id,
            thought=thought,
            action=action,
            meaning=meaning,
            status=status
        )
        self.add_node(candidate_node)
        return candidate_node
    
    def add_candidate_to_state(self, state_node_id: str, candidate_node_id: str) -> None:
        """Add a candidate node to a state node's candidates list."""
        state_node = self.get_node(state_node_id)
        if state_node and state_node.is_state():
            if candidate_node_id not in state_node.candidates:
                state_node.candidates.append(candidate_node_id)
    
    def mark_candidate_as_selected(self, candidate_node_id: str) -> None:
        """Mark a candidate node as selected."""
        candidate_node = self.get_node(candidate_node_id)
        if candidate_node and candidate_node.is_candidate():
            candidate_node.status = CandidateNodeStatus.SELECTED