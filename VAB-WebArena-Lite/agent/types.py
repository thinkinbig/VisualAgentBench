"""
Agent-layer type definitions for staged policy and reward evaluation.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, validator
from enum import Enum
import re
if TYPE_CHECKING:
    from browser_env.trajectory import Trajectory
else:
    Trajectory = list  # type: ignore[assignment]

# Avoid circular import with llms.types by importing ParsedAction only for type checking
if TYPE_CHECKING:
    from llms.types import ParsedAction  # type: ignore
else:
    ParsedAction = Any  # type: ignore


class NodeStatus(str, Enum):
    """节点状态：简化为两种状态。"""
    CANDIDATE = "candidate"     # 候选状态：节点包含候选动作
    SELECTED = "selected"       # 已选择：节点被选中并执行


class CheckpointInfo(BaseModel):
    """Checkpoint information from agent response (logging/memory for policy)."""
    step: int = Field(description="Monotonic turn counter (1-based). Increment every agent turn.")
    url: str = Field(description="Canonical current page URL (normalized by the environment).")
    block: Optional["BlockInfo"] = Field(
        None,
        description="The full BLOCK (thought + action) that was executed last turn.",
    )
    objective: str = Field(description="Echo of the task's OBJECTIVE. Do not rewrite or expand.")
    observation: str = Field(description="AXTREE text of the current page. Include bids and labels.")
    screenshot_path: Optional[str] = Field(None, description="Path to the screenshot image for this checkpoint.")


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
    """
    step: int = Field(default=0, description="Monotonic turn counter (1-based preferred externally).")
    meta: Meta = Field(default_factory=Meta, description="Shared runtime context.")
    selected_block: Optional[BlockInfo] = Field(None, description="Winner after knockout.")
    tournament_history: List[PairwiseMatch] = Field(default_factory=list, description="Pairwise comparison records.")
    checkpoint: Optional[CheckpointInfo] = Field(None, description="Latest CHECKPOINT snapshot parsed from policy output.")
    aggregate: Optional[AggregateInfo] = Field(default_factory=AggregateInfo, description="Latest AGGREGATE working memory parsed from policy output.")
    current_round_samples: List[str] = Field(default_factory=list, description="Actions sampled in current round to avoid duplicates.")
    trajectory_tree: Optional["TrajectoryTree"] = Field(default=None, description="Full trajectory tree with root/nodes/edges for this run.")


# ========= Trajectory (轨迹树) 数据结构 =========

class TrajRoot(BaseModel):
    """Root 节点：仅保存全局元信息，不含动作。"""
    run_id: str = Field(..., description="Unique id for this episode/run")
    intent: str = Field(..., description="Task intent/objective")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional extra metadata (seed, model tags, etc.)")

    @validator("run_id", "intent", pre=True, always=True)
    def _strip_basic(cls, v):
        return v.strip() if isinstance(v, str) else v


class TrajNode(BaseModel):
    """普通节点：代表到达后的浏览器状态（URL/指纹/可选 checkpoint）。"""
    node_id: str = Field(..., description="Unique id within the trajectory tree")
    parent_id: Optional[str] = Field(None, description="Parent node id; None for root")
    step: int = Field(..., description="1-based step index along the EXECUTED main path (root=0)")
    url: Optional[str] = Field(None, description="Current page URL at this node")
    observation_hash: Optional[str] = Field(None, description="Fingerprint of AXTREE/screenshot for dedup/debug")
    checkpoint: Optional[CheckpointInfo] = Field(None, description="Optional checkpoint snapshot for this node")
    screenshot_path: Optional[str] = Field(None, description="Filesystem path to the screenshot image for this node")
    obs_nodes_info: Optional[Dict[str, Any]] = Field(
        None,
        description="AXTREE/SoM nodes mapping (ids -> bounds/centers/text) for clickable overlays",
    )
    labels: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary tags for filtering/searching")
    status: NodeStatus = Field(default=NodeStatus.CANDIDATE, description="Current status of this node")
    candidates: List[BlockInfo] = Field(default_factory=list, description="Candidate actions available at this node")

    @validator("node_id", "parent_id", "url", "observation_hash", pre=True, always=True)
    def _strip_opt(cls, v):
        return v.strip() if isinstance(v, str) else v
    
    @validator("screenshot_path", pre=True, always=True)
    def _strip_path(cls, v):
        return v.strip() if isinstance(v, str) else v

    def is_candidate(self) -> bool:
        """检查节点是否为候选状态。"""
        return self.status == NodeStatus.CANDIDATE

    def is_selected(self) -> bool:
        """检查节点是否已被选择。"""
        return self.status == NodeStatus.SELECTED


class TrajEdge(BaseModel):
    """边：从 parent -> child 的'如何到达'（动作/思考）。"""
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


class TrajectoryTree(BaseModel):
    """完整轨迹树：一个 root + 多个节点/边。"""
    root: TrajRoot
    nodes: List[TrajNode] = Field(default_factory=list)
    edges: List[TrajEdge] = Field(default_factory=list)

    # ---- 运行期便捷方法（不涉及业务逻辑） ----

    def add_node(self, node: TrajNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: TrajEdge) -> None:
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[TrajNode]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def children_of(self, node_id: str) -> List[TrajNode]:
        child_ids = [e.child_id for e in self.edges if e.parent_id == node_id]
        return [n for n in self.nodes if n.node_id in child_ids]

    def main_path_nodes(self) -> List[TrajNode]:
        """按 step 排序返回主路径节点（含 root: step=0）。"""
        # 主路径由已选择的节点组成
        main_nodes = [n for n in self.nodes if n.status == NodeStatus.SELECTED]
        return sorted(main_nodes, key=lambda n: n.step)

    def main_path_edges(self) -> List[TrajEdge]:
        """按 child.step 排序返回主路径上的边。"""
        # 主路径的边连接已选择的节点
        selected_node_ids = {n.node_id for n in self.nodes if n.status == NodeStatus.SELECTED}
        step_by_child = {n.node_id: n.step for n in self.nodes}
        path_edges = [e for e in self.edges if e.child_id in selected_node_ids]
        return sorted(path_edges, key=lambda e: step_by_child.get(e.child_id, 10**9))

    def edges_from(self, node_id: str) -> List[TrajEdge]:
        """返回从指定节点出发的所有边。"""
        return [e for e in self.edges if e.parent_id == node_id]

    def edges_to(self, node_id: str) -> List[TrajEdge]:
        """返回到达指定节点的所有边。"""
        return [e for e in self.edges if e.child_id == node_id]

    def get_candidates_at_node(self, node_id: str) -> List[BlockInfo]:
        """获取指定节点的候选动作列表。"""
        node = self.get_node(node_id)
        if node:
            return node.candidates
        return []

    def set_candidates_at_node(self, node_id: str, candidates: List[BlockInfo]) -> None:
        """设置指定节点的候选动作列表。"""
        node = self.get_node(node_id)
        if node:
            node.candidates = candidates

    def to_graphviz(self) -> str:
        """生成Graphviz DOT格式的轨迹图。"""
        lines = ["digraph Trajectory {", "  rankdir=TB;", "  node [shape=box, style=filled];"]
        
        # 定义节点样式
        lines.append("  // Node styles")
        lines.append('  root [label="Root\\n' + (self.root.intent or "Task") + '", fillcolor=lightblue];')
        
        # 添加所有节点（使用安全的节点ID）
        node_id_map = {"root": "root"}
        for i, node in enumerate(self.nodes):
            if node.node_id == "root":
                continue
            safe_id = f"node_{i}"
            node_id_map[node.node_id] = safe_id
            
            label = f"Step {node.step}"
            if node.url:
                # 截断长URL
                url_short = node.url[:50] + "..." if len(node.url) > 50 else node.url
                label += f"\\n{url_short}"
            
            # 添加candidates信息到节点标签
            candidates = self.get_candidates_at_node(node.node_id)
            if candidates:
                candidates_text = f"\\nCandidates: {len(candidates)}"
                label += candidates_text
                
            lines.append(f'  {safe_id} [label="{label}", fillcolor=lightgreen];')
        
        # 添加主路径边（SELECTED）
        lines.append("  // Main path (selected actions)")
        for edge in self.main_path_edges():
            parent = node_id_map.get(edge.parent_id, "root")
            child = node_id_map.get(edge.child_id, f"temp_{edge.child_id}")
            
            action_short = edge.action[:30] + "..." if len(edge.action) > 30 else edge.action
            lines.append(f'  {parent} -> {child} [label="{action_short}", color=green, penwidth=2];')
        
        # 添加候选边（连接到候选节点）
        lines.append("  // Candidate actions")
        for edge in self.edges:
            # 检查目标节点是否为候选状态
            target_node = self.get_node(edge.child_id)
            if target_node and target_node.status == NodeStatus.CANDIDATE:
                parent = node_id_map.get(edge.parent_id, "root")
                child = f"temp_{edge.child_id}"
                
                action_short = edge.action[:30] + "..." if len(edge.action) > 30 else edge.action
                lines.append(f'  {parent} -> {child} [label="{action_short}", color=red, style=dashed];')
        
        # 添加节点candidates作为子图
        lines.append("  // Node candidates details")
        for i, node in enumerate(self.nodes):
            if node.node_id == "root":
                continue
            safe_id = f"node_{i}"
            candidates = self.get_candidates_at_node(node.node_id)
            if candidates:
                for j, candidate in enumerate(candidates):
                    candidate_id = f"{safe_id}_candidate_{j}"
                    action_short = candidate.action[:40] + "..." if len(candidate.action) > 40 else candidate.action
                    lines.append(f'  {candidate_id} [label="{action_short}", shape=ellipse, fillcolor=lightyellow, style=dashed];')
                    lines.append(f'  {safe_id} -> {candidate_id} [style=dotted, color=orange, label="candidate"];')
        
        lines.append("}")
        return "\n".join(lines)

    def to_interactive_html(self, output_path: str = None) -> str:
        """生成统一的交互式HTML轨迹图，所有动作都是节点，用状态区分。"""
        import json
        import base64
        from pathlib import Path
        
        # 构建节点和边的数据
        nodes_data = []
        edges_data = []
        
        # 添加root节点
        nodes_data.append({
            "id": "root",
            "label": f"Root\n{self.root.intent or 'Task'}",
            "type": "root",
            "step": 0,
            "url": None,
            "screenshot": None,
            "status": "root"
        })
        
        # 为每个轨迹节点创建状态和动作节点
        for i, node in enumerate(self.nodes):
            if node.node_id == "root":
                continue
            
            # 处理截图
            screenshot_data = None
            if node.screenshot_path and Path(node.screenshot_path).exists():
                try:
                    with open(node.screenshot_path, 'rb') as f:
                        img_data = f.read()
                        screenshot_data = base64.b64encode(img_data).decode('utf-8')
                except Exception:
                    screenshot_data = None
            
            # 添加状态节点（表示到达某个状态）
            state_node_id = f"state_{i}"
            nodes_data.append({
                "id": state_node_id,
                "label": f"Step {node.step}",
                "type": "state",
                "step": node.step,
                "url": node.url,
                "screenshot": screenshot_data,
                "status": "state"
            })
            
            # 为每个candidate创建动作节点
            candidates = self.get_candidates_at_node(node.node_id)
            for j, candidate in enumerate(candidates):
                action_node_id = f"action_{i}_{j}"
                action_short = candidate.action[:40] + "..." if len(candidate.action) > 40 else candidate.action
                
                # 确定动作状态
                status = "candidate"  # 默认为候选
                if node.checkpoint and node.checkpoint.block and node.checkpoint.block.action:
                    if candidate.action == node.checkpoint.block.action:
                        status = "selected"  # 已选择
                
                nodes_data.append({
                    "id": action_node_id,
                    "label": action_short,
                    "type": "action",
                    "step": node.step,
                    "url": None,
                    "screenshot": None,
                    "status": status,
                    "thought": candidate.thought,
                    "action": candidate.action
                })
                
                # 添加从状态节点到动作节点的边
                edges_data.append({
                    "from": state_node_id,
                    "to": action_node_id,
                    "label": "",
                    "type": "action_edge",
                    "status": status
                })
                
                # 如果动作已选择，添加从动作节点到下一个状态节点的边
                if status == "selected":
                    next_state_id = f"state_{i+1}" if i+1 < len(self.nodes) else "end"
                    edges_data.append({
                        "from": action_node_id,
                        "to": next_state_id,
                        "label": "",
                        "type": "execution_edge",
                        "status": "executed"
                    })
        
        # 添加从root到第一个状态节点的边
        if self.nodes:
            first_state_id = "state_0"
            edges_data.append({
                "from": "root",
                "to": first_state_id,
                "label": "",
                "type": "start_edge",
                "status": "start"
            })
        
        # 生成HTML内容
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Trajectory Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        #network {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            background-color: white;
            border-radius: 8px;
        }}
        #info {{
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        #screenshot {{
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .candidate {{
            margin: 5px 0;
            padding: 8px;
            background-color: #f8f9fa;
            border-left: 3px solid #dc3545;
            border-radius: 4px;
        }}
        .executed-candidate {{
            border-left-color: #28a745;
        }}
        .node-info {{
            margin: 10px 0;
        }}
        .url {{
            color: #007bff;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>Interactive Trajectory Visualization</h1>
    <div id="network"></div>
    <div id="info">
        <h3>Node Information</h3>
        <p>Click on a node to view details</p>
    </div>

    <script>
        // 数据
        const nodes = new vis.DataSet({json.dumps(nodes_data, indent=2)});
        const edges = new vis.DataSet({json.dumps(edges_data, indent=2)});
        
        // 网络配置
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'box',
                font: {{ size: 14 }},
                borderWidth: 2,
                shadow: true,
                color: {{
                    background: '#e1f5fe',
                    border: '#01579b',
                    highlight: {{
                        background: '#b3e5fc',
                        border: '#0277bd'
                    }}
                }},
                chosen: {{
                    node: function(values, id, selected, hovering) {{
                        if (selected || hovering) {{
                            values.color.border = '#ff6f00';
                            values.color.background = '#fff3e0';
                        }}
                    }}
                }}
            }},
            edges: {{
                font: {{ size: 12 }},
                arrows: {{ to: {{ enabled: true, scaleFactor: 1 }} }},
                smooth: {{ type: 'continuous' }},
                color: {{
                    color: '#666',
                    highlight: '#ff6f00'
                }}
            }},
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed'
                }}
            }},
            physics: {{
                enabled: false
            }}
        }};
        
        // 为不同类型的节点设置不同样式
        nodes.forEach(function(node) {{
            if (node.type === 'root') {{
                node.color = {{
                    background: '#e3f2fd',
                    border: '#1976d2'
                }};
                node.shape = 'box';
            }} else if (node.type === 'step') {{
                node.color = {{
                    background: '#e8f5e8',
                    border: '#388e3c'
                }};
                node.shape = 'box';
            }} else if (node.type === 'candidate') {{
                node.color = {{
                    background: '#fff8e1',
                    border: '#f57c00'
                }};
                node.shape = 'ellipse';
                node.font = {{ size: 12 }};
            }}
        }});
        
        // 为不同类型的边设置不同样式
        edges.forEach(function(edge) {{
            if (edge.type === 'execution_edge' && edge.status === 'selected') {{
                edge.color = {{
                    color: '#4caf50',
                    highlight: '#2e7d32'
                }};
                edge.width = 3;
            }} else if (edge.type === 'action_edge' && edge.status === 'candidate') {{
                edge.color = {{
                    color: '#ff9800',
                    highlight: '#f57c00'
                }};
                edge.dashes = [5, 5];
                edge.width = 1;
            }} else if (edge.type === 'action_edge' && edge.status === 'selected') {{
                edge.color = {{
                    color: '#4caf50',
                    highlight: '#2e7d32'
                }};
                edge.width = 2;
            }}
        }});
        
        const network = new vis.Network(container, data, options);
        
        // 节点点击事件
        network.on('click', function (params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                displayNodeInfo(node);
            }}
        }});
        
        function displayNodeInfo(node) {{
            const infoDiv = document.getElementById('info');
            let html = `<h3>Node: ${{node.label}}</h3>`;
            
            if (node.type === 'root') {{
                html += `<p><strong>Task:</strong> ${{node.label.split('\\n')[1] || 'N/A'}}</p>`;
            }} else if (node.type === 'candidate') {{
                html += `<div class="node-info">`;
                html += `<p><strong>Type:</strong> Candidate Action</p>`;
                html += `<p><strong>Parent Step:</strong> ${{node.step}}</p>`;
                html += `<p><strong>Action:</strong> ${{node.action}}</p>`;
                if (node.thought) {{
                    html += `<p><strong>Thought:</strong> ${{node.thought}}</p>`;
                }}
                html += `</div>`;
            }} else {{
                html += `<div class="node-info">`;
                html += `<p><strong>Step:</strong> ${{node.step}}</p>`;
                if (node.url) {{
                    html += `<p><strong>URL:</strong> <span class="url">${{node.url}}</span></p>`;
                }}
                html += `</div>`;
                
                // 显示截图
                if (node.screenshot) {{
                    html += `<h4>Screenshot:</h4>`;
                    html += `<img id="screenshot" src="data:image/png;base64,${{node.screenshot}}" alt="Screenshot">`;
                }} else {{
                    html += `<p><em>No screenshot available</em></p>`;
                }}
                
                // 显示候选动作
                if (node.candidates && node.candidates.length > 0) {{
                    html += `<h4>Candidate Actions (${{node.candidates.length}}):</h4>`;
                    node.candidates.forEach((candidate, index) => {{
                        // 检查动作是否被选择（通过checkpoint中的action匹配）
                        const isSelected = node.checkpoint && node.checkpoint.block && 
                                         node.checkpoint.block.action === candidate.action;
                        const className = isSelected ? 'candidate executed-candidate' : 'candidate';
                        html += `<div class="${{className}}">`;
                        html += `<strong>Action ${{index + 1}}:</strong> ${{candidate.action}}<br>`;
                        if (candidate.thought) {{
                            html += `<strong>Thought:</strong> ${{candidate.thought}}`;
                        }}
                        if (isSelected) {{
                            html += `<br><strong>Status:</strong> <span style="color: green;">✓ Selected</span>`;
                        }} else {{
                            html += `<br><strong>Status:</strong> <span style="color: orange;">○ Candidate</span>`;
                        }}
                        html += `</div>`;
                    }});
                }} else {{
                    html += `<p><em>No candidate actions available</em></p>`;
                }}
                
                // 显示checkpoint信息
                if (node.checkpoint) {{
                    html += `<h4>Checkpoint:</h4>`;
                    html += `<p><strong>Objective:</strong> ${{node.checkpoint.objective || 'N/A'}}</p>`;
                    if (node.checkpoint.observation) {{
                        html += `<p><strong>Observation:</strong> <pre style="white-space: pre-wrap; font-size: 12px;">${{node.checkpoint.observation.substring(0, 500)}}${{node.checkpoint.observation.length > 500 ? '...' : ''}}</pre></p>`;
                    }}
                }}
            }}
            
            infoDiv.innerHTML = html;
        }}
        
        // 初始化显示root节点信息
        displayNodeInfo(nodes.get('root'));
    </script>
</body>
</html>
"""
        
        # 保存HTML文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content


# ========= Snapshot 数据结构 =========


class SnapshotMeta(BaseModel):
    intent: Optional[str] = Field(None)
    start_url: Optional[str] = Field(None)
    current_url: Optional[str] = Field(None)
    step: int = Field(...)


class SnapshotCandidate(BaseModel):
    index: int
    thought: Optional[str] = None
    action: Optional[str] = None
    meaning: Optional[str] = None


class SnapshotRequest(BaseModel):
    intent: str
    observation: str
    trajectory: str
    start_url: str
    current_url: str


class SnapshotResponse(BaseModel):
    raw: str
    decision: str
    winner: Optional[int] = None
    is_valid: bool = False
    parse_errors: List[str] = Field(default_factory=list)
    criteria: Optional[str] = None
    analysis: Optional[str] = None
    think: Optional[str] = None


class SnapshotMatch(BaseModel):
    a: SnapshotCandidate
    b: SnapshotCandidate
    request: SnapshotRequest
    response: SnapshotResponse


class SnapshotRound(BaseModel):
    round_index: int
    pairs: List[SnapshotMatch] = Field(default_factory=list)


class SnapshotWinner(BaseModel):
    index: Optional[int] = None
    thought: Optional[str] = None
    action: Optional[str] = None
    meaning: Optional[str] = None


class SnapShot(BaseModel):
    meta: SnapshotMeta
    checkpoint: Optional[CheckpointInfo] = None
    candidates: List[SnapshotCandidate] = Field(default_factory=list)
    rounds: List[SnapshotRound] = Field(default_factory=list)
    winner: Optional[SnapshotWinner] = None
