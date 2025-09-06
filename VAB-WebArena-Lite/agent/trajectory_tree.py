from typing import List, Optional, Dict, Any
import json
import base64
import mimetypes
from pathlib import Path

from .types import TrajRoot, TrajNode, TrajEdge, NodeStatus, BlockInfo


class TrajectoryTree:
    """Complete trajectory tree: one root + multiple nodes/edges."""
    
    def __init__(self, root: TrajRoot):
        self.root = root
        self.nodes: List[TrajNode] = [root]  # Include root in nodes list
        self.edges: List[TrajEdge] = []

    # ---- Runtime convenience methods (no business logic) ----

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
        """Return main path nodes sorted by step (including root: step=0)."""
        # Main path consists of selected nodes
        main_nodes = [n for n in self.nodes if n.status == NodeStatus.SELECTED]
        return sorted(main_nodes, key=lambda n: n.step)

    def main_path_edges(self) -> List[TrajEdge]:
        """Return main path edges sorted by child.step."""
        # Main path edges connect selected nodes
        selected_node_ids = {n.node_id for n in self.nodes if n.status == NodeStatus.SELECTED}
        step_by_child = {n.node_id: n.step for n in self.nodes}
        path_edges = [e for e in self.edges if e.child_id in selected_node_ids]
        return sorted(path_edges, key=lambda e: step_by_child.get(e.child_id, 10**9))

    def edges_from(self, node_id: str) -> List[TrajEdge]:
        """Return all edges from the specified node."""
        return [e for e in self.edges if e.parent_id == node_id]

    def edges_to(self, node_id: str) -> List[TrajEdge]:
        """Return all edges to the specified node."""
        return [e for e in self.edges if e.child_id == node_id]

    def get_candidates_at_node(self, node_id: str) -> List[BlockInfo]:
        """Get candidate actions list for the specified node."""
        node = self.get_node(node_id)
        if node:
            return node.candidates
        return []

    def set_candidates_at_node(self, node_id: str, candidates: List[BlockInfo]) -> None:
        """Set candidate actions list for the specified node."""
        node = self.get_node(node_id)
        if node:
            node.candidates = candidates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "nodes": [node.model_dump() if hasattr(node, 'model_dump') else node.dict() for node in self.nodes],
            "edges": [edge.model_dump() if hasattr(edge, 'model_dump') else edge.dict() for edge in self.edges]
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_graphviz(self) -> str:
        """Generate Graphviz DOT format trajectory graph."""
        lines = ["digraph Trajectory {", "  rankdir=TB;", "  node [shape=box, style=filled];"]
        
        # Add node definitions
        lines.extend(self._generate_graphviz_nodes())
        
        # Add edge definitions
        lines.extend(self._generate_graphviz_edges())
        
        lines.append("}")
        return "\n".join(lines)

    def to_svg(self, output_path: str = None) -> str:
        """Generate SVG trajectory graph, all actions as nodes distinguished by status."""
        
        # Convert tree structure to visualization data
        nodes_data, edges_data = self._build_visualization_data()
        
        # Generate SVG content
        svg_content = self._generate_svg_template(nodes_data, edges_data)
        
        # Save SVG file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
        
        return svg_content

    def _build_visualization_data(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build visualization data from the tree structure."""
        nodes_data = []
        edges_data = []
        
        # Add root node first
        nodes_data.append({
            "id": "root",
            "label": f"Root\n{self.root.intent or 'Task'}",
            "type": "root",
            "step": 0,
            "url": None,
            "screenshot": None,
            "status": "root"
        })
        
        # Process each non-root node in the tree
        state_index = 0
        for i, node in enumerate(self.nodes):
            if node.is_root():
                continue
            
            # Handle screenshots
            screenshot_data = None
            screenshot_path = None
            mime = None

            if node.checkpoint and node.checkpoint.observation:
                screenshot_path = node.checkpoint.observation.screenshot_path  # 先原样记录
                if screenshot_path:
                    mime, _ = mimetypes.guess_type(screenshot_path)
                    mime = mime or "image/png"
                    try:
                        with open(screenshot_path, "rb") as f:
                            img_data = f.read()
                            screenshot_data = base64.b64encode(img_data).decode("ascii")
                    except Exception:
                        screenshot_data = None  # 读不到就走回退
            
            # Add state node (representing reaching a certain state)
            state_node_id = f"state_{state_index}"
            nodes_data.append({
                "id": state_node_id,
                "label": f"Step {node.step}",
                "type": "state",
                "step": node.step,
                "url": node.url,
                "screenshot": screenshot_data,
                "screenshot_path": screenshot_path,  # 总是有（如果传入了）
                "mime": mime or "image/png",
                "status": "state"
            })
            
            # Create action nodes for each candidate
            candidates = self.get_candidates_at_node(node.node_id)
            selected_action = None
            if node.checkpoint and node.checkpoint.block and node.checkpoint.block.action:
                selected_action = node.checkpoint.block.action
            
            # If there are candidates, create action nodes for each
            if candidates:
                selected_candidate_found = False
                for j, candidate in enumerate(candidates):
                    action_node_id = f"action_{state_index}_{j}"
                    action_short = candidate.action[:40] + "..." if len(candidate.action) > 40 else candidate.action
                    
                    # Determine action status
                    status = "candidate"  # Default to candidate
                    if selected_action and candidate.action == selected_action:
                        status = "selected"  # Selected
                        selected_candidate_found = True
                    
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
                    
                    # Add edge from state node to action node
                    edges_data.append({
                        "from": state_node_id,
                        "to": action_node_id,
                        "label": "",
                        "type": "action_edge",
                        "status": status
                    })
                    
                    # If action is selected, add edge from action node to next state node
                    if status == "selected":
                        next_state_id = f"state_{state_index + 1}" if state_index + 1 < len([n for n in self.nodes if not n.is_root()]) else "end"
                        edges_data.append({
                            "from": action_node_id,
                            "to": next_state_id,
                            "label": "",
                            "type": "execution_edge",
                            "status": "executed"
                        })
                
                # If selected action is not in candidates, add it as a separate selected action node
                if selected_action and not selected_candidate_found:
                    action_node_id = f"action_{state_index}_selected"
                    action_short = selected_action[:40] + "..." if len(selected_action) > 40 else selected_action
                    
                    nodes_data.append({
                        "id": action_node_id,
                        "label": action_short,
                        "type": "action",
                        "step": node.step,
                        "url": None,
                        "screenshot": None,
                        "status": "selected",
                        "thought": node.checkpoint.block.thought if node.checkpoint and node.checkpoint.block else "",
                        "action": selected_action
                    })
                    
                    # Add edge from state node to selected action node
                    edges_data.append({
                        "from": state_node_id,
                        "to": action_node_id,
                        "label": "",
                        "type": "action_edge",
                        "status": "selected"
                    })
                    
                    # Add edge from selected action node to next state node
                    next_state_id = f"state_{state_index + 1}" if state_index + 1 < len([n for n in self.nodes if not n.is_root()]) else "end"
                    edges_data.append({
                        "from": action_node_id,
                        "to": next_state_id,
                        "label": "",
                        "type": "execution_edge",
                        "status": "executed"
                    })
            else:
                # If no candidates but there's a selected action, create a single action node
                if selected_action:
                    action_node_id = f"action_{state_index}_0"
                    action_short = selected_action[:40] + "..." if len(selected_action) > 40 else selected_action
                    
                    nodes_data.append({
                        "id": action_node_id,
                        "label": action_short,
                        "type": "action",
                        "step": node.step,
                        "url": None,
                        "screenshot": None,
                        "status": "selected",
                        "thought": node.checkpoint.block.thought if node.checkpoint and node.checkpoint.block else "",
                        "action": selected_action
                    })
                    
                    # Add edge from state node to action node
                    edges_data.append({
                        "from": state_node_id,
                        "to": action_node_id,
                        "label": "",
                        "type": "action_edge",
                        "status": "selected"
                    })
                    
                    # Add edge from action node to next state node
                    next_state_id = f"state_{state_index + 1}" if state_index + 1 < len([n for n in self.nodes if not n.is_root()]) else "end"
                    edges_data.append({
                        "from": action_node_id,
                        "to": next_state_id,
                        "label": "",
                        "type": "execution_edge",
                        "status": "executed"
                    })
                else:
                    # If no candidates and no selected action, add direct edge to next state
                    next_state_id = f"state_{state_index + 1}" if state_index + 1 < len([n for n in self.nodes if not n.is_root()]) else "end"
                    edges_data.append({
                        "from": state_node_id,
                        "to": next_state_id,
                        "label": "",
                        "type": "execution_edge",
                        "status": "executed"
                    })
            
            state_index += 1
        
        # Add edge from root to first state node
        if len(self.nodes) > 1:  # More than just root
            first_state_id = "state_0"
            edges_data.append({
                "from": "root",
                "to": first_state_id,
                "label": "",
                "type": "start_edge",
                "status": "start"
            })
        
        return nodes_data, edges_data

    def _generate_svg_template(self, nodes_data: List[Dict[str, Any]], edges_data: List[Dict[str, Any]]) -> str:
        """Generate SVG template for trajectory visualization."""
        
        # Calculate layout dimensions
        node_width = 300
        node_height = 120
        action_width = 250
        action_height = 80
        margin = 80
        level_width = 400  # Horizontal spacing between steps
        vertical_spacing = 150  # Vertical spacing between nodes
        
        # Calculate total dimensions based on actual content
        max_step = max((node['step'] for node in nodes_data), default=0)
        total_width = max_step * level_width + node_width + margin * 2
        total_height = len(nodes_data) * vertical_spacing + margin * 2
        
        svg_lines = [
            f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">',
            '  <defs>',
            '    <style>',
            '      .node { fill: #e1f5fe; stroke: #01579b; stroke-width: 2; cursor: pointer; }',
            '      .root { fill: #bbdefb; stroke: #0277bd; stroke-width: 3; cursor: pointer; }',
            '      .state { fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2; cursor: pointer; }',
            '      .action { fill: #fff3e0; stroke: #f57c00; stroke-width: 1; cursor: pointer; }',
            '      .selected { fill: #c8e6c9; stroke: #388e3c; stroke-width: 2; cursor: pointer; }',
            '      .candidate { fill: #ffecb3; stroke: #f9a825; stroke-width: 1; cursor: pointer; }',
            '      .edge { stroke: #666; stroke-width: 2; fill: none; }',
            '      .execution-edge { stroke: #4caf50; stroke-width: 3; fill: none; }',
            '      .action-edge { stroke: #ff9800; stroke-width: 1; fill: none; }',
            '      .text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }',
            '      .title-text { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; text-anchor: middle; }',
            '      .node:hover { stroke-width: 4; }',
            '    </style>',
            '  </defs>',
            '  <script>',
            '    function openScreenshot(screenshotPath) {',
            '      if (screenshotPath) {',
            '        window.open("file://" + screenshotPath, "_blank");',
            '      }',
            '    }',
            '  </script>'
        ]
        
        # Position nodes
        node_positions = {}
        
        # Position root node
        root_node = next((n for n in nodes_data if n['type'] == 'root'), None)
        if root_node:
            x = margin
            y = margin
            node_positions[root_node['id']] = (x, y)
        
        # Position state and action nodes by step
        step_groups = {}
        for node in nodes_data:
            if node['type'] != 'root':
                step = node['step']
                if step not in step_groups:
                    step_groups[step] = []
                step_groups[step].append(node)
        
        for step in sorted(step_groups.keys()):
            x = margin + step * level_width
            y = margin + step * vertical_spacing
            
            # Position state nodes first
            state_nodes = [n for n in step_groups[step] if n['type'] == 'state']
            for i, state_node in enumerate(state_nodes):
                node_positions[state_node['id']] = (x, y + i * (node_height + 20))
            
            # Position action nodes below state nodes
            action_nodes = [n for n in step_groups[step] if n['type'] == 'action']
            for i, action_node in enumerate(action_nodes):
                action_y = y + len(state_nodes) * (node_height + 20) + 20 + i * (action_height + 10)
                node_positions[action_node['id']] = (x + 30, action_y)
        
        # Draw edges
        for edge in edges_data:
            from_pos = node_positions.get(edge['from'])
            to_pos = node_positions.get(edge['to'])
            
            if from_pos and to_pos:
                x1, y1 = from_pos
                x2, y2 = to_pos
                
                # Adjust positions to center of nodes
                from_node = next((n for n in nodes_data if n['id'] == edge['from']), None)
                to_node = next((n for n in nodes_data if n['id'] == edge['to']), None)
                
                # Calculate center positions based on node type
                if from_node and from_node['type'] == 'root':
                    x1 += node_width // 2
                    y1 += node_height // 2
                elif from_node and from_node['type'] == 'state':
                    x1 += node_width // 2
                    y1 += node_height // 2
                elif from_node and from_node['type'] == 'action':
                    x1 += action_width // 2
                    y1 += action_height // 2
                
                if to_node and to_node['type'] == 'state':
                    x2 += node_width // 2
                    y2 += node_height // 2
                elif to_node and to_node['type'] == 'action':
                    x2 += action_width // 2
                    y2 += action_height // 2
                
                edge_class = "execution-edge" if edge['type'] == 'execution_edge' else "action-edge"
                svg_lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{edge_class}"/>')
        
        # Draw nodes
        for node in nodes_data:
            pos = node_positions.get(node['id'])
            if not pos:
                continue
                
            x, y = pos
            node_type = node['type']
            status = node.get('status', '')
            
            # Determine node class
            if node_type == 'root':
                node_class = 'root'
            elif node_type == 'state':
                node_class = 'state'
            elif node_type == 'action':
                if status == 'selected':
                    node_class = 'selected'
                else:
                    node_class = 'candidate'
            else:
                node_class = 'node'
            
            # Draw node rectangle with click event
            width = node_width if node_type != 'action' else action_width
            height = node_height if node_type != 'action' else action_height
            
            # Add click event for state nodes to open screenshot
            click_event = ""
            if node_type == 'state' and node.get('screenshot_path'):
                screenshot_path = node['screenshot_path']
                click_event = f' onclick="openScreenshot(\'{screenshot_path}\')"'
            
            svg_lines.append(f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" class="{node_class}"{click_event}/>')
            
            # Draw node text
            text_x = x + width // 2
            text_y = y + height // 2
            
            if node_type == 'root':
                # Split root label into multiple lines
                label = node["label"]
                lines = label.split('\n')
                for i, line in enumerate(lines):
                    line_y = text_y - (len(lines) - 1) * 8 + i * 16
                    svg_lines.append(f'  <text x="{text_x}" y="{line_y}" class="title-text">{line}</text>')
            else:
                # Split long labels into multiple lines
                label = node['label']
                max_chars = 30 if node_type == 'action' else 40
                
                if len(label) > max_chars:
                    words = label.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) <= max_chars:
                            current_line += (" " + word) if current_line else word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                else:
                    lines = [label]
                
                for i, line in enumerate(lines):
                    line_y = text_y - (len(lines) - 1) * 8 + i * 16
                    svg_lines.append(f'  <text x="{text_x}" y="{line_y}" class="text">{line}</text>')
        
        svg_lines.append('</svg>')
        
        return '\n'.join(svg_lines)

    def _generate_graphviz_nodes(self) -> List[str]:
        """Generate Graphviz node definitions."""
        lines = []
        
        # Define node styles
        lines.append("  // Node styles")
        lines.append('  root [label="Root\\n' + (self.root.intent or "Task") + '", fillcolor=lightblue];')
        
        # Add all nodes (using safe node IDs)
        for i, node in enumerate(self.nodes):
            if node.is_root():
                continue  # Root node is already defined above
            
            safe_id = f"node_{i}"
            label = f"Step {node.step}"
            if node.url:
                # Truncate long URLs
                url_short = node.url[:50] + "..." if len(node.url) > 50 else node.url
                label += f"\\n{url_short}"
            
            # Add candidates info to node label
            candidates = self.get_candidates_at_node(node.node_id)
            if candidates:
                candidates_text = f"\\nCandidates: {len(candidates)}"
                label += candidates_text
                
            lines.append(f'  {safe_id} [label="{label}", fillcolor=lightgreen];')
        
        return lines

    def _generate_graphviz_edges(self) -> List[str]:
        """Generate Graphviz edge definitions."""
        lines = []
        
        # Add main path edges (SELECTED)
        lines.append("  // Main path (selected actions)")
        for edge in self.main_path_edges():
            parent = self._get_safe_node_id(edge.parent_id)
            child = self._get_safe_node_id(edge.child_id, is_temp=True)
            
            action_short = edge.action[:30] + "..." if len(edge.action) > 30 else edge.action
            lines.append(f'  {parent} -> {child} [label="{action_short}", color=green, penwidth=2];')
        
        # Add candidate edges (connecting to candidate nodes)
        lines.append("  // Candidate actions")
        for edge in self.edges:
            # Check if target node is in candidate state
            target_node = self.get_node(edge.child_id)
            if target_node and target_node.status == NodeStatus.CANDIDATE:
                parent = self._get_safe_node_id(edge.parent_id)
                child = f"temp_{edge.child_id}"
                
                action_short = edge.action[:30] + "..." if len(edge.action) > 30 else edge.action
                lines.append(f'  {parent} -> {child} [label="{action_short}", color=red, style=dashed];')
        
        # Add node candidates as subgraph
        lines.append("  // Node candidates details")
        for i, node in enumerate(self.nodes):
            if node.is_root():
                continue
            safe_id = f"node_{i}"
            candidates = self.get_candidates_at_node(node.node_id)
            if candidates:
                for j, candidate in enumerate(candidates):
                    candidate_id = f"{safe_id}_candidate_{j}"
                    action_short = candidate.action[:40] + "..." if len(candidate.action) > 40 else candidate.action
                    lines.append(f'  {candidate_id} [label="{action_short}", shape=ellipse, fillcolor=lightyellow, style=dashed];')
                    lines.append(f'  {safe_id} -> {candidate_id} [style=dotted, color=orange, label="candidate"];')
        
        return lines

    def _get_safe_node_id(self, node_id: str, is_temp: bool = False) -> str:
        """Get safe node ID for Graphviz output."""
        if node_id == self.root.node_id:
            return "root"
        
        # Find the node index
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                if is_temp:
                    return f"temp_{node_id}"
                else:
                    return f"node_{i}"
        
        # Fallback
        return f"temp_{node_id}" if is_temp else node_id
