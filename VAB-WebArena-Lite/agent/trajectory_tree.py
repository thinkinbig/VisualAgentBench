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

    def to_interactive_html(self, output_path: str = None) -> str:
        """Generate unified interactive HTML trajectory graph, all actions as nodes distinguished by status."""
        
        # Convert tree structure to visualization data
        nodes_data, edges_data = self._build_visualization_data()
        
        # Generate HTML content
        html_content = self._generate_html_template(nodes_data, edges_data)
        
        # Save HTML file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content

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

    def _generate_html_template(self, nodes_data: List[Dict[str, Any]], edges_data: List[Dict[str, Any]]) -> str:
        """Generate the complete HTML template with embedded data."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Trajectory Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        {self._get_css_styles()}
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
        {self._get_javascript_code(nodes_data, edges_data)}
    </script>
</body>
</html>"""

    def _get_css_styles(self) -> str:
        """Get CSS styles for the visualization."""
        return """
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        #network {
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            background-color: white;
            border-radius: 8px;
        }
        #info {
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #screenshot {
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }
        .candidate {
            margin: 5px 0;
            padding: 8px;
            background-color: #f8f9fa;
            border-left: 3px solid #dc3545;
            border-radius: 4px;
        }
        .executed-candidate {
            border-left-color: #28a745;
        }
        .node-info {
            margin: 10px 0;
        }
        .url {
            color: #007bff;
            word-break: break-all;
        }
        """

    def _get_javascript_code(self, nodes_data: List[Dict[str, Any]], edges_data: List[Dict[str, Any]]) -> str:
        """Get JavaScript code for the visualization."""
        # Create a separate data structure without screenshots for JSON serialization
        nodes_data_no_screenshots = []
        for node in nodes_data:
            node_copy = node.copy()
            if 'screenshot' in node_copy:
                del node_copy['screenshot']  # Remove screenshot from JSON
            nodes_data_no_screenshots.append(node_copy)
        
        # Store screenshots and mime types separately
        screenshots = {}
        mime_types = {}
        for node in nodes_data:
            if 'screenshot' in node and node['screenshot']:
                screenshots[node['id']] = node['screenshot']
                mime_types[node['id']] = node.get('mime', 'image/png')
        
        return f"""
        // Screenshot data (stored separately to avoid JSON serialization issues)
        const screenshots = {json.dumps(screenshots)};
        const mimeTypes = {json.dumps(mime_types)};
        
        // Data
        const nodes = new vis.DataSet({json.dumps(nodes_data_no_screenshots, indent=2)});
        const edges = new vis.DataSet({json.dumps(edges_data, indent=2)});
        
        // Network configuration
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
        
        // Set different styles for different node types
        nodes.forEach(function(node) {{
            if (node.type === 'root') {{
                node.color = {{
                    background: '#e3f2fd',
                    border: '#1976d2'
                }};
                node.shape = 'box';
            }} else if (node.type === 'state') {{
                node.color = {{
                    background: '#e8f5e8',
                    border: '#388e3c'
                }};
                node.shape = 'box';
            }} else if (node.type === 'action') {{
                node.color = {{
                    background: '#fff8e1',
                    border: '#f57c00'
                }};
                node.shape = 'ellipse';
                node.font = {{ size: 12 }};
            }}
        }});
        
        // Set different styles for different edge types
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
        
        // Node click event
        network.on('click', function (params) {{
            console.log('Click event triggered!', params);
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Clicked node:', node);
                displayNodeInfo(node);
            }} else {{
                console.log('No nodes clicked');
            }}
        }});
        
        function displayNodeInfo(node) {{
            console.log('displayNodeInfo called with node:', node);
            const infoDiv = document.getElementById('info');
            let html = `<h3>Node: ${{node.label}}</h3>`;
            
            if (node.type === 'root') {{
                html += `<p><strong>Task:</strong> ${{node.label.split('\\n')[1] || 'N/A'}}</p>`;
            }} else if (node.type === 'action') {{
                html += `<div class="node-info">`;
                html += `<p><strong>Type:</strong> Candidate Action</p>`;
                html += `<p><strong>Parent Step:</strong> ${{node.step}}</p>`;
                html += `<p><strong>Action:</strong> ${{node.action}}</p>`;
                if (node.thought) {{
                    html += `<p><strong>Thought:</strong> ${{node.thought}}</p>`;
                }}
                html += `</div>`;
            }} else {{
                console.log('Processing state node, screenshot available:', !!node.screenshot);
                html += `<div class="node-info">`;
                html += `<p><strong>Step:</strong> ${{node.step}}</p>`;
                if (node.url) {{
                    html += `<p><strong>URL:</strong> <span class="url">${{node.url}}</span></p>`;
                }}
                html += `</div>`;
                
                // Display screenshot
                console.log('Looking for screenshot for node:', node.id);
                console.log('Available screenshots:', Object.keys(screenshots));
                const screenshotData = screenshots[node.id];
                const mimeType = mimeTypes[node.id] || 'image/png';
                console.log('Screenshot data found:', !!screenshotData);
                console.log('MIME type:', mimeType);
                if (screenshotData) {{
                    console.log('Adding screenshot to HTML from screenshots object');
                    console.log('Screenshot data length:', screenshotData.length);
                    html += `<h4>Screenshot:</h4>`;
                    html += `<img id="screenshot" src="data:${{mimeType}};base64,${{screenshotData}}" alt="Screenshot" style="max-width: 100%; max-height: 400px; border: 1px solid #ddd; border-radius: 4px;">`;
                }} else if (node.screenshot_path) {{
                    console.log('Using screenshot_path fallback');
                    html += `<h4>Screenshot:</h4>`;
                    // Convert absolute path to file:// URL as fallback
                    let screenshotPath = node.screenshot_path;
                    if (screenshotPath.startsWith('/')) {{
                        screenshotPath = 'file://' + screenshotPath;
                    }}
                    html += `<img id="screenshot" src="${{screenshotPath}}" alt="Screenshot" style="max-width: 100%; max-height: 400px; border: 1px solid #ddd; border-radius: 4px;">`;
                }} else {{
                    console.log('No screenshot available');
                    html += `<p><em>No screenshot available</em></p>`;
                }}
            }}
            
            console.log('Final HTML length:', html.length);
            infoDiv.innerHTML = html;
        }}
        
        // Initialize display with root node information
        displayNodeInfo(nodes.get('root'));
        """
