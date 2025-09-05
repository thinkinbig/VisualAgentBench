#!/usr/bin/env python3
"""
轨迹可视化工具 - 重新实现正确的树状态维护
使用方法：
1. python visualize_trajectory.py - 生成最新的轨迹图
2. python visualize_trajectory.py --file trajectory_xxx_final.json - 可视化指定JSON文件
3. python visualize_trajectory.py --file trajectory_xxx_final.dot - 可视化指定DOT文件
4. python visualize_trajectory.py --interactive --open - 生成交互式HTML并自动打开
5. python visualize_trajectory.py --list - 列出所有可用的轨迹文件

设计原则：
- selected 动作：绿色实线
- candidate 动作：黄色虚线
- 所有动作都来自同一个父状态节点
- 正确的树状态维护
"""

import os
import sys
import argparse
import subprocess
import glob
import json
import base64
from pathlib import Path

def find_latest_trajectory():
    """找到最新的轨迹文件（优先JSON，其次DOT）"""
    trajectory_dir = Path("outputs/trajectory")
    if not trajectory_dir.exists():
        print("❌ 轨迹目录不存在: outputs/trajectory")
        return None
    
    # 优先查找JSON文件
    json_files = list(trajectory_dir.glob("*_final.json"))
    if json_files:
        # 按修改时间排序，返回最新的
        latest = max(json_files, key=lambda f: f.stat().st_mtime)
        return latest
    
    # 如果没有JSON文件，查找DOT文件
    dot_files = list(trajectory_dir.glob("*_final.dot"))
    if not dot_files:
        print("❌ 没有找到轨迹文件（_final.json 或 _final.dot）")
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(dot_files, key=lambda f: f.stat().st_mtime)
    return latest

def generate_visualization(trajectory_file, output_name=None, open_file=False, interactive=False):
    """生成轨迹可视化图 - 重新实现正确的树状态维护"""
    file_path = Path(trajectory_file)
    if not file_path.exists():
        print(f"❌ 轨迹文件不存在: {trajectory_file}")
        return False
    
    # 设置输出文件名
    if output_name is None:
        output_name = file_path.stem.replace("_final", "")
    
    output_dir = file_path.parent
    svg_path = output_dir / f"{output_name}.svg"
    png_path = output_dir / f"{output_name}.png"
    html_path = output_dir / f"{output_name}_interactive.html"
    dot_path = output_dir / f"{output_name}.dot"
    
    print(f"📊 生成轨迹图: {file_path.name}")
    
    # 确定JSON和DOT文件路径
    if file_path.suffix == '.json':
        json_file = file_path
        existing_dot_file = file_path.with_suffix('.dot')
    else:
        # 假设是DOT文件
        json_file = file_path.with_suffix('.json')
        existing_dot_file = file_path
    
    success = True
    
    # 生成交互式HTML（优先）
    if interactive:
        try:
            # 从JSON文件重新构建轨迹树
            if json_file.exists():
                sys.path.append('.')
                from agent.types import TrajectoryTree
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 重建轨迹树
                tree = TrajectoryTree(**data)
                
                # 生成交互式HTML
                generate_interactive_html(tree, str(html_path))
                print(f"✅ 交互式HTML生成成功: {html_path}")
                
                if open_file:
                    try:
                        if sys.platform.startswith('linux'):
                            subprocess.run(["xdg-open", str(html_path)])
                        elif sys.platform.startswith('darwin'):
                            subprocess.run(["open", str(html_path)])
                        elif sys.platform.startswith('win'):
                            subprocess.run(["start", str(html_path)], shell=True)
                        print(f"🔍 已打开交互式页面: {html_path}")
                    except Exception as e:
                        print(f"⚠️  无法自动打开文件: {e}")
                        print(f"   请手动打开: {html_path}")
            else:
                print(f"⚠️  找不到对应的JSON文件: {json_file}")
                print(f"   无法生成交互式HTML")
        except Exception as e:
            print(f"❌ 交互式HTML生成失败: {e}")
            success = False
    
    # 生成DOT文件和静态图片
    if json_file.exists():
        try:
            sys.path.append('.')
            from agent.types import TrajectoryTree
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建轨迹树
            tree = TrajectoryTree(**data)
            
            # 生成DOT文件
            dot_content = generate_dot_content(tree)
            with open(dot_path, 'w', encoding='utf-8') as f:
                f.write(dot_content)
            print(f"✅ DOT文件生成成功: {dot_path}")
            
            # 生成静态图片
            try:
                # 生成SVG
                subprocess.run([
                    "dot", "-Tsvg", str(dot_path), "-o", str(svg_path)
                ], check=True, capture_output=True)
                print(f"✅ SVG生成成功: {svg_path}")
                
                # 生成PNG
                subprocess.run([
                    "dot", "-Tpng", str(dot_path), "-o", str(png_path)
                ], check=True, capture_output=True)
                print(f"✅ PNG生成成功: {png_path}")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Graphviz生成失败: {e}")
                print(f"   请确保已安装Graphviz: sudo apt install graphviz")
                success = False
            except FileNotFoundError:
                print("❌ 找不到dot命令，请安装Graphviz: sudo apt install graphviz")
                success = False
                
        except Exception as e:
            print(f"❌ DOT文件生成失败: {e}")
            success = False
    
    # 如果指定了打开文件且没有生成交互式HTML，打开静态图片
    if open_file and not interactive:
        try:
            if sys.platform.startswith('linux'):
                subprocess.run(["xdg-open", str(svg_path)])
            elif sys.platform.startswith('darwin'):
                subprocess.run(["open", str(svg_path)])
            elif sys.platform.startswith('win'):
                subprocess.run(["start", str(svg_path)], shell=True)
            print(f"🔍 已打开: {svg_path}")
        except Exception as e:
            print(f"⚠️  无法自动打开文件: {e}")
            print(f"   请手动打开: {svg_path}")
    
    return success

def generate_dot_content(tree):
    """生成正确的DOT内容 - 修复树状态维护问题"""
    lines = ["digraph Trajectory {", "  rankdir=TB;", "  node [shape=box, style=filled];"]
    
    # 添加root节点
    lines.append('  root [label="Root\\n' + (tree.root.intent or "Task") + '", fillcolor=lightblue, shape=box];')
    
    # 为每个状态节点创建节点和候选动作
    for i, node in enumerate(tree.nodes):
        if node.node_id == "root":
            # 处理root节点的候选动作
            # 检查下一个节点的checkpoint来确定选中的动作
            selected_action = None
            if i + 1 < len(tree.nodes):
                next_node = tree.nodes[i + 1]
                if next_node.checkpoint and next_node.checkpoint.block and next_node.checkpoint.block.action:
                    selected_action = next_node.checkpoint.block.action
            
            candidates = tree.get_candidates_at_node(node.node_id)
            for j, candidate in enumerate(candidates):
                action_id = f"action_root_{j}"
                action_short = candidate.action[:40] + "..." if len(candidate.action) > 40 else candidate.action
                
                # 确定动作状态
                is_selected = (selected_action == candidate.action)
                
                # 根据状态设置颜色和样式
                if is_selected:
                    # 已选择的动作：绿色实线
                    lines.append(f'  {action_id} [label="{action_short}", fillcolor=lightgreen, shape=ellipse];')
                    lines.append(f'  root -> {action_id} [label="", color=green, penwidth=3];')
                else:
                    # 候选动作：黄色虚线
                    lines.append(f'  {action_id} [label="{action_short}", fillcolor=lightyellow, shape=ellipse, style=dashed];')
                    lines.append(f'  root -> {action_id} [label="", color=orange, style=dashed, penwidth=1];')
            continue
            
        # 状态节点
        state_id = f"state_{i}"
        url_short = node.url[:50] + "..." if node.url and len(node.url) > 50 else (node.url or "No URL")
        lines.append(f'  {state_id} [label="Step {node.step}\\n{url_short}", fillcolor=lightgray, shape=box];')
        
        # 从root或前一个状态连接到当前状态
        if i == 0:
            lines.append(f'  root -> {state_id} [label="Start", color=blue, penwidth=2];')
        else:
            prev_state_id = f"state_{i-1}"
            lines.append(f'  {prev_state_id} -> {state_id} [label="", color=blue, penwidth=2];')
        
        # 从edges中获取候选动作
        candidates = tree.get_candidates_at_node(node.node_id)
        for j, candidate in enumerate(candidates):
            action_id = f"action_{i}_{j}"
            action_short = candidate.action[:40] + "..." if len(candidate.action) > 40 else candidate.action
            
            # 确定动作状态
            is_selected = False
            if node.checkpoint and node.checkpoint.block and node.checkpoint.block.action:
                is_selected = (candidate.action == node.checkpoint.block.action)
            
            # 根据状态设置颜色和样式
            if is_selected:
                # 已选择的动作：绿色实线
                lines.append(f'  {action_id} [label="{action_short}", fillcolor=lightgreen, shape=ellipse];')
                lines.append(f'  {state_id} -> {action_id} [label="", color=green, penwidth=3];')
            else:
                # 候选动作：黄色虚线
                lines.append(f'  {action_id} [label="{action_short}", fillcolor=lightyellow, shape=ellipse, style=dashed];')
                lines.append(f'  {state_id} -> {action_id} [label="", color=orange, style=dashed, penwidth=1];')
    
    lines.append("}")
    return "\n".join(lines)

def generate_interactive_html(tree, output_path):
    """生成交互式HTML - 修复树状态维护问题"""
    # 构建节点和边的数据
    nodes_data = []
    edges_data = []
    
    # 添加root节点
    nodes_data.append({
        "id": "root",
        "label": f"Root\n{tree.root.intent or 'Task'}",
        "type": "root",
        "step": 0,
        "url": None,
        "screenshot": None,
        "status": "root"
    })
    
    # 为每个轨迹节点创建状态和动作节点
    for i, node in enumerate(tree.nodes):
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
            "status": "state",
            "checkpoint": node.checkpoint.model_dump() if node.checkpoint else None,
            "candidates": [candidate.model_dump() for candidate in tree.get_candidates_at_node(node.node_id)]
        })
        
        # 从root或前一个状态连接到当前状态
        if i == 0:
            edges_data.append({
                "from": "root",
                "to": state_node_id,
                "label": "Start",
                "type": "state_transition",
                "status": "executed"
            })
        else:
            prev_state_id = f"state_{i-1}"
            edges_data.append({
                "from": prev_state_id,
                "to": state_node_id,
                "label": "",
                "type": "state_transition",
                "status": "executed"
            })
        
        # 从edges中获取候选动作
        candidates = tree.get_candidates_at_node(node.node_id)
        for j, candidate in enumerate(candidates):
            action_node_id = f"action_{i}_{j}"
            action_short = candidate.action[:40] + "..." if len(candidate.action) > 40 else candidate.action
            
            # 确定动作状态
            is_selected = False
            if node.checkpoint and node.checkpoint.block and node.checkpoint.block.action:
                is_selected = (candidate.action == node.checkpoint.block.action)
            
            nodes_data.append({
                "id": action_node_id,
                "label": action_short,
                "type": "action",
                "step": node.step,
                "url": None,
                "screenshot": None,
                "status": "selected" if is_selected else "candidate",
                "thought": candidate.thought,
                "action": candidate.action,
                "is_selected": is_selected
            })
            
            # 添加从状态节点到动作节点的边
            if is_selected:
                # 已选择的动作：绿色实线
                edges_data.append({
                    "from": state_node_id,
                    "to": action_node_id,
                    "label": "",
                    "type": "action_edge",
                    "status": "selected"
                })
            else:
                # 候选动作：黄色虚线
                edges_data.append({
                    "from": state_node_id,
                    "to": action_node_id,
                    "label": "",
                    "type": "action_edge",
                    "status": "candidate"
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
            border-left: 3px solid #ffc107;
            border-radius: 4px;
        }}
        .executed-candidate {{
            border-left-color: #28a745;
            background-color: #d4edda;
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
            }} else if (node.type === 'state') {{
                node.color = {{
                    background: '#e8f5e8',
                    border: '#388e3c'
                }};
                node.shape = 'box';
            }} else if (node.type === 'action') {{
                if (node.status === 'selected') {{
                    node.color = {{
                        background: '#d4edda',
                        border: '#28a745'
                    }};
                }} else {{
                    node.color = {{
                        background: '#fff3cd',
                        border: '#ffc107'
                    }};
                }}
                node.shape = 'ellipse';
                node.font = {{ size: 12 }};
            }}
        }});
        
        // 为不同类型的边设置不同样式
        edges.forEach(function(edge) {{
            if (edge.type === 'state_transition') {{
                edge.color = {{
                    color: '#007bff',
                    highlight: '#0056b3'
                }};
                edge.width = 2;
            }} else if (edge.type === 'action_edge' && edge.status === 'selected') {{
                edge.color = {{
                    color: '#28a745',
                    highlight: '#1e7e34'
                }};
                edge.width = 3;
            }} else if (edge.type === 'action_edge' && edge.status === 'candidate') {{
                edge.color = {{
                    color: '#ffc107',
                    highlight: '#e0a800'
                }};
                edge.dashes = [5, 5];
                edge.width = 1;
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
            }} else if (node.type === 'action') {{
                html += `<div class="node-info">`;
                html += `<p><strong>Type:</strong> ${{node.status === 'selected' ? 'Selected Action' : 'Candidate Action'}}</p>`;
                html += `<p><strong>Parent Step:</strong> ${{node.step}}</p>`;
                html += `<p><strong>Action:</strong> ${{node.action}}</p>`;
                if (node.thought) {{
                    html += `<p><strong>Thought:</strong> ${{node.thought}}</p>`;
                }}
                html += `<p><strong>Status:</strong> <span style="color: ${{node.status === 'selected' ? 'green' : 'orange'}};">${{node.status === 'selected' ? '✓ Selected' : '○ Candidate'}}</span></p>`;
                html += `</div>`;
            }} else if (node.type === 'state') {{
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
                        // 检查动作是否被选择
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
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def list_trajectory_files():
    """列出所有可用的轨迹文件"""
    trajectory_dir = Path("outputs/trajectory")
    if not trajectory_dir.exists():
        print("❌ 轨迹目录不存在: outputs/trajectory")
        return
    
    # 查找所有final文件（JSON和DOT）
    json_files = list(trajectory_dir.glob("*_final.json"))
    dot_files = list(trajectory_dir.glob("*_final.dot"))
    
    all_files = json_files + dot_files
    if not all_files:
        print("❌ 没有找到轨迹文件（_final.json 或 _final.dot）")
        return
    
    print("📁 可用的轨迹文件:")
    for i, file in enumerate(sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)):
        mtime = file.stat().st_mtime
        mtime_str = subprocess.run(["date", "-d", f"@{mtime}", "+%Y-%m-%d %H:%M:%S"], 
                                 capture_output=True, text=True).stdout.strip()
        print(f"  {i+1:2d}. {file.name} ({mtime_str})")

def main():
    parser = argparse.ArgumentParser(description="轨迹可视化工具")
    parser.add_argument("--file", "-f", help="指定轨迹文件路径（JSON或DOT文件）")
    parser.add_argument("--output", "-o", help="输出文件名（不含扩展名）")
    parser.add_argument("--open", action="store_true", help="生成后自动打开图片")
    parser.add_argument("--interactive", "-i", action="store_true", help="生成交互式HTML（支持点击节点显示截图）")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有可用的轨迹文件")
    
    args = parser.parse_args()
    
    if args.list:
        list_trajectory_files()
        return
    
    # 确定要处理的文件
    if args.file:
        trajectory_file = args.file
    else:
        trajectory_file = find_latest_trajectory()
        if not trajectory_file:
            return
    
    # 生成可视化
    success = generate_visualization(trajectory_file, args.output, args.open, args.interactive)
    
    if success:
        print("\n🎉 轨迹可视化完成！")
        print("💡 提示:")
        if args.interactive:
            print("   - 交互式HTML文件支持点击节点查看截图和详细信息")
            print("   - 绿色实线表示已选择的动作")
            print("   - 橙色虚线表示候选动作")
        else:
            print("   - SVG文件适合在浏览器中查看")
            print("   - PNG文件适合在文档中使用")
            print("   - 使用 --interactive 参数生成可交互的HTML")

if __name__ == "__main__":
    main()
