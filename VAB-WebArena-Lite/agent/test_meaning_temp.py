
import json
from trajectory_tree import TrajectoryTree

# Load trajectory from JSON file
with open('../outputs/trajectory/trajectory_148_20250907_214347_final.json', 'r', encoding='utf-8') as f:
    json_content = f.read()

# Create trajectory tree from JSON
trajectory_tree = TrajectoryTree.from_json(json_content)
print(f"Successfully loaded trajectory tree with {len(trajectory_tree.nodes)} nodes")

# Get selected nodes to show examples
selected_nodes = trajectory_tree.get_selected_nodes()
print(f"Number of selected nodes: {len(selected_nodes)}")
for selected in selected_nodes:
    print(f"  Selected: {selected.node_id}")
    print(f"    Action: {selected.action}")
    print(f"    Meaning: {selected.meaning}")
    print()

# Test Graphviz visualization
print("Generating Graphviz visualization with meaningful labels...")
try:
    graphviz_source = trajectory_tree.to_graphviz("meaningful_trajectory")
    print("Graphviz source generated successfully!")
    
    # Save to file for inspection
    with open("../meaningful_trajectory.dot", "w") as f:
        f.write(graphviz_source)
    print("Graphviz source saved to meaningful_trajectory.dot")
    
    # Show first few lines
    lines = graphviz_source.split("\n")
    print("\nFirst 15 lines of Graphviz source:")
    for i, line in enumerate(lines[:15]):
        print(f"  {i+1}: {line}")
        
except Exception as e:
    print(f"Error generating Graphviz: {e}")
    import traceback
    traceback.print_exc()
