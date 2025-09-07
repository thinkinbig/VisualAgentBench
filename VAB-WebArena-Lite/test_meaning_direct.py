#!/usr/bin/env python3
"""
Direct test of the modified trajectory_tree.py visualization.
"""

import json
import subprocess
import sys
import os

def test_meaning_visualization():
    """Test the modified visualization by directly calling the trajectory_tree module."""
    
    try:
        # Change to agent directory to avoid import conflicts
        original_dir = os.getcwd()
        os.chdir('agent')
        
        # Create a simple test script
        test_script = '''
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
    lines = graphviz_source.split("\\n")
    print("\\nFirst 15 lines of Graphviz source:")
    for i, line in enumerate(lines[:15]):
        print(f"  {i+1}: {line}")
        
except Exception as e:
    print(f"Error generating Graphviz: {e}")
    import traceback
    traceback.print_exc()
'''
        
        # Write and execute the test script
        with open('test_meaning_temp.py', 'w') as f:
            f.write(test_script)
        
        result = subprocess.run([sys.executable, 'test_meaning_temp.py'], 
                              capture_output=True, text=True, cwd='agent')
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Clean up
        os.remove('agent/test_meaning_temp.py')
        
        # Generate PNG if DOT file was created
        if os.path.exists('meaningful_trajectory.dot'):
            print("\nGenerating PNG visualization...")
            png_result = subprocess.run(['dot', '-Tpng', 'meaningful_trajectory.dot', '-o', 'meaningful_trajectory.png'], 
                                      capture_output=True, text=True)
            if png_result.returncode == 0:
                print("PNG visualization generated successfully: meaningful_trajectory.png")
            else:
                print(f"Error generating PNG: {png_result.stderr}")
        
        os.chdir(original_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_meaning_visualization()
