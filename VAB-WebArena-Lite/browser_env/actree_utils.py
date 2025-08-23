"""
Action Tree (actree) utilities for WebArena

This module provides tools to extract and process accessibility trees
in the WebArena format, which is based on Chrome DevTools Protocol
Accessibility API rather than Playwright's accessibility.snapshot().
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from playwright.sync_api import Page, CDPSession
import pandas as pd
from .utils import AccessibilityTree, AccessibilityTreeNode, BrowserConfig
from .constants import IN_VIEWPORT_RATIO_THRESHOLD


class ActreeExtractor:
    """
    Extract and process accessibility trees in WebArena format.
    
    This class provides methods to:
    1. Extract raw accessibility tree using CDP
    2. Process and filter nodes
    3. Generate WebArena-style actree with unique IDs
    4. Convert to various output formats
    """
    
    def __init__(self, page: Page):
        self.page = page
        self.client: Optional[CDPSession] = None
        
    def __enter__(self):
        self.client = self.page.context.new_cdp_session(self.page)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.detach()
    
    def get_browser_config(self) -> BrowserConfig:
        """Get browser configuration including viewport and scroll information."""
        win_upper_bound = self.page.evaluate("window.pageYOffset")
        win_left_bound = self.page.evaluate("window.pageXOffset")
        win_width = self.page.evaluate("window.screen.width")
        win_height = self.page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        
        # Force device pixel ratio to 1.0 for WebArena compatibility
        device_pixel_ratio = 1.0
        
        return {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }
    
    def get_bounding_client_rect(self, backend_node_id: str) -> Dict[str, Any]:
        """Get bounding rectangle for a DOM node using CDP."""
        if not self.client:
            raise RuntimeError("CDP client not initialized. Use context manager.")
            
        try:
            remote_object = self.client.send(
                "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
            )
            remote_object_id = remote_object["object"]["objectId"]
            response = self.client.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": remote_object_id,
                    "functionDeclaration": """
                        function() {
                            if (this.nodeType == 3) {
                                var range = document.createRange();
                                range.selectNode(this);
                                var rect = range.getBoundingClientRect().toJSON();
                                range.detach();
                                return rect;
                            } else {
                                return this.getBoundingClientRect().toJSON();
                            }
                        }
                    """,
                    "returnByValue": True,
                },
            )
            return response
        except Exception as e:
            return {"result": {"subtype": "error"}}
    
    def get_element_in_viewport_ratio(
        self,
        elem_left_bound: float,
        elem_top_bound: float,
        width: float,
        height: float,
        config: BrowserConfig,
    ) -> float:
        """Calculate what percentage of an element is visible in the viewport."""
        elem_right_bound = elem_left_bound + width
        elem_lower_bound = elem_top_bound + height
        
        # Calculate intersection
        left_intersection = max(elem_left_bound, config["win_left_bound"])
        right_intersection = min(elem_right_bound, config["win_right_bound"])
        top_intersection = max(elem_top_bound, config["win_upper_bound"])
        bottom_intersection = min(elem_lower_bound, config["win_lower_bound"])
        
        if left_intersection >= right_intersection or top_intersection >= bottom_intersection:
            return 0.0
        
        intersection_area = (right_intersection - left_intersection) * (bottom_intersection - top_intersection)
        element_area = width * height
        
        return intersection_area / element_area if element_area > 0 else 0.0
    
    def extract_raw_accessibility_tree(self) -> AccessibilityTree:
        """Extract raw accessibility tree using CDP Accessibility.getFullAXTree."""
        if not self.client:
            raise RuntimeError("CDP client not initialized. Use context manager.")
            
        accessibility_tree: AccessibilityTree = self.client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]
        
        # Remove duplicate nodes
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        
        return _accessibility_tree
    
    def process_accessibility_tree(
        self, 
        accessibility_tree: AccessibilityTree,
        current_viewport_only: bool = True
    ) -> AccessibilityTree:
        """
        Process accessibility tree by:
        1. Adding bounding rectangles
        2. Filtering by viewport visibility
        3. Cleaning up removed nodes
        """
        config = self.get_browser_config()
        nodeid_to_cursor = {}
        
        # Add cursor mapping and bounding rectangles
        for cursor, node in enumerate(accessibility_tree):
            nodeid_to_cursor[node["nodeId"]] = cursor
            
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
                
            backend_node_id = str(node["backendDOMNodeId"])
            
            if node["role"]["value"] == "RootWebArea":
                # Root web area is always in viewport
                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(backend_node_id)
                if response.get("result", {}).get("subtype", "") == "error":
                    node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    node["union_bound"] = [x, y, width, height]
        
        # Filter nodes by viewport visibility if requested
        if current_viewport_only:
            def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
                """Remove a node from the tree and reparent its children."""
                nodeid = node["nodeId"]
                node_cursor = nodeid_to_cursor[nodeid]
                parent_nodeid = node["parentId"]
                children_nodeids = node["childIds"]
                parent_cursor = nodeid_to_cursor[parent_nodeid]
                
                # Update parent's children
                assert accessibility_tree[parent_cursor].get("parentId", "Root") is not None
                index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
                accessibility_tree[parent_cursor]["childIds"].pop(index)
                
                # Insert children in the same location
                for child_nodeid in children_nodeids:
                    accessibility_tree[parent_cursor]["childIds"].insert(index, child_nodeid)
                    index += 1
                
                # Update children's parent
                for child_nodeid in children_nodeids:
                    child_cursor = nodeid_to_cursor[child_nodeid]
                    accessibility_tree[child_cursor]["parentId"] = parent_nodeid
                
                # Mark as removed
                accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"
            
            # Apply viewport filtering
            for node in accessibility_tree:
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue
                
                [x, y, width, height] = node["union_bound"]
                
                # Skip invisible nodes
                if width == 0 or height == 0:
                    remove_node_in_graph(node)
                    continue
                
                # Check viewport visibility ratio
                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )
                
                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)
            
            # Remove marked nodes
            accessibility_tree = [
                node for node in accessibility_tree
                if node.get("parentId", "Root") != "[REMOVED]"
            ]
        
        return accessibility_tree
    
    def generate_webarena_actree(
        self, 
        accessibility_tree: AccessibilityTree,
        include_metadata: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate WebArena-style actree matching the target format.
        
        Returns:
            Tuple of (actree_string, obs_nodes_info)
        """
        obs_nodes_info = {}
        actree_lines = []
        next_id = 200  # Start from 200 to match target format
        
        def extract_node_properties(node: AccessibilityTreeNode) -> List[str]:
            """Extract relevant properties for the node."""
            properties = []
            
            # Extract properties from the properties array
            for prop in node.get("properties", []):
                try:
                    prop_name = prop["name"]
                    prop_value = prop["value"]["value"]
                    
                    # Include important properties
                    if prop_name in ["focusable", "editable", "readonly", "hasPopup", "expanded", "controls", "focused"]:
                        if prop_name == "hasPopup" and prop_value == "dialog":
                            properties.append(f"hasPopup='dialog'")
                        elif prop_name == "expanded":
                            properties.append(f"expanded={prop_value}")
                        elif prop_name == "controls":
                            properties.append(f"controls='{prop_value}'")
                        elif prop_name == "focused":
                            if prop_value:
                                properties.append("focused")
                        else:
                            properties.append(f"{prop_name}={prop_value}")
                except (KeyError, TypeError):
                    pass
            
            # Extract URL for links
            if node["role"]["value"] == "link":
                try:
                    # Try to get href from properties or attributes
                    url_found = False
                    for prop in node.get("properties", []):
                        if prop["name"] == "url":
                            url = prop["value"]["value"]
                            if url and url != "null":
                                properties.append(f"url='{url}'")
                                url_found = True
                            break
                    
                    # If no URL found in properties, try to get it from DOM attributes
                    if not url_found and "backendDOMNodeId" in node:
                        try:
                            # Use CDP to get the actual href attribute
                            if self.client:
                                remote_object = self.client.send(
                                    "DOM.resolveNode", {"backendNodeId": int(node["backendDOMNodeId"])}
                                )
                                remote_object_id = remote_object["object"]["objectId"]
                                href_response = self.client.send(
                                    "Runtime.callFunctionOn",
                                    {
                                        "objectId": remote_object_id,
                                        "functionDeclaration": """
                                            function() {
                                                if (this.tagName === 'A' && this.href) {
                                                    return this.href;
                                                }
                                                return null;
                                            }
                                        """,
                                        "returnByValue": True,
                                    },
                                )
                                
                                if href_response.get("result", {}).get("value"):
                                    href = href_response["result"]["value"]
                                    if href and href != "null":
                                        properties.append(f"url='{href}'")
                        except Exception:
                            pass
                            
                except (KeyError, TypeError):
                    pass
            
            return properties
        
        def process_node(node: AccessibilityTreeNode) -> Optional[str]:
            """Process a single node and return its string representation."""
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                
                # Skip certain node types
                if role in ["StaticText", "generic", "img", "list", "strong", "paragraph", 
                           "banner", "navigation", "Section", "LabelText", "Legend", "listitem"]:
                    if not name.strip():
                        return None
                
                # Generate node string
                node_id = str(next_id)
                node_str = f"[{node_id}] {role} {repr(name)}"
                
                # Add properties
                properties = extract_node_properties(node)
                if properties:
                    node_str += ", " + ", ".join(properties)
                
                # Store metadata
                if include_metadata:
                    obs_nodes_info[node_id] = {
                        "backend_id": node.get("backendDOMNodeId"),
                        "union_bound": node.get("union_bound"),
                        "text": node_str,
                        "role": role,
                        "name": name,
                        "node_id": node["nodeId"]
                    }
                
                return node_str
                
            except Exception as e:
                return None
        
        def collect_interactive_nodes(nodes: AccessibilityTree) -> List[str]:
            """Collect all interactive nodes in a flat list."""
            interactive_nodes = []
            
            def process_node_recursive(node: AccessibilityTreeNode) -> None:
                nonlocal next_id
                
                # Process current node
                node_str = process_node(node)
                if node_str:
                    interactive_nodes.append(node_str)
                    next_id += 1
                
                # Process children recursively
                for child_id in node.get("childIds", []):
                    child_node = next((n for n in nodes if n["nodeId"] == child_id), None)
                    if child_node:
                        process_node_recursive(child_node)
            
            # Process all nodes
            for node in nodes:
                process_node_recursive(node)
            
            return interactive_nodes
        
        # Collect all interactive nodes
        interactive_nodes = collect_interactive_nodes(accessibility_tree)
        
        # Join into single string
        actree_string = "\n".join(interactive_nodes)
        
        return actree_string, obs_nodes_info
    
    def get_actree(
        self, 
        current_viewport_only: bool = True,
        include_metadata: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Main method to get complete WebArena-style actree.
        
        Args:
            current_viewport_only: Whether to filter nodes outside viewport
            include_metadata: Whether to include node metadata
            
        Returns:
            Tuple of (actree_string, obs_nodes_info)
        """
        # Extract raw tree
        raw_tree = self.extract_raw_accessibility_tree()
        
        # Process and filter tree
        processed_tree = self.process_accessibility_tree(
            raw_tree, 
            current_viewport_only
        )
        
        # Generate final actree
        actree_string, obs_nodes_info = self.generate_webarena_actree(
            processed_tree, 
            include_metadata
        )
        
        return actree_string, obs_nodes_info


def get_actree_from_page(
    page: Page, 
    current_viewport_only: bool = True,
    include_metadata: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to get actree from a Playwright page.
    
    Args:
        page: Playwright Page object
        current_viewport_only: Whether to filter nodes outside viewport
        include_metadata: Whether to include node metadata
        
    Returns:
        Tuple of (actree_string, obs_nodes_info)
    """
    with ActreeExtractor(page) as extractor:
        return extractor.get_actree(current_viewport_only, include_metadata)


def save_actree_to_file(
    actree_string: str, 
    obs_nodes_info: Dict[str, Any], 
    filepath: str
) -> None:
    """Save actree and metadata to a JSON file."""
    data = {
        "actree": actree_string,
        "metadata": obs_nodes_info,
        "timestamp": str(pd.Timestamp.now())
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_actree_from_file(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """Load actree and metadata from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["actree"], data["metadata"]


# Example usage and testing
if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    
    def example_usage():
        """Example of how to use the ActreeExtractor."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            
            # Navigate to a page
            page.goto("https://example.com")
            page.wait_for_load_state("networkidle")
            
            # Extract actree
            actree_string, obs_nodes_info = get_actree_from_page(page)
            
            print("=== WebArena Action Tree ===")
            print(actree_string)
            print("\n=== Node Metadata ===")
            print(json.dumps(obs_nodes_info, indent=2))
            
            # Save to file
            save_actree_to_file(actree_string, obs_nodes_info, "example_actree.json")
            
            browser.close()
    
    example_usage()
