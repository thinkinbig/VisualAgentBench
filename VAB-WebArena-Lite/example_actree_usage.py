#!/usr/bin/env python3
"""
Simple example of using ActreeExtractor

This script shows the basic usage of the ActreeExtractor to get
WebArena-style action trees from web pages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from browser_env.actree_utils import get_actree_from_page
from playwright.sync_api import sync_playwright


def main():
    """Demonstrate basic actree extraction."""
    print("🌐 WebArena Action Tree Extractor Example")
    print("=" * 50)
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)  # Set to True for headless mode
        page = browser.new_page()
        
        try:
            # Navigate to a page
            print("📱 Navigating to website...")
            page.goto("https://www.theinfatuation.com/atlanta")
            page.wait_for_load_state("networkidle")
            
            # Extract the action tree
            print("🔍 Extracting action tree...")
            actree_string, obs_nodes_info = get_actree_from_page(page)
            
            # Display results
            print(f"\n✅ Successfully extracted action tree!")
            print(f"📊 Total nodes: {len(obs_nodes_info)}")
            print(f"📝 Tree length: {len(actree_string)} characters")
            
            # Show the action tree
            print("\n🌳 Action Tree (first 20 lines):")
            print("-" * 40)
            lines = actree_string.split('\n')[:20]
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}: {line}")
            
            total_lines = len(actree_string.split('\n'))
            if total_lines > 20:
                print(f"... and {total_lines - 20} more lines")
            
            # Show some node metadata
            print(f"\n📋 Node Metadata (first 3 nodes):")
            print("-" * 40)
            sample_nodes = list(obs_nodes_info.items())[:3]
            for node_id, metadata in sample_nodes:
                print(f"Node {node_id}:")
                print(f"  Text: {metadata['text'][:80]}...")
                print(f"  Role: {metadata['role']}")
                print(f"  Bounds: {metadata['union_bound']}")
                print()
            
            # Interactive mode - let user explore
            print("🔍 Interactive exploration:")
            print("You can now interact with the browser to see how the action tree changes.")
            print("Press Enter in this terminal to continue...")
            input()
            
            # Extract updated tree after potential user interaction
            print("\n🔄 Extracting updated action tree...")
            actree_updated, obs_nodes_updated = get_actree_from_page(page)
            
            print(f"📊 Updated tree has {len(obs_nodes_updated)} nodes")
            if len(obs_nodes_updated) != len(obs_nodes_info):
                print(f"🔄 Node count changed: {len(obs_nodes_info)} → {len(obs_nodes_updated)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n🧹 Closing browser...")
            browser.close()
    
    print("✅ Example completed!")


if __name__ == "__main__":
    main()
