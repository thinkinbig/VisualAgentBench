#!/usr/bin/env python3
"""Simple script to capture accessibility tree observation from a webpage."""

import argparse
import json
import time
from pathlib import Path

def load_env_from_dotenv() -> None:
    """Load environment variables from .env file."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    chosen: Path | None = None
    for p in candidates:
        if p.exists():
            chosen = p
            break
    if chosen is None:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=chosen, override=False)
    except Exception:
        import os
        for line in chosen.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# Load .env early
load_env_from_dotenv()

def simplify_observation(text_observation: str) -> str:
    """
    Simplify the accessibility tree observation to match the target format.
    
    Args:
        text_observation: Raw accessibility tree text
        
    Returns:
        Simplified observation in target format
    """
    lines = []
    
    # Define which element types to keep
    keep_types = {'link', 'button', 'textbox', 'combobox', 'checkbox', 'radio', 'select', 'input', 'StaticText'}
    
    for line in text_observation.split('\n'):
        line = line.strip()
        if not line or line.startswith('Tab') or line.startswith('RootWebArea'):
            continue
            
        # Skip lines that are just indentation or generic elements
        if line.startswith('\t') or 'generic' in line.lower():
            continue
            
        # Extract element information
        if '[' in line and ']' in line:
            # Find the ID
            start = line.find('[')
            end = line.find(']')
            if start != -1 and end != -1:
                element_id = line[start+1:end]
                
                # Extract element type and content
                parts = line[end+1:].strip().split(' ', 1)
                if len(parts) >= 2:
                    element_type = parts[0]
                    content_part = parts[1]
                    
                    # Only keep important element types
                    if element_type not in keep_types:
                        continue
                    
                    # Clean up the content - extract only the main text content
                    content = content_part
                    
                    # For buttons and links, extract only the visible text (first quoted string)
                    if element_type == 'button' or element_type == 'link':
                        # Find the first quote and extract content
                        quote_start = content.find("'")
                        if quote_start != -1:
                            quote_end = content.find("'", quote_start + 1)
                            if quote_end != -1:
                                content = content[quote_start + 1:quote_end]
                    elif element_type == 'textbox':
                        # For textbox, remove properties from content
                        if 'required:' in content:
                            content = content.split('required:')[0].strip().strip("'\"")
                        else:
                            content = content.strip("'\"")
                    else:
                        # For other elements, clean up quotes
                        content = content.strip("'\"")
                    
                    # Skip empty content except for links (keep link '' per reference)
                    if element_type != 'link' and (not content or content == ''):
                        continue
                    
                    # Build the simplified line
                    simplified_line = f"[{element_id}] {element_type} '{content}'"
                    
                    # Add properties based on element type
                    properties = []
                    
                    # For links, extract a single URL property; prefer explicit url='...'
                    if element_type == 'link':
                        extracted_url = ''
                        if "url='" in line:
                            url_start = line.find("url='")
                            if url_start != -1:
                                url_end = line.find("'", url_start + 5)
                                if url_end != -1:
                                    extracted_url = line[url_start + 5:url_end]
                        if not extracted_url and 'url:' in line:
                            colon_pos = line.find('url:')
                            if colon_pos != -1:
                                remainder = line[colon_pos + 4:].strip()
                                if remainder:
                                    token = remainder.split()[0].strip("'\" ,;")
                                    extracted_url = token
                        properties.append(f"url='{extracted_url}'")
                    
                    # For buttons, add relevant properties
                    if element_type == 'button':
                        if 'hasPopup:' in line and 'dialog' in line:
                            properties.append("hasPopup='dialog'")
                        if 'expanded:' in line:
                            if 'expanded: True' in line:
                                properties.append("expanded=True")
                                # Only add controls when expanded=True
                                if 'controls:' in line:
                                    controls_start = line.find('controls:')
                                    if controls_start != -1:
                                        controls_part = line[controls_start:].split()[1]
                                        properties.append(f"controls='{controls_part}'")
                            else:
                                properties.append("expanded=False")
                    
                    # For textbox, add focused property
                    if element_type == 'textbox' and 'focused' in line:
                        properties.append("focused")
                    
                    # For any element, add focused property if present
                    if 'focused: True' in line:
                        properties.append("focused")
                    
                    # Do not include required property for textbox in simplified output
                    
                    # Add properties to the line if any exist
                    if properties:
                        simplified_line += ", " + ", ".join(properties)
                    
                    lines.append(simplified_line)
    
    return '\n'.join(lines)

def capture_actree(url: str, wait_time: float = 2.0, render: bool = False, viewport_width: int = 1440, viewport_height: int = 900) -> str:
    """
    Capture accessibility tree observation from a webpage.
    
    Args:
        url: The URL to capture
        wait_time: Wait time after page load
        render: Whether to render the browser UI
        viewport_width: Viewport width (smaller = fewer elements)
        viewport_height: Viewport height (smaller = fewer elements)
        
    Returns:
        The text observation as a string in simplified format
    """
    from browser_env import ScriptBrowserEnv
    from browser_env.actions import create_playwright_action
    
    # Create browser environment - using smaller viewport to limit elements
    env = ScriptBrowserEnv(
        headless=not render,
        slow_mo=0,
        observation_type="accessibility_tree",
        current_viewport_only=True,  # Only capture elements in current viewport
        viewport_size={"width": viewport_width, "height": viewport_height},
        save_trace_enabled=False,
        sleep_after_execution=0.0,
        captioning_fn=None,
    )
    
    try:
        print(f"Navigating to: {url}")
        print(f"Using viewport: {viewport_width}x{viewport_height}")
        
        # Reset environment
        env.reset()
        
        # Navigate to the specified URL
        action = create_playwright_action(f'page.goto("{url}")')
        obs, _, _, _, info = env.step(action)
        
        # Wait for page to load
        if wait_time > 0:
            print(f"Waiting {wait_time} seconds for page to load...")
            time.sleep(wait_time)
        
        # Extract the text observation
        text_observation = obs.get("text", "")
        
        # Always simplify the output
        text_observation = simplify_observation(text_observation)
        
        return text_observation
        
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser(description="Capture accessibility tree from a webpage")
    parser.add_argument("url", help="URL to capture")
    parser.add_argument("--wait", type=float, default=2.0, help="Wait time after page load (default: 2.0)")
    parser.add_argument("--render", action="store_true", help="Show browser UI")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--viewport-width", type=int, default=1440, help="Viewport width (default: 1440)")
    parser.add_argument("--viewport-height", type=int, default=900, help="Viewport height (default: 900)")
    
    args = parser.parse_args()
    
    try:
        print("Starting accessibility tree capture...")
        print("Note: Only elements in current viewport will be captured (current_viewport_only=True)")
        print(f"Using viewport size: {args.viewport_width}x{args.viewport_height}")
        
        text_observation = capture_actree(
            args.url, 
            args.wait, 
            args.render, 
            args.viewport_width, 
            args.viewport_height
        )
        
        # Print the result
        print("\n" + "="*80)
        print("ACCESSIBILITY TREE OBSERVATION (TARGET FORMAT)")
        print("="*80)
        print(f"URL: {args.url}")
        print(f"Wait time: {args.wait}s")
        print(f"Render: {args.render}")
        print(f"Viewport: {args.viewport_width}x{args.viewport_height}")
        print("-"*80)
        print("TEXT OBSERVATION:")
        print("-"*80)
        print(text_observation)
        print("="*80)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = {
                "url": args.url,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "text_observation": text_observation,
                "viewport_size": {"width": args.viewport_width, "height": args.viewport_height},
                "current_viewport_only": True
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\nObservation saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
