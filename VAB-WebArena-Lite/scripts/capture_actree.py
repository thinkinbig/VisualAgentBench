#!/usr/bin/env python3
"""Script to capture accessibility tree observation from a specified webpage."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

from beartype import beartype

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
        for line in chosen.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                import os
                os.environ[key] = value

# Load .env early
load_env_from_dotenv()

# Prefer importing directly from envs to avoid None fallback in package __init__
try:
    from browser_env.envs import ScriptBrowserEnv  # type: ignore
except Exception:
    # Fallback to package-level import (may be None if optional deps missing)
    from browser_env import ScriptBrowserEnv  # type: ignore
    if ScriptBrowserEnv is None:  # type: ignore
        raise ImportError(
            "ScriptBrowserEnv is unavailable. Ensure WebArena/browser_env dependencies are installed "
            "and that 'browser_env/envs.py' imports correctly."
        )
from browser_env.processors import TextObervationProcessor

# Setup logging
SCRIPT_DIR = Path(__file__).parent
LOG_FOLDER = SCRIPT_DIR / "log_files"
LOG_FOLDER.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("actree_capture_logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

@beartype
def config() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture accessibility tree observation from a webpage"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the webpage to capture"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file path for the observation (optional)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the browser UI (headless=False)"
    )
    parser.add_argument(
        "--wait_time",
        type=float,
        default=3.0,
        help="Wait time after page load in seconds"
    )
    parser.add_argument(
        "--viewport_width",
        type=int,
        default=1280,
        help="Viewport width"
    )
    parser.add_argument(
        "--viewport_height",
        type=int,
        default=2048,
        help="Viewport height"
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only capture elements in current viewport"
    )
    
    return parser.parse_args()

@beartype
def capture_actree_observation(
    url: str,
    render: bool = False,
    wait_time: float = 0.5,
    viewport_width: int = 1024,
    viewport_height: int = 768,
    current_viewport_only: bool = True,
    post_action: Optional[str] = None
) -> dict:
    """
    Capture accessibility tree observation from a specified webpage.
    
    Args:
        url: The URL to capture
        render: Whether to render the browser UI
        wait_time: Wait time after page load
        viewport_width: Viewport width
        viewport_height: Viewport height
        current_viewport_only: Only capture elements in current viewport
        
    Returns:
        Dictionary containing the observation data
    """
    # Create browser environment
    env = ScriptBrowserEnv(
        headless=not render,
        slow_mo=0,
        observation_type="accessibility_tree",
        current_viewport_only=current_viewport_only,
        viewport_size={"width": viewport_width, "height": viewport_height},
        save_trace_enabled=False,
        sleep_after_execution=0.0,
        captioning_fn=None,
    )
    
    logger.info(f"Browser environment created (headless: {not render})")
    logger.info(f"Capturing from URL: {url}")
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        # Navigate to the specified URL
        from browser_env.actions import create_playwright_action
        action = create_playwright_action(f'page.goto("{url}")')
        
        logger.info("Navigating to page...")
        obs, _, _, _, info = env.step(action)
        
        # Wait for page to load
        if wait_time > 0:
            logger.info(f"Waiting {wait_time} seconds for page to load...")
            time.sleep(wait_time)
        
        # Optionally perform a follow-up action (e.g., a rejected action) before capturing
        if post_action:
            try:
                logger.info(f"Executing post action: {post_action}")
                from browser_env.actions import create_playwright_action
                pw_action = create_playwright_action(post_action)
                obs, _, _, _, info = env.step(pw_action)
                if wait_time > 0:
                    time.sleep(min(wait_time, 0.5))
            except Exception as e:
                logger.error(f"Failed to execute post_action '{post_action}': {e}")

        # Get the observation (latest after goto (+ optional post_action) + wait)
        logger.info("Capturing accessibility tree...")
        
        # Extract the text observation
        text_observation = obs.get("text", "")
        # Clean up formatting: remove tab characters which can hinder downstream parsing
        if "\t" in text_observation:
            text_observation = text_observation.replace("\t", "")
        
        # Get page title and URL
        page_title = info.get("observation_metadata", {}).get("page_title", "Unknown")
        # Prefer the actual page URL from info['page'] to handle redirects
        page = info.get("page")
        current_url = getattr(page, "url", None) or info.get("observation_metadata", {}).get("url", url)
        
        result = {
            "url": current_url,
            "page_title": page_title,
            "text_observation": text_observation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "viewport_size": {"width": viewport_width, "height": viewport_height},
            "current_viewport_only": current_viewport_only
        }
        
        logger.info("Capture completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error capturing observation: {e}")
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass

@beartype
def save_observation(observation: dict, output_file: Optional[str] = None) -> None:
    """Save the observation to a file."""
    if output_file is None:
        # Generate default filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        url_part = observation["url"].replace("https://", "").replace("http://", "").replace("/", "_")[:50]
        output_file = f"actree_capture_{url_part}_{timestamp}.json"
    
    output_path = Path(output_file)
    
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(observation, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Observation saved to: {output_path}")

@beartype
def main() -> None:
    """Main function to capture accessibility tree observation."""
    args = config()
    
    try:
        # Capture the observation
        observation = capture_actree_observation(
            url=args.url,
            render=args.render,
            wait_time=args.wait_time,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            current_viewport_only=args.current_viewport_only
        )
        
        # Print the observation
        print("\n" + "="*80)
        print("ACCESSIBILITY TREE OBSERVATION")
        print("="*80)
        print(f"URL: {observation['url']}")
        print(f"Page Title: {observation['page_title']}")
        print(f"Timestamp: {observation['timestamp']}")
        print(f"Viewport: {observation['viewport_size']['width']}x{observation['viewport_size']['height']}")
        print(f"Current Viewport Only: {observation['current_viewport_only']}")
        print("-"*80)
        print("TEXT OBSERVATION:")
        print("-"*80)
        print(observation['text_observation'])
        print("="*80)
        
        # Save to file if requested
        if args.output_file:
            save_observation(observation, args.output_file)
        else:
            # Ask user if they want to save
            response = input("\nDo you want to save this observation to a file? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                filename = input("Enter filename (or press Enter for default): ").strip()
                if not filename:
                    filename = None
                save_observation(observation, filename)
        
    except Exception as e:
        logger.error(f"Failed to capture observation: {e}")
        return

if __name__ == "__main__":
    main()
