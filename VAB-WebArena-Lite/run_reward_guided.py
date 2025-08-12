#!/usr/bin/env python3
"""Script to run reward-guided trajectory search agent on WebArena Lite."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List

def load_env_from_dotenv() -> None:
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
        from dotenv import load_dotenv  # type: ignore
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
                os.environ[key] = value

# Load .env early so downstream imports see env vars
load_env_from_dotenv()

from agent import construct_agent

# Setup logging
LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/reward_guided_log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("reward_guided_logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reward-guided trajectory search agent on WebArena Lite"
    )
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="configs/reward_guided_agent.yaml",
        help="Path to the instruction file"
    )
    parser.add_argument(
        "--action_set_tag",
        type=str,
        default="id_accessibility_tree",
        choices=["playwright", "id_accessibility_tree", "som", "webrl_id"],
        help="Action set tag"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "google", "huggingface", "api", "finetune"],
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name"
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="gpt-4o",
        help="Reward model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples for action generation"
    )
    parser.add_argument(
        "--max_refinements",
        type=int,
        default=2,
        help="Maximum number of refinements"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        choices=["chat", "completion"],
        help="LLM mode"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=4096,
        help="Context length"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens"
    )
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Stop token"
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        default=2048,
        help="Max observation length"
    )
    parser.add_argument(
        "--max_retry",
        type=int,
        default=3,
        help="Max retry attempts"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="reward_guided",
        choices=["teacher_forcing", "prompt", "reward_guided"],
        help="Agent type"
    )
    parser.add_argument(
        "--test_config_file",
        type=str,
        required=True,
        help="Path to test configuration file"
    )
    parser.add_argument(
        "--planner_ip",
        type=str,
        default="",
        help="Planner IP address"
    )
    parser.add_argument(
        "--output_response",
        action="store_true",
        help="Output agent responses"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for console and file"
    )
    
    return parser.parse_args()

def main():
    args = config()
    # Update logger level based on flag
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
    
    # Load test configuration
    with open(args.test_config_file, "r") as f:
        test_config = json.load(f)
    
    # Resolve instruction path relative to this script if needed
    instr_path = Path(args.instruction_path)
    if not instr_path.exists():
        candidate = Path(__file__).parent / args.instruction_path
        if candidate.exists():
            args.instruction_path = str(candidate)

    # Create agent
    agent = construct_agent(args)
    
    # Set action set tag
    agent.set_action_set_tag(args.action_set_tag)
    
    # Reset agent if needed
    if hasattr(agent, 'reset'):
        agent.reset(args.test_config_file)
    
    # Extract task information
    task_name = test_config.get("task_name", "Unknown Task")
    intent = test_config.get("intent", "Complete the task")
    meta_data = test_config.get("meta_data", {})
    
    logger.info(f"Starting task: {task_name}")
    logger.info(f"Intent: {intent}")
    logger.info(f"Agent type: {args.agent_type}")
    logger.info(f"Policy model: {args.model}")
    logger.info(f"Reward model: {args.reward_model}")
    
    # Simulate trajectory (in real usage, this would come from the environment)
    trajectory = [
        {
            "observation": {
                "text": "Starting the task...",
                "image": None
            },
            "action": None,
            "info": {
                "page": type('Page', (), {'url': 'http://localhost:7770'})()
            }
        }
    ]
    
    # Add meta_data with action_history
    meta_data["action_history"] = ["No previous action"]
    
    # Generate action using reward-guided approach
    try:
        action = agent.next_action(
            trajectory=trajectory,
            intent=intent,
            meta_data=meta_data,
            images=None,
            output_response=args.output_response
        )
        
        logger.info(f"Generated action: {action}")
        logger.info(f"Action type: {action.get('action_type', 'Unknown')}")
        
        if args.output_response:
            print(f"\n=== Reward-Guided Agent Output ===")
            print(f"Task: {task_name}")
            print(f"Intent: {intent}")
            print(f"Generated Action: {action}")
            print(f"Raw Prediction: {action.get('raw_prediction', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error generating action: {e}")
        print(f"Error: {e}")
    
    logger.info("Task completed")

if __name__ == "__main__":
    main()
