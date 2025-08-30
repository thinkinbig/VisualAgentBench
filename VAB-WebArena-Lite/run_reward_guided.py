#!/usr/bin/env python3
"""Simplified script to run reward-guided agent with enhanced prompts."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional

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
                os.environ[key] = value

# Load .env early
load_env_from_dotenv()

from agent import construct_agent
from typing import Dict
from browser_env import (
    ScriptBrowserEnv,
    ActionTypes,
    Trajectory,
    StateInfo,
    create_stop_action,
)
from browser_env.helper_functions import get_action_description

# Setup logging
SCRIPT_DIR = Path(__file__).parent
LOG_FOLDER = SCRIPT_DIR / "log_files"
LOG_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = LOG_FOLDER / f"reward_guided_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("reward_guided_logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

@beartype
def config() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run reward-guided agent with enhanced prompts"
    )
    parser.add_argument(
        "--test_config_file",
        type=str,
        default=None,
        help="Path to single test configuration file (if omitted, use --task_id from --task_pool)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the browser UI"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum number of steps"
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
        help="Logging level"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Task ID to select from task pool when --test_config_file is not provided"
    )
    parser.add_argument(
        "--task_pool",
        type=str,
        default=str(SCRIPT_DIR / "config_files/wa/test_webarena_lite_webpilot.raw.json"),
        help="Path to a task pool JSON (list of tasks) to select by --task_id"
    )
    
    return parser.parse_args()

@beartype
def main() -> None:
    """Main function to run the reward-guided agent."""
    args = config()
    
    # Update logger level
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)

    # Load test configuration
    selected_config_file: Path | None = None
    test_config: dict
    if args.test_config_file:
        if not Path(args.test_config_file).exists():
            logger.error(f"Test config file not found: {args.test_config_file}")
            return
        selected_config_file = Path(args.test_config_file)
        with open(selected_config_file, "r") as f:
            test_config = json.load(f)
    else:
        pool_path = Path(args.task_pool)
        if not pool_path.exists():
            logger.error(f"Task pool file not found: {pool_path}")
            return
        try:
            pool = json.loads(pool_path.read_text())
        except Exception as e:
            logger.error(f"Failed to read task pool: {e}")
            return
        matched = None
        for item in pool:
            try:
                if int(item.get("task_id")) == int(args.task_id):
                    matched = item
                    break
            except Exception:
                continue
        if matched is None:
            logger.error(f"Task id {args.task_id} not found in pool {pool_path}")
            return
        # If storage_state path is provided but missing, null it to avoid login dependency
        storage_state = matched.get("storage_state")
        if storage_state and not Path(storage_state).exists():
            matched["storage_state"] = None
        # Materialize to a temp single-task config file
        out_dir = SCRIPT_DIR / "config_files/wa"
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_config_file = out_dir / f"auto_task_{args.task_id}.json"
        test_config = matched
    
    # Resolve placeholder start_url like "__SHOPPING_ADMIN__" to actual URL
    try:
        placeholder_map: Dict[str, str] = {
            "__SHOPPING_ADMIN__": "http://localhost:7780/admin",
            "__SHOPPING__": "http://localhost:7770",
            "__REDDIT__": "http://reddit.com",
            "__GITLAB__": "http://gitlab.com",
            "__WIKIPEDIA__": "http://wikipedia.org",
            "__MAP__": "http://openstreetmap.org",
            "__HOMEPAGE__": "http://homepage.com",
        }
        su = test_config.get("start_url")
        if isinstance(su, str) and su:
            parts = [p.strip() for p in su.split("|AND|")]
            resolved_parts = [placeholder_map.get(p.strip(), p.strip()) for p in parts]
            resolved = " |AND| ".join(resolved_parts)
            test_config["start_url"] = resolved
    except Exception:
        pass
    
    # Persist possibly-updated config for environment consumption
    try:
        if selected_config_file is None:
            selected_config_file = SCRIPT_DIR / "config_files/wa/auto_task_runtime.json"
        selected_config_file.write_text(json.dumps(test_config, indent=2))
    except Exception:
        pass
    
    # Extract task information
    task_name = f"Task {test_config.get('task_id', 'Unknown')}"
    intent = test_config.get("intent", "Complete the task")
    meta_data = test_config.get("meta_data", {})
    start_url = test_config.get("start_url")
    
    logger.info(f"Starting task: {task_name}")
    logger.info(f"Intent: {intent}")
    if start_url:
        logger.info(f"Start URL: {start_url}")

    # Load agent configuration from file (relative to script dir)
    config_path = SCRIPT_DIR / "configs/reward_guided_agent.json"
    if not config_path.exists():
        logger.error(f"Agent config file not found: {config_path}")
        return
        
    with open(config_path, "r") as f:
        agent_config = json.load(f)
    
    # Create agent with configuration from file
    from types import SimpleNamespace
    agent_args = SimpleNamespace(
        agent_type="reward_guided",
        instruction_path=str(config_path),
        provider="openai",  # Default provider, will be overridden by config
        model="gpt-4o-mini",  # Default model, will be overridden by config
        mode="chat",
        temperature=agent_config.get("temperature", 1.0),
        top_p=agent_config.get("top_p", 0.9),
        context_length=4096,
        max_tokens=agent_config.get("policy_lm_config", {}).get("gen_config", {}).get("max_tokens", 512),
        stop_token=None,
        max_obs_length=2048,
        max_retry=3,
        model_endpoint=None,
    )
    
    try:
        agent = construct_agent(agent_args)
        logger.info("Agent initialized successfully")
        
        # Log model information
        try:
            policy_model = getattr(getattr(agent, "policy_lm_config", None), "model", None)
            reward_model = getattr(getattr(agent, "reward_lm_config", None), "model", None)
            if policy_model:
                logger.info(f"Policy model: {policy_model}")
            if reward_model:
                logger.info(f"Reward model: {reward_model}")
        except Exception:
            pass
            
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return

    # Reset agent if needed
    if hasattr(agent, 'reset'):
        agent.reset(str(selected_config_file))
    
    # Create browser environment
    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=0,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 2048},
        save_trace_enabled=True,
        sleep_after_execution=0.0,
        captioning_fn=None,
    )
    logger.info(f"Browser environment created (headless: {not args.render})")

    try:
        # Reset environment
        obs, info = env.reset(options={"config_file": str(selected_config_file)})
        state_info: StateInfo = {"observation": obs, "info": info}
        trajectory: Trajectory = [state_info]
        meta_data["action_history"] = ["None"]

        step_idx = 0
        while step_idx < args.max_steps:
            logger.debug(f"=== Action Step {step_idx + 1} ===")
            
            # Generate next action
            try:
                # Add discovery context if available
                if hasattr(agent, 'get_discovery_context'):
                    discovery_context = agent.get_discovery_context()
                    if discovery_context:
                        discovery_str = "\n\nPrevious Discoveries:\n"
                        for entry in discovery_context[-3:]:  # Last 3 discoveries
                            discovery_str += f"- {entry.get('message', '')}\n"
                        enhanced_intent = intent + discovery_str
                    else:
                        enhanced_intent = intent
                else:
                    enhanced_intent = intent
                
                action = agent.next_action(
                    trajectory=trajectory,
                    intent=enhanced_intent,
                    meta_data=meta_data,
                    images=None,
                    output_response=args.output_response,
                )
            except Exception as e:
                logger.error(f"Error generating action at step {step_idx}: {e}")
                action = create_stop_action(f"ERROR: {str(e)}")

            logger.info(f"Generated action: {action}")
            # Per-step summary: concise action + reward score
            try:
                action_type = action.get("action_type")
                if action_type is not None:
                    action_name = str(action_type).split(".")[-1].lower()
                else:
                    action_name = "unknown"
                element_id = action.get("element_id", "")
                concise = f"{action_name} [{element_id}]" if element_id else action_name
                reward_score = action.get("reward_score")
                if reward_score is not None:
                    logger.info(f"Step {step_idx + 1} chosen: {concise} (score={reward_score})")
                else:
                    logger.info(f"Step {step_idx + 1} chosen: {concise}")
            except Exception:
                pass
            trajectory.append(action)

            # Handle send_msg actions
            if action.get("action_type") == ActionTypes.SEND_MESSAGE_TO_USER:
                message = action.get("answer", "")
                logger.info(f"=== SEND_MSG_TO_USER DISCOVERY ===\n{message}\n")
                if args.output_response:
                    print(f"\n=== New Discovery ===\n{message}\n")

            # Get action description for history
            try:
                action_str = get_action_description(
                    action,
                    state_info["info"].get("observation_metadata", {}),
                    action_set_tag=getattr(agent, "action_set_tag", "id_accessibility_tree"),
                    prompt_constructor=getattr(agent, "prompt_constructor", None),
                )
            except Exception:
                action_str = str(action)

            meta_data["action_history"].append(action_str)

            # Check for stop action
            if action["action_type"] == ActionTypes.STOP:
                final_answer = action.get("answer", "")
                logger.info("Received STOP action. Terminating.")
                if final_answer:
                    logger.info(f"Final answer: {final_answer}")
                    if args.output_response:
                        print(f"\n=== Final Answer ===\n{final_answer}\n")
                break

            # Step environment
            obs, _, terminated, _, info = env.step(action)
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)

            if terminated:
                logger.info("Environment signaled termination.")
                trajectory.append(create_stop_action(""))
                break

            step_idx += 1
            
        # Save trace
        try:
            trace_path = Path(LOG_FILE_NAME).with_name(Path(LOG_FILE_NAME).stem + "_trace.zip")
            env.save_trace(trace_path)
            logger.info(f"Saved Playwright trace to {trace_path}")
        except Exception:
            pass
            
        logger.info("Task completed successfully")
        
    except Exception as e:
        logger.error(f"Task failed: {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
