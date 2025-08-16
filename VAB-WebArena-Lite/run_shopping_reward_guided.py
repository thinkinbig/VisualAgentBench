#!/usr/bin/env python3
"""Script to run reward-guided trajectory search agent on WebArena Lite."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import BaseModel, ValidationError, field_validator
from beartype import beartype

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
from browser_env import (
    ScriptBrowserEnv,
    ActionTypes,
    Trajectory,
    StateInfo,
    create_stop_action,
)
from browser_env.helper_functions import get_action_description

# Setup logging (write relative to this script's directory)
SCRIPT_DIR = Path(__file__).parent
LOG_FOLDER = SCRIPT_DIR / "log_files"
LOG_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = LOG_FOLDER / f"reward_guided_log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

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

class RunConfig(BaseModel):
    instruction_path: str
    test_config_file: str
    # Real env options (kept as runtime controls)
    real_env: bool = True
    render: bool = True
    observation_type: Literal[
        "accessibility_tree",
        "accessibility_tree_with_captioner",
        "html",
        "image",
        "image_som",
        "webrl",
    ] = "accessibility_tree"
    viewport_width: int = 1280
    viewport_height: int = 2048
    sleep_after_execution: float = 0.0
    max_steps: int = 30
    planner_ip: str = ""
    output_response: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("instruction_path")
    def _validate_instruction_exists(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"instruction_path not found: {v}")
        return v

    @field_validator("test_config_file")
    def _validate_test_config_exists(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"test_config_file not found: {v}")
        return v

    @field_validator("viewport_width", "viewport_height", "max_steps")
    def _validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("sleep_after_execution")
    def _validate_sleep(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("sleep_after_execution must be >= 0.0")
        return v

@beartype
def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reward-guided trajectory search agent on WebArena Lite"
    )
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="configs/reward_guided_agent.json",
        help="Path to the instruction file"
    )
    # Removed model/sampling args; now configured via JSON. This script runs reward-guided agent.
    parser.add_argument(
        "--test_config_file",
        type=str,
        required=True,
        help="Path to test configuration file"
    )
    # Real env options
    parser.add_argument(
        "--real_env",
        action="store_true",
        default=True,
        help="Run against real browser env instead of offline mock (default: True)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the browser UI"
    )
    parser.add_argument(
        "--observation_type",
        type=str,
        default="accessibility_tree",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
            "webrl",
        ],
        help="Observation type for real env"
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=30)
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

@beartype
def main() -> None:
    args = config()
    # Resolve instruction path relative to this script if needed before validation
    instr_path = Path(args.instruction_path)
    if not instr_path.exists():
        candidate = Path(__file__).parent / args.instruction_path
        if candidate.exists():
            args.instruction_path = str(candidate)

    # Validate and normalize with Pydantic
    try:
        validated = RunConfig(**vars(args))
    except ValidationError as e:
        logger.error("Configuration validation failed: %s", e)
        print(f"Configuration error:\n{e}")
        return
    cfg = validated

    # Update logger level based on validated config
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)

    # Load test configuration
    with open(cfg.test_config_file, "r") as f:
        test_config = json.load(f)
    # Normalize start_url preference: CLI JSON runtime.start_url > test_config.start_url > legacy meta_data.start_url
    start_url_from_test = test_config.get("start_url") or test_config.get("meta_data", {}).get("start_url")

    # Apply runtime overrides from instruction JSON (if provided)
    try:
        from agent.config_schema import load_and_validate_instruction
        inst = load_and_validate_instruction(cfg.instruction_path)
        if inst.runtime is not None:
            rt = inst.runtime
            # Override known runtime fields
            # Precedence: CLI > JSON. Only apply JSON when CLI is unset/False.
            if rt.real_env is not None and cfg.real_env is False:
                cfg.real_env = rt.real_env
            if rt.render is not None and cfg.render is False:
                cfg.render = rt.render
            # runtime.start_url can override test_config if present
            if getattr(rt, "start_url", None):
                start_url_from_test = rt.start_url  # type: ignore[assignment]
            if rt.observation_type is not None:
                cfg.observation_type = rt.observation_type  # type: ignore[assignment]
            if rt.viewport_width is not None:
                cfg.viewport_width = rt.viewport_width  # type: ignore[assignment]
            if rt.viewport_height is not None:
                cfg.viewport_height = rt.viewport_height  # type: ignore[assignment]
            if rt.sleep_after_execution is not None:
                cfg.sleep_after_execution = rt.sleep_after_execution  # type: ignore[assignment]
            if rt.max_steps is not None:
                cfg.max_steps = rt.max_steps  # type: ignore[assignment]
            if rt.planner_ip is not None:
                cfg.planner_ip = rt.planner_ip  # type: ignore[assignment]
            if rt.output_response is not None and cfg.output_response is False:
                cfg.output_response = rt.output_response  # type: ignore[assignment]
            if rt.log_level is not None:
                cfg.log_level = rt.log_level  # type: ignore[assignment]
    except Exception:
        pass

    # Create agent with required constructor fields via a lightweight namespace
    from types import SimpleNamespace
    agent_args = SimpleNamespace(**vars(cfg))
    agent_args.agent_type = "reward_guided"
    # Provide minimal fields expected by construct_llm_config if not present
    if not hasattr(agent_args, "provider"):
        agent_args.provider = "openai"
    if not hasattr(agent_args, "model"):
        agent_args.model = "gpt-4o-mini"
    if not hasattr(agent_args, "mode"):
        agent_args.mode = "chat"
    # Default generation params (will be overridden by JSON if provided inside agent)
    defaults = dict(
        temperature=1.0,
        top_p=0.9,
        context_length=4096,
        max_tokens=512,
        stop_token=None,
        max_obs_length=2048,
        max_retry=3,
        model_endpoint=None,
    )
    for k, v in defaults.items():
        if not hasattr(agent_args, k):
            setattr(agent_args, k, v)
    agent = construct_agent(agent_args)
    
    # Action set tag is configured by JSON/constructor
    
    # Reset agent if needed
    if hasattr(agent, 'reset'):
        agent.reset(cfg.test_config_file)
    
    # Extract task information
    task_name = test_config.get("task_name", "Unknown Task")
    intent = test_config.get("intent", "Complete the task")
    meta_data = test_config.get("meta_data", {})
    
    logger.info(f"Starting task: {task_name}")
    logger.info(f"Intent: {intent}")
    try:
        policy_model = getattr(getattr(agent, "policy_lm_config", None), "model", None) or getattr(getattr(agent, "lm_config", None), "model", None)
        reward_model = getattr(getattr(agent, "reward_lm_config", None), "model", None)
        logger.info(f"Agent type: reward_guided")
        if policy_model:
            logger.info(f"Policy model: {policy_model}")
        if reward_model:
            logger.info(f"Reward model: {reward_model}")
    except Exception:
        logger.info("Agent initialized")
    
    # Always use real environment mode
    logger.info("Running in real environment mode - using actual browser")
    
    # Real env mode
    env = ScriptBrowserEnv(
        headless=not cfg.render,
        slow_mo=0,
        observation_type=cfg.observation_type,
        current_viewport_only=True,
        viewport_size={"width": cfg.viewport_width, "height": cfg.viewport_height},
        save_trace_enabled=True,
        sleep_after_execution=cfg.sleep_after_execution,
        captioning_fn=None,
    )
    logger.info(f"Real env: {cfg.real_env}, render: {cfg.render}, headless: {not cfg.render}, observation_type: {cfg.observation_type}")
    if start_url_from_test:
        logger.info(f"Start URL: {start_url_from_test}")

    try:
        # Reset env with the provided test config file
        # Pass through config file; ScriptBrowserEnv will read start_url from it
        obs, info = env.reset(options={"config_file": cfg.test_config_file})
        state_info: StateInfo = {"observation": obs, "info": info}
        trajectory: Trajectory = [state_info]
        meta_data["action_history"] = ["None"]

        # Removed: initial HTML snapshot saving

        step_idx = 0
        while step_idx < cfg.max_steps:
            # Generate next action
            try:
                action = agent.next_action(
                    trajectory=trajectory,
                    intent=intent,
                    meta_data=meta_data,
                    images=None,
                    output_response=cfg.output_response,
                )
            except Exception as e:
                logger.error(f"Error generating action at step {step_idx}: {e}")
                action = create_stop_action(f"ERROR: {str(e)}")

            logger.info(f"Generated action: {action}")
            trajectory.append(action)

            # Render-friendly action string for history
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

            if action["action_type"] == ActionTypes.STOP:
                final_answer = action.get("answer", "")
                logger.info("Received STOP action. Terminating.")
                if final_answer:
                    logger.info(f"Final answer (send_msg_to_user): {final_answer}")
                    if cfg.output_response:
                        print(f"\n=== Final Answer ===\n{final_answer}\n")
                # Removed: step_0 HTML snapshot saving
                break

            # Step environment
            obs, _, terminated, _, info = env.step(action)
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)

            # Removed: per-step HTML snapshot saving

            if terminated:
                logger.info("Environment signaled termination.")
                trajectory.append(create_stop_action(""))
                break

            step_idx += 1
        # Save Playwright trace for visual inspection
        try:
            trace_path = Path(LOG_FILE_NAME).with_name(Path(LOG_FILE_NAME).stem + "_trace.zip")
            env.save_trace(trace_path)
            logger.info(f"Saved Playwright trace to {trace_path}")
        except Exception:
            pass
        logger.info("Task completed")
    finally:
        try:
            env.close()
        except Exception:
            pass



if __name__ == "__main__":
    main()
