#!/usr/bin/env python3
"""
Interactive step tool:
- Default loads dataset: data/WebPRMCollection_preference_pair_8k_test.json
- Step 1: Ask for task_id and step_id
- Step 2: Read current_url, capture accessibility-tree observation and print it
- Step 3: Ask user to input chosen action
- Step 4: Build thought prompt from intent + thought_history[:step_id] + current page state
- Step 5: Generate thought with GPT-4o
- Step 6: Produce next-step JSON (step_id + 1) with updated observation, chosen.thought/action
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

# Ensure project root is importable ('analysis' package, etc.) when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive dataset stepper -> observation + thought + chosen action")
    p.add_argument(
        "--dataset_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "WebPRMCollection_preference_pair_8k_test.json"),
        help="Path to dataset JSON array",
    )
    p.add_argument("--render", action="store_true", help="Render browser (non-headless)")
    p.add_argument("--wait_time", type=float, default=3.0, help="Seconds to wait after load")
    p.add_argument("--viewport_width", type=int, default=1280, help="Viewport width")
    p.add_argument("--viewport_height", type=int, default=2048, help="Viewport height")
    p.add_argument("--current_viewport_only", action="store_true", help="Capture current viewport only")
    # Non-interactive overrides
    p.add_argument("--task_id", type=str, default="", help="Task ID to load")
    p.add_argument("--step_id", type=str, default="", help="Step ID to load")
    p.add_argument("--chosen_action", type=str, default="", help="Chosen action to set for the next step")
    p.add_argument("--rejected_action", type=str, default="", help="Optional rejected action to record for this step")
    p.add_argument("--save_json", type=str, default="", help="Optional output path for the next-step JSON")
    return p


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_sample(samples: List[Dict[str, Any]], task_id: str, step_id: str) -> Optional[Dict[str, Any]]:
    for item in samples:
        if str(item.get("task_id")) == str(task_id) and str(item.get("step_id")) == str(step_id):
            return item
    return None


def _increment_step_id(step_id: str | int) -> str:
    try:
        return str(int(step_id) + 1)
    except Exception:
        return str(step_id) + "+1"


def main() -> None:
    # Local imports to avoid optional dependency on the removed pipeline module
    from analysis.thought_prompt import generate_thought_with_gpt4o
    from analysis.pipeline_utils import (
        extract_visible_elements,
        summarize_page_state,
        capture_observation,
        load_env_from_dotenv,
        get_element_context,
    )

    args = build_arg_parser().parse_args()

    # Load dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(2)
    samples = _load_dataset(dataset_path)

    # Step 1: Ask for task_id and step_id
    task_id = args.task_id.strip() if args.task_id else input("Enter task_id: ").strip()
    step_id = args.step_id.strip() if args.step_id else input("Enter step_id: ").strip()

    sample = _find_sample(samples, task_id, step_id)
    if sample is None:
        print(f"Sample not found for task_id={task_id} step_id={step_id}")
        sys.exit(2)

    # Step 2: Read URL and capture observation
    url = str(sample.get("current_url") or sample.get("start_url") or "").strip()
    if not url:
        print("Sample missing current_url/start_url")
        sys.exit(2)

    # Load env for API keys etc.
    load_env_from_dotenv()

    obs = capture_observation(
        url=url,
        render=args.render,
        wait_time=args.wait_time,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        current_viewport_only=args.current_viewport_only,
    )

    print("\n" + "=" * 80)
    print("ACCESSIBILITY TREE OBSERVATION")
    print("=" * 80)
    print(obs.get("text_observation", ""))
    print("=" * 80 + "\n")

    # Step 3: Ask for chosen action
    chosen_action = args.chosen_action.strip() if args.chosen_action else input("Enter chosen action (e.g., click('2253')): ").strip()
    # Optional rejected action (same interaction as chosen)
    rejected_action = args.rejected_action.strip() if args.rejected_action else input("(Optional) Enter rejected action (press Enter to skip): ").strip()

    # Step 4: Build prompt inputs
    intent = str(sample.get("intent", ""))
    thoughts = sample.get("thought_history")
    # Slice thought_history up to current step index
    if isinstance(thoughts, list):
        try:
            step_n = int(step_id)
        except Exception:
            step_n = 0
        history = " ".join(str(t).strip() for t in thoughts[: max(0, min(step_n, len(thoughts)))])
    else:
        history = ""

    visible = extract_visible_elements(obs.get("text_observation", ""))
    page_state = summarize_page_state(obs.get("text_observation", ""))

    # Step 5: Generate thought
    chosen_element_context = get_element_context(obs.get("text_observation", ""), element_id=chosen_action.split("(")[-1].split(")")[0].strip("'\"")) if chosen_action else ""
    rejected_element_context = get_element_context(obs.get("text_observation", ""), element_id=rejected_action.split("(")[-1].split(")")[0].strip("'\"")) if rejected_action else ""

    thought = generate_thought_with_gpt4o(
        goal=intent,
        history=history,
        page_state=page_state,
        visible_elements=visible,
        chosen_action=chosen_action,
        chosen_element_context=chosen_element_context,
        full_observation=obs.get("text_observation", ""),
    )

    # If rejected_action provided, generate a rejected thought as well
    rejected_thought: Optional[str] = None
    if rejected_action:
        rejected_thought = generate_thought_with_gpt4o(
            goal=intent,
            history=history,
            page_state=page_state,
            visible_elements=visible,
            chosen_action=rejected_action,
            chosen_element_context=rejected_element_context,
            full_observation=obs.get("text_observation", ""),
        )

    # Step 6: Build next-step JSON (dataset-like)
    next_record: Dict[str, Any] = dict(sample)
    next_record["step_id"] = _increment_step_id(step_id)
    next_record["current_url"] = obs.get("url", url)
    next_record["text_observation"] = obs.get("text_observation", "")

    chosen = next_record.get("chosen")
    if isinstance(chosen, dict):
        chosen["thought"] = thought
        chosen["action"] = chosen_action
    else:
        next_record["chosen"] = {"thought": thought, "action": chosen_action}

    # Update/add rejected field (single dict like dataset examples)
    if rejected_action:
        next_record["rejected"] = {
            "thought": rejected_thought or "",
            "action": rejected_action,
        }

    # Append thought to thought_history to keep the full history
    if isinstance(thoughts, list):
        next_record["thought_history"] = list(thoughts) + [thought]
    else:
        next_record["thought_history"] = [thought]

    # Print summary and optionally save
    print("THOUGHT:\n" + thought)
    print("\nNEXT STEP (JSON):")
    print(json.dumps(next_record, ensure_ascii=False, indent=2))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(next_record, f, ensure_ascii=False, indent=2)
        print(f"Saved next step to: {out_path}")


if __name__ == "__main__":
    main()


