import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Minimal env variables to avoid provider imports
os.environ.setdefault("DATASET", "webarena")
os.environ.setdefault("OPENAI_API_KEY", "fake_key")

from agent.runtime_manager import RuntimeManager
from agent.types import AgentRuntime

from browser_env import ScriptBrowserEnv, create_goto_url_action


def main() -> None:
    start_url = "http://www.google.com"
    dest_url = "https://www.google.com/search?q=github+copilot"

    env = ScriptBrowserEnv(
        headless=True,
        slow_mo=0,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 800, "height": 600},
        save_trace_enabled=False,
        sleep_after_execution=0.0,
        captioning_fn=None,
    )

    rt = RuntimeManager(runtime=AgentRuntime(), env=env)  # type: ignore[arg-type]
    # Initialize initial checkpoint by resetting the environment with a start_url config
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"start_url": start_url}, f)
        config_path = f.name
    try:
        rt.initialize_from_environment(
            intent="Demo: real env initial checkpoint then goto",
            start_url=start_url,
            reset_options={"config_file": config_path},
        )
    finally:
        try:
            os.unlink(config_path)
        except Exception:
            pass

    print("Initial checkpoint (no action):")
    cp0 = rt.get_checkpoint()
    if cp0 is None:
        print("  None")
    else:
        print(f"  step={cp0.step}")
        print(f"  url={cp0.url}")
        print(f"  action={cp0.action}")
        print("  observation=")
        print(cp0.observation)

    # Perform a real goto action via browser_env factory
    thought = "Open GitHub Copilot"
    action_str = f"goto [{dest_url}]"
    action_dict = create_goto_url_action(dest_url)

    rt.execute_action(action_dict=action_dict, thought=thought, action_str=action_str)

    cp = rt.get_checkpoint()
    print("\nAfter goto checkpoint:")
    if cp is None:
        print("  None")
    else:
        print(f"  step={cp.step}")
        print(f"  url={cp.url}")
        print(f"  action={cp.action}")
        print("  observation=")
        print(cp.observation)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()


