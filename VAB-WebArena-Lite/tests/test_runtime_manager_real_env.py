import os
import sys
from pathlib import Path
import unittest

# Ensure project root on sys.path and DATASET is set before importing modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DATASET", "webarena")
os.environ.setdefault("OPENAI_API_KEY", "fake_key")



try:
    from browser_env import ScriptBrowserEnv
    from browser_env.actions import create_scroll_action
    ENV_AVAILABLE = ScriptBrowserEnv is not None
except Exception:
    ScriptBrowserEnv = None  # type: ignore[assignment]
    create_scroll_action = None  # type: ignore[assignment]
    ENV_AVAILABLE = False

from agent.runtime_manager import RuntimeManager
from agent.types import AgentRuntime


@unittest.skipUnless(ENV_AVAILABLE, "ScriptBrowserEnv is not available in this environment")
class TestRuntimeManagerRealEnv(unittest.TestCase):
    def setUp(self) -> None:
        assert ScriptBrowserEnv is not None
        # Use accessibility_tree to populate obs_nodes_info
        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=0,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 800, "height": 600},
            save_trace_enabled=False,
            sleep_after_execution=0.0,
            captioning_fn=None,
        )
        # Minimal reset without start_url; opens about:blank-like page but still yields a tree
        self.env.reset()

    def tearDown(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def test_execute_scroll_updates_runtime(self) -> None:
        rt = RuntimeManager(runtime=AgentRuntime(), env=self.env)  # type: ignore[arg-type]
        # Seed meta
        rt.update_meta(trajectory=[], intent="Scroll Test", meta_data={
            "start_url": "",
            "current_url": "",
            "obs_nodes_info": {},
        })
        # Execute a simple scroll action
        assert create_scroll_action is not None
        action_dict = create_scroll_action("down")
        thought = "Scroll down to reveal more content"
        action_str = "scroll [down]"

        rt.execute_action(action_dict=action_dict, thought=thought, action_str=action_str)

        # Step incremented
        self.assertEqual(rt.runtime.step, 1)
        # Checkpoint populated
        cp = rt.get_checkpoint()
        self.assertIsNotNone(cp)
        assert cp is not None
        self.assertIn("scroll", cp.action)
        # Meta updated
        self.assertIsInstance(rt.get_obs_nodes_info(), dict)
        # Trajectory appended
        self.assertEqual(len(rt.get_trajectory()), 1)


if __name__ == "__main__":
    unittest.main()


