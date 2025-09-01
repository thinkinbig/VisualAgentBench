from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING

from llms.types import ThoughtActionPair

# Avoid importing heavy browser_env modules at import time to prevent optional
# dependency errors when only types are needed.
if TYPE_CHECKING:
    from browser_env.trajectory import Trajectory
    from browser_env import (
        Action,
        ScriptBrowserEnv,
        AsyncScriptBrowserEnv,
    )
else:
    Trajectory = list  # type: ignore[assignment]
    Action = Dict[str, Any]  # type: ignore[assignment]
    ScriptBrowserEnv = Any  # type: ignore[assignment]
    AsyncScriptBrowserEnv = Any  # type: ignore[assignment]
from .types import (
    AgentRuntime,
    Meta,
    PolicyRequest,
    BlockInfo,
    PairwiseMatch,
    AggregateInfo,
    CheckpointInfo,
)


class RuntimeManager:
    """Encapsulates all AgentRuntime mutations and environment-bridging state updates."""

    def __init__(self, runtime: Optional[AgentRuntime] = None, env: Optional[Union["ScriptBrowserEnv", "AsyncScriptBrowserEnv"]] = None) -> None:
        self._runtime: AgentRuntime = runtime or AgentRuntime()
        self._env: Optional[Union[ScriptBrowserEnv, AsyncScriptBrowserEnv]] = env

    @property
    def runtime(self) -> AgentRuntime:
        return self._runtime

    # Lightweight getters for meta fields
    def get_meta(self) -> Meta:
        return self._runtime.meta

    def get_intent(self) -> Optional[str]:
        return self._runtime.meta.intent

    def get_current_url(self) -> Optional[str]:
        return self._runtime.meta.current_url

    def get_start_url(self) -> Optional[str]:
        return self._runtime.meta.start_url

    def get_obs_nodes_info(self) -> Optional[Dict[str, Any]]:
        return self._runtime.meta.obs_nodes_info

    def get_trajectory(self) -> Trajectory:
        return self._runtime.meta.trajectory

    def update_meta(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> None:
        m: Meta = self._runtime.meta
        m.intent = intent
        m.trajectory = trajectory or []
        m.start_url = meta_data.get("start_url", m.start_url)
        m.current_url = meta_data.get("current_url", m.current_url)
        m.obs_nodes_info = None
        try:
            if isinstance(meta_data.get("obs_nodes_info"), dict):
                m.obs_nodes_info = meta_data.get("obs_nodes_info")
        except Exception:
            m.obs_nodes_info = None

    # State setters/getters

    def set_block_candidates(self, candidates: List[BlockInfo]) -> None:
        self._runtime.block_candidates = candidates

    def set_selected_block(self, block: Optional[BlockInfo]) -> None:
        self._runtime.selected_block = block

    def append_tournament_match(self, match: PairwiseMatch) -> None:
        self._runtime.tournament_history.append(match)

    def set_aggregate(self, aggregate: AggregateInfo) -> None:
        self._runtime.aggregate = aggregate

    def get_aggregate(self) -> Optional[AggregateInfo]:
        return self._runtime.aggregate

    def append_trajectory(self, thought: str, action: str) -> None:
        try:
            self._runtime.meta.trajectory.append(ThoughtActionPair(thought=thought, action=action))
        except Exception:
            pass

    def increment_step(self) -> None:
        self._runtime.step += 1

    # Environment binding
    def set_environment(self, env: Union["ScriptBrowserEnv", "AsyncScriptBrowserEnv"]) -> None:
        self._env = env

    def has_environment(self) -> bool:
        return self._env is not None

    # Initialize checkpoint and meta from environment reset info (no action)
    def initialize_from_reset_info(self, info: Dict[str, Any], intent: str, start_url: Optional[str] = None) -> None:
        current_url = self._extract_current_url(info, start_url)
        obs_nodes = self._extract_obs_nodes_info(info)
        observation_text = ""
        try:
            if isinstance(obs_nodes, dict) and obs_nodes:
                lines: List[str] = []
                for _, node in obs_nodes.items():
                    t = str(node.get("text", ""))
                    if t:
                        lines.append(t)
                observation_text = "\n".join(lines)
        except Exception:
            observation_text = ""

        cp = CheckpointInfo(
            step=self._runtime.step,
            url=current_url or "",
            action=None,
            objective=intent or "",
            observation=observation_text,
        )
        self.set_checkpoint(cp)

        self.update_meta(
            trajectory=self.get_trajectory(),
            intent=intent,
            meta_data={
                "start_url": start_url or current_url,
                "current_url": current_url,
                "obs_nodes_info": obs_nodes,
            },
        )

    # Convenience: call env.reset() and initialize checkpoint/meta from its info
    def initialize_from_environment(self, intent: str, start_url: Optional[str] = None, reset_options: Optional[Dict[str, Any]] = None) -> None:
        if self._env is None or not hasattr(self._env, "reset"):
            raise RuntimeError("Environment with a reset() method is required to initialize from environment.")
        try:
            _, info = self._env.reset(**({"options": reset_options} if reset_options is not None else {}))  # type: ignore[misc]
        except TypeError:
            # Some envs accept no kwargs for reset
            _, info = self._env.reset()  # type: ignore[misc]
        self.initialize_from_reset_info(info=info, intent=intent, start_url=start_url)

    # Environment helpers
    def _extract_obs_nodes_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            om = info.get("observation_metadata", {})
            if isinstance(om.get("obs_nodes_info"), dict):
                return om.get("obs_nodes_info")
            if isinstance(om.get("text", {}), dict) and isinstance(om.get("text", {}).get("obs_nodes_info"), dict):
                return om.get("text", {}).get("obs_nodes_info")
            if isinstance(om.get("image", {}), dict) and isinstance(om.get("image", {}).get("obs_nodes_info"), dict):
                return om.get("image", {}).get("obs_nodes_info")
        except Exception:
            pass
        return {}

    def _extract_current_url(self, info: Dict[str, Any], fallback: Optional[str]) -> str:
        try:
            page = info.get("page")
            if hasattr(page, "url"):
                return page.url
        except Exception:
            pass
        return fallback or ""

    # Execute action in environment and update runtime (checkpoint/meta/trajectory)
    def execute_action(self, action_dict: Action, thought: str, action_str: str) -> None:
        if self._env is None:
            raise RuntimeError("Environment is not set in RuntimeManager")
        # Increment step at the moment of executing an action
        self.increment_step()
        try:
            obs, _, terminated, _, info = self._env.step(action_dict)
        except Exception as e:
            raise RuntimeError(f"Environment execution failed: {e}")

        current_url = self._extract_current_url(info, self.get_current_url())
        obs_nodes = self._extract_obs_nodes_info(info)
        observation_text = ""
        try:
            # Build observation text from nodes (mirrors agent helper semantics)
            if isinstance(obs_nodes, dict) and obs_nodes:
                lines: List[str] = []
                for _, node in obs_nodes.items():
                    t = str(node.get("text", ""))
                    if t:
                        lines.append(t)
                observation_text = "\n".join(lines)
        except Exception:
            observation_text = ""

        cp = CheckpointInfo(
            step=self._runtime.step,
            url=current_url,
            action=action_str,
            objective=self.get_intent() or "",
            observation=observation_text,
        )
        self.set_checkpoint(cp)

        # Update meta with fresh page state (keep trajectory)
        self.update_meta(
            trajectory=self.get_trajectory(),
            intent=self.get_intent() or "",
            meta_data={
                "start_url": self.get_start_url(),
                "current_url": current_url,
                "obs_nodes_info": obs_nodes,
            },
        )
        # Append trajectory item
        self.append_trajectory(thought=thought, action=action_str)

    # Checkpoint controls (environment-driven)
    def set_checkpoint(self, checkpoint: CheckpointInfo) -> None:
        self._runtime.checkpoint = checkpoint

    def get_checkpoint(self) -> Optional[CheckpointInfo]:
        return self._runtime.checkpoint


