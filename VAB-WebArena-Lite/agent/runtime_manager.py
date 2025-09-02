from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import logging

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

    def bootstrap_turn(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> None:
        """Ensure runtime has intent/URL/AXTREE for this turn.

        Always update meta first from caller, then, if AXTREE is missing and env exists,
        pull initial page state from env and set checkpoint + meta fields without a second
        update_meta call.
        """
        # First, persist caller-provided intent/urls/trajectory once
        # Ensure current_url falls back to start_url if not provided by caller
        try:
            patched_meta = dict(meta_data or {})
            if not patched_meta.get("current_url") and patched_meta.get("start_url"):
                patched_meta["current_url"] = patched_meta.get("start_url")
        except Exception:
            patched_meta = meta_data
        self.update_meta(trajectory=trajectory, intent=intent, meta_data=patched_meta)
        

        try:
            if self.get_obs_nodes_info() or not self.has_environment():
                return

            # Initialize from environment once to obtain URL + AXTREE
            start_url = meta_data.get("start_url") or meta_data.get("current_url")
            try:
                _, info = self._env.reset()  # type: ignore[misc]
            except TypeError:
                _, info = self._env.reset()  # type: ignore[misc]

            current_url = self._extract_current_url(info, start_url)
            obs_nodes = self._extract_obs_nodes_info(info)

            # If reset opened about:blank (or empty), fall back to start_url and defer AXTREE
            try:
                is_blank = (not current_url) or str(current_url).strip().lower().startswith("about:blank")
            except Exception:
                is_blank = False
            if is_blank and start_url:
                current_url = start_url
                obs_nodes = {}
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

            # Update checkpoint
            cp = CheckpointInfo(
                step=self._runtime.step,
                url=current_url or "",
                block=BlockInfo(thought=None, action=None),
                objective=intent or "",
                observation=observation_text,
            )
            self.set_checkpoint(cp)

            m: Meta = self._runtime.meta
            m.start_url = start_url or current_url
            m.current_url = current_url
            m.obs_nodes_info = obs_nodes
            
        except Exception:
            pass

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
            block=BlockInfo(thought=thought, action=action_str),
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

        # Informational log after action execution
        try:
            logging.getLogger("reward_guided_logger").info("action performed")
        except Exception:
            pass

    # Checkpoint controls (environment-driven)
    def set_checkpoint(self, checkpoint: CheckpointInfo) -> None:
        self._runtime.checkpoint = checkpoint

    def get_checkpoint(self) -> Optional[CheckpointInfo]:
        return self._runtime.checkpoint


