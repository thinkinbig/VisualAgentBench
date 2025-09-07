from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import logging
import uuid
import os
import json
import re

from llms.types import ThoughtActionPair

# Import runtime dependencies that are actually used
try:
    from browser_env.trajectory import Trajectory
    from browser_env import (
        Action,
        ScriptBrowserEnv,
        AsyncScriptBrowserEnv,
        extract_current_url,
        extract_obs_nodes_info,
    )
    from .trajectory_tree import TrajectoryTree
except ImportError:
    # Fallback if browser_env is not available
    Trajectory = list  # type: ignore[assignment]
    Action = Dict[str, Any]  # type: ignore[assignment]
    ScriptBrowserEnv = Any  # type: ignore[assignment]
    AsyncScriptBrowserEnv = Any  # type: ignore[assignment]
    TrajectoryTree = Any  # type: ignore[assignment]
    
    def extract_current_url(info: Dict[str, Any], fallback: Optional[str] = None) -> str:
        return fallback or ""
    
    def extract_obs_nodes_info(info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract obs_nodes_info from environment info."""
        try:
            observation_metadata = info.get("observation_metadata", {})
            text_metadata = observation_metadata.get("text", {})
            return text_metadata.get("obs_nodes_info", {})
        except Exception:
            return {}
from .types import (
    AgentRuntime,
    Meta,
    BlockInfo,
    PairwiseMatch,
    RewardRequest,
    RewardResponse,
    CheckpointInfo,
)


class RuntimeManager:
    """Encapsulates all AgentRuntime mutations and environment-bridging state updates."""

    def __init__(self, runtime: Optional[AgentRuntime] = None, env: Optional[Union["ScriptBrowserEnv", "AsyncScriptBrowserEnv"]] = None, max_steps: int = 30) -> None:
        self._runtime: AgentRuntime = runtime or AgentRuntime()
        self._env: Optional[Union[ScriptBrowserEnv, AsyncScriptBrowserEnv]] = env
        self._max_steps: int = max_steps
        self._task_ended: bool = False
        self._trajectory_tree: Optional[TrajectoryTree] = None

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

        # Initialize trajectory tree if not already done
        if self._trajectory_tree is None:
            self.initialize_trajectory_tree(intent)

        

        try:
            if self.get_obs_nodes_info() or not self.has_environment():
                return

            # Initialize from environment once to obtain URL + AXTREE
            start_url = meta_data.get("start_url") or meta_data.get("current_url")
            try:
                _, info = self._env.reset()  # type: ignore[misc]
            except TypeError:
                _, info = self._env.reset()  # type: ignore[misc]

            current_url = extract_current_url(info, start_url)
            obs_nodes = extract_obs_nodes_info(info)

            # If reset opened about:blank (or empty), fall back to start_url and defer AXTREE
            try:
                is_blank = (not current_url) or str(current_url).strip().lower().startswith("about:blank")
            except Exception:
                is_blank = False
            if is_blank and start_url:
                current_url = start_url
                obs_nodes = {}
            observation_text = self.compose_observation_from_nodes(obs_nodes)

            # Update checkpoint
            screenshot_path = self._get_current_screenshot_path()
            cp = CheckpointInfo(
                step=self._runtime.step,
                url=current_url or "",
                block=BlockInfo(thought="", action=""),
                objective=intent or "",
                observation={
                    "text": observation_text,
                    "nodes_info": obs_nodes or {},
                    "url": current_url or "",
                },
                screenshot_path=screenshot_path,
            )
            self.set_checkpoint(cp)

            m: Meta = self._runtime.meta
            m.start_url = start_url or current_url
            m.current_url = current_url
            m.obs_nodes_info = obs_nodes
            
        except Exception:
            pass

    # State setters/getters

    def set_selected_block(self, block: Optional[BlockInfo]) -> None:
        self._runtime.selected_block = block

    def append_tournament_match(self, match: PairwiseMatch) -> None:
        self._runtime.tournament_history.append(match)

    def record_pair(self, round_idx: int, idx_a: int, idx_b: int, rr: RewardRequest, resp: RewardResponse) -> None:
        """Append one PairwiseMatch into runtime for later visualization."""
        try:
            pm = PairwiseMatch(
                round_index=round_idx,
                index_a=idx_a,
                index_b=idx_b,
                reward_request=rr,
                reward_response=resp,
            )
            self.append_tournament_match(pm)
        except Exception:
            # best-effort; never break the policy loop
            pass


    def compose_trajectory_from_meta(self) -> str:
        """Compose trajectory text from trajectory tree (preferred) or runtime metadata."""
        # First try to get trajectory from trajectory tree
        if self._trajectory_tree is not None:
            try:
                lines: List[str] = []
                # Get all state nodes in order
                state_nodes = self._trajectory_tree.get_state_nodes()
                state_nodes.sort(key=lambda n: n.step)
                
                for state_node in state_nodes:
                    if state_node.step == 0:
                        continue  # Skip root node
                    
                    # Find the selected candidate for this state
                    selected_candidate = self._trajectory_tree._get_selected_candidate_for_state(state_node)
                    if selected_candidate:
                        thought = selected_candidate.thought or ""
                        action = selected_candidate.action or ""
                        if thought and action:
                            lines.append(f"{{THOUGHT: {thought}, ACTION: {action}}}")
                
                if lines:
                    return "\n".join(lines)
            except Exception:
                pass
        
        # Fallback to runtime metadata
        m = self.get_meta()
        if not isinstance(m.trajectory, list) or not m.trajectory:
            return ""
        lines: List[str] = []
        for item in m.trajectory:
            try:
                thought = getattr(item, "thought", None)
                action = getattr(item, "action", None)
                if isinstance(thought, str) and isinstance(action, str):
                    lines.append(f"{{THOUGHT: {thought}, ACTION: {action}}}")
            except Exception:
                continue
        return "\n".join(lines)


    def append_trajectory(self, thought: str, action: str) -> None:
        try:
            self._runtime.meta.trajectory.append(ThoughtActionPair(thought=thought, action=action))
        except Exception:
            pass

    def increment_step(self) -> None:
        self._runtime.step += 1

    # Sampling memory management
    def clear_current_round_samples(self) -> None:
        """Clear current round samples at the start of each turn."""
        self._runtime.current_round_samples.clear()

    def add_current_round_sample(self, action: str) -> None:
        """Add an action to current round samples."""
        if action and action not in self._runtime.current_round_samples:
            self._runtime.current_round_samples.append(action)

    def get_current_round_samples(self) -> List[str]:
        """Get current round samples."""
        return self._runtime.current_round_samples.copy()

    # Environment binding
    def set_environment(self, env: Union["ScriptBrowserEnv", "AsyncScriptBrowserEnv"]) -> None:
        self._env = env

    def has_environment(self) -> bool:
        return self._env is not None

    def compose_observation_from_nodes(self, obs_nodes: Optional[Dict[str, Any]]) -> str:
        """Compose observation text from nodes info."""
        if not obs_nodes:
            return ""
        
        # Simple composition - can be enhanced based on actual needs
        lines = []
        for node_id, node_info in obs_nodes.items():
            if isinstance(node_info, dict):
                text = node_info.get('text', '')
                if text:
                    lines.append(f"{text}")
        
        return "\n".join(lines)

    def _get_current_screenshot_path(self) -> Optional[str]:
        """Get the current screenshot path if available."""
        try:
            if hasattr(self._env, 'page') and self._env.page:
                # Create screenshots directory
                screenshots_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)
                
                # Generate screenshot filename
                run_id = "run"
                screenshot_filename = f"screenshot_{run_id}_step_{self.get_step()}.png"
                return os.path.join(screenshots_dir, screenshot_filename)
        except Exception:
            pass
        return None
    

    def _describe_action(self, action_str: str) -> str:
        """Human-readable action meaning by looking up AXTREE id label from current observation."""
        try:
            s = (action_str or "").strip()
            # Extract element id for click/type/hover
            m = re.search(r"^(click|type|hover)\s*\[([^\]]+)\]", s, re.IGNORECASE)
            if not m:
                # For goto, include URL
                g = re.search(r"^goto\s*\[([^\]]+)\]", s, re.IGNORECASE)
                if g:
                    return f"goto → {g.group(1)}"
                return s
            elem_id = str(m.group(2)).strip()
            verb = m.group(1).lower()
            
            # Get nodes info from current checkpoint observation or runtime meta
            nodes = None
            checkpoint = self.get_checkpoint()
            if (
                checkpoint
                and checkpoint.observation
                and isinstance(checkpoint.observation, dict)
                and checkpoint.observation.get("nodes_info")
            ):
                nodes = checkpoint.observation.get("nodes_info")
            else:
                # Fallback to runtime meta obs_nodes_info
                nodes = self.get_obs_nodes_info()
            
            # Special handling for type actions - extract the content being typed
            if verb == "type":
                # Extract content from type [id] content format
                type_match = re.match(r"^type\s*\[([^\]]+)\]\s+(.+)$", s, re.IGNORECASE)
                if type_match:
                    content = type_match.group(2).strip()
                    if isinstance(nodes, dict) and elem_id in nodes:
                        node = nodes.get(elem_id, {})
                        node_text = str(node.get("text", "")).strip()
                        # Try to extract role and name from node_text
                        mm = re.search(r"^\s*\[[^\]]+\]\s*(?P<role>[A-Za-z]+)(?:\s+'(?P<name>[^']*)')?", node_text)
                        if mm:
                            role = mm.group("role").lower()
                            name = (mm.group("name") or "").strip()
                            if name:
                                return f"type {role} '{name}' with '{content}' (#{elem_id})"
                            return f"type {role} with '{content}' (#{elem_id})"
                        return f"type #{elem_id} '{node_text}' with '{content}'"
                    return f"type #{elem_id} with '{content}'"
            
            if isinstance(nodes, dict) and elem_id in nodes:
                node = nodes.get(elem_id, {})
                node_text = str(node.get("text", "")).strip()
                # Try to extract role and name from node_text like: "[430] menuitem 'Beauty & Personal Care' ..."
                mm = re.search(r"^\s*\[[^\]]+\]\s*(?P<role>[A-Za-z]+)(?:\s+'(?P<name>[^']*)')?", node_text)
                if mm:
                    role = mm.group("role").lower()
                    name = (mm.group("name") or "").strip()
                    if name:
                        return f"{verb} {role} '{name}' (#{elem_id})"
                    return f"{verb} {role} (#{elem_id})"
                return f"{verb} #{elem_id} '{node_text}'"
            # Fallback: try to find any matching id key as string
            if isinstance(nodes, dict):
                node = nodes.get(str(elem_id)) or nodes.get(int(elem_id)) if str(elem_id).isdigit() else None  # type: ignore[index]
                if isinstance(node, dict):
                    node_text = str(node.get("text", "")).strip()
                    mm = re.search(r"^\s*\[[^\]]+\]\s*(?P<role>[A-Za-z]+)(?:\s+'(?P<name>[^']*)')?", node_text)
                    if mm:
                        role = mm.group("role").lower()
                        name = (mm.group("name") or "").strip()
                        if name:
                            return f"{verb} {role} '{name}' (#{elem_id})"
                        return f"{verb} {role} (#{elem_id})"
                    return f"{verb} #{elem_id} '{node_text}'"
            return s
        except Exception:
            return action_str

    # Execute action in environment and update runtime (checkpoint/meta/trajectory)
    def execute_action(self, action_dict: Action, thought: str, action_str: str) -> None:
        if self._env is None:
            raise RuntimeError("Environment is not set in RuntimeManager")
        
        # Check if task has already ended
        if self._task_ended:
            try:
                logging.getLogger("reward_guided_logger").warning("[EXECUTE] Task has already ended, skipping action execution")
            except Exception:
                pass
            return
        
        # Increment step at the moment of executing an action
        self.increment_step()
        try:
            obs, _, terminated, _, info = self._env.step(action_dict)
        except Exception as e:
            raise RuntimeError(f"Environment execution failed: {e}")

        # Check if task should end after this action
        if self._check_task_completion(terminated, info):
            self._task_ended = True

        current_url = extract_current_url(info, self.get_current_url())
        obs_nodes = extract_obs_nodes_info(info)
        observation_text = self.compose_observation_from_nodes(obs_nodes)

        # Save screenshot if available
        screenshot_path = None
        try:
            if hasattr(self._env, 'page') and self._env.page:
                # Create screenshots directory
                screenshots_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)
                
                # Generate screenshot filename
                run_id = "run"
                screenshot_filename = f"screenshot_{run_id}_step_{self.get_step()}.png"
                screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
                
                # Take screenshot
                screenshot_bytes = self._env.page.screenshot()
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_bytes)
                
                try:
                    logging.getLogger("reward_guided_logger").info(f"[SCREENSHOT_SAVED] {screenshot_path}")
                except Exception:
                    pass
        except Exception as e:
            try:
                logging.getLogger("reward_guided_logger").warning(f"[SCREENSHOT_SAVE_FAILED] {e}")
            except Exception:
                pass

        cp = CheckpointInfo(
            step=self._runtime.step,
            url=current_url,
            block=BlockInfo(thought=thought, action=action_str),
            objective=self.get_intent() or "",
            observation={
                "text": observation_text,
                "nodes_info": obs_nodes or {},
                "url": current_url or "",
            },
            screenshot_path=screenshot_path,
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

        # Update trajectory tree
        self.update_trajectory_tree_after_action(thought, action_str, current_url)

        # Informational log after action execution
        try:
            if self._task_ended:
                logging.getLogger("reward_guided_logger").info("action performed - TASK ENDED")
            else:
                logging.getLogger("reward_guided_logger").info("action performed")
        except Exception:
            pass


    # Checkpoint controls (environment-driven)
    def set_checkpoint(self, checkpoint: CheckpointInfo) -> None:
        self._runtime.checkpoint = checkpoint

    def get_checkpoint(self) -> Optional[CheckpointInfo]:
        return self._runtime.checkpoint


    # --------- Tournament / Candidates getters & setters (for viz) ---------
    def get_selected_block(self) -> Optional[BlockInfo]:
        return self._runtime.selected_block

    def get_tournament_history(self) -> List[PairwiseMatch]:
        return list(self._runtime.tournament_history or [])

    def set_tournament_history(self, matches: List[PairwiseMatch]) -> None:
        self._runtime.tournament_history = matches

    def clear_tournament_history(self) -> None:
        self._runtime.tournament_history.clear()

    def get_step(self) -> int:
        return int(self._runtime.step or 0)

    def is_task_ended(self) -> bool:
        return self._task_ended

    def set_max_steps(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def _check_task_completion(self, terminated: bool, info: Dict[str, Any]) -> bool:
        """check task completion (reached max steps or environment terminated)"""
        # check reached max steps
        if self.get_step() >= self._max_steps:
            try:
                logging.getLogger("reward_guided_logger").info(f"[TASK_END] Reached max steps: {self.get_step()}/{self._max_steps}")
            except Exception:
                pass
            return True
        
        # check environment terminated
        if terminated:
            try:
                logging.getLogger("reward_guided_logger").info(f"[TASK_END] Environment terminated at step {self.get_step()}")
            except Exception:
                pass
            return True
        
        if isinstance(info, dict) and info.get("fail_error"):
            try:
                logging.getLogger("reward_guided_logger").info(f"[TASK_END] Task failed due to error: {info.get('fail_error')}")
            except Exception:
                pass
            return True
        
        return False


    # ---- Serialization helpers ----
    def _model_to_dict(self, model_obj: Any) -> Dict[str, Any]:
        try:
            return model_obj.model_dump()  # type: ignore[attr-defined]
        except Exception:
            try:
                return model_obj.dict()  # type: ignore[attr-defined]
            except Exception:
                try:
                    if hasattr(model_obj, "model_dump_json"):
                        return json.loads(model_obj.model_dump_json())  # type: ignore[attr-defined]
                except Exception:
                    pass
        return {}

    # ---- Trajectory Tree Management ----
    
    def initialize_trajectory_tree(self, intent: str, run_id: Optional[str] = None) -> None:
        """Initialize the trajectory tree with root node."""
        from .trajectory_tree import TrajectoryTree, TrajRoot
        
        # Create root node with screenshot if available
        root = TrajRoot(
            node_id="root",
            parent_id=None,
            step=0,
            intent=intent,
            run_id=run_id or "default_run",
            screenshot_path=self._get_current_screenshot_path()
        )
        
        # Initialize trajectory tree
        self._trajectory_tree = TrajectoryTree(root)

    def get_trajectory_tree(self) -> Optional[Any]:
        """Get the current trajectory tree."""
        return self._trajectory_tree




    def get_current_node_id(self) -> Optional[str]:
        """Get the current active node ID (the most recently added state node)."""
        if self._trajectory_tree is None:
            return None
        
        # Find the most recent state node by step
        current_step = self.get_step()
        state_nodes = self._trajectory_tree.get_state_nodes()
        for node in state_nodes:
            if node.step == current_step:
                return node.node_id
        return None

    def get_parent_node_id(self) -> Optional[str]:
        """Get the parent node ID for the current step."""
        if self._trajectory_tree is None:
            return "root"
        
        # Find the most recent state node (parent for new state)
        state_nodes = self._trajectory_tree.get_state_nodes()
        if not state_nodes:
            return "root"  # Default to root if no state nodes
        
        # Return the most recent state node
        return max(state_nodes, key=lambda n: n.step).node_id

    def update_trajectory_tree_after_action(self, thought: str, action: str, url: Optional[str] = None) -> None:
        """Update trajectory tree after executing an action."""
        if self._trajectory_tree is None:
            return
        
        import uuid
        
        # Get current step and parent
        current_step = self.get_step()
        parent_id = self.get_parent_node_id()
        
        # Get current observation info
        checkpoint = self.get_checkpoint()
        observation_hash = None
        obs_nodes_info = None
        if checkpoint and checkpoint.observation and isinstance(checkpoint.observation, dict):
            obs_nodes_info = checkpoint.observation.get("nodes_info")
            # Generate simple hash from observation text
            obs_text = checkpoint.observation.get("text", "")
            if obs_text:
                import hashlib
                observation_hash = hashlib.md5(obs_text.encode()).hexdigest()[:16]
        
        # Create a new state node representing the result of this action
        state_node_id = f"state_{current_step}_{uuid.uuid4().hex[:8]}"
        self._trajectory_tree.add_state_node(
            node_id=state_node_id,
            parent_id=parent_id,
            step=current_step,
            url=url,
            observation_hash=observation_hash,
            obs_nodes_info=obs_nodes_info,
            screenshot_path=self._get_current_screenshot_path()
        )
        
        # Add candidate nodes for this state (if we have pending candidates)
        selected_candidate_id = None
        if hasattr(self, '_pending_candidates') and self._pending_candidates:
            for i, candidate in enumerate(self._pending_candidates):
                # Use simple candidate naming: candidate_1, candidate_2, etc.
                candidate_node_id = f"candidate_{i + 1}"
                self._trajectory_tree.add_candidate_node(
                    node_id=candidate_node_id,
                    parent_id=state_node_id,
                    thought=candidate.thought or "",
                    action=candidate.action or "",
                    meaning=self._describe_action(candidate.action or "")
                )
                
                # Add to state's candidates list
                self._trajectory_tree.add_candidate_to_state(state_node_id, candidate_node_id)
                
                # Check if this candidate matches the executed action
                if candidate.action == action:
                    selected_candidate_id = candidate_node_id
            
            # Mark the selected candidate as selected
            if selected_candidate_id:
                self._trajectory_tree.mark_candidate_as_selected(selected_candidate_id)
        
        # Clear pending candidates
        if hasattr(self, '_pending_candidates'):
            self._pending_candidates = []

    def export_trajectory_tree_graphviz(self) -> str:
        """Export trajectory tree as Graphviz DOT format."""
        if self._trajectory_tree is None:
            return ""
        return self._trajectory_tree.to_graphviz()

    def export_trajectory_tree_json(self) -> str:
        """Export trajectory tree as JSON."""
        if self._trajectory_tree is None:
            return "{}"
        return self._trajectory_tree.to_json()

    def record_candidates(self, candidates: List[BlockInfo]) -> None:
        """Record candidate actions for the current step."""
        # Store candidates in runtime for later use
        self._runtime.current_round_samples = [c.action for c in candidates]
        # Store the full candidate objects in a temporary location
        if not hasattr(self, '_pending_candidates'):
            self._pending_candidates = []
        self._pending_candidates = candidates

    def get_current_node_candidates(self) -> List[BlockInfo]:
        """Get candidate actions for the current step."""
        # Return pending candidates if available
        if hasattr(self, '_pending_candidates'):
            return self._pending_candidates
        return []

    def get_trajectory_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the trajectory tree."""
        if self._trajectory_tree is None:
            return {
                "total_nodes": 0,
                "state_nodes": 0,
                "candidate_nodes": 0,
                "selected_nodes": 0,
                "total_candidates": 0
            }
        
        # Use the new methods from TrajectoryTree
        state_nodes = self._trajectory_tree.get_state_nodes()
        candidate_nodes = self._trajectory_tree.get_candidate_nodes()
        selected_nodes = self._trajectory_tree.get_selected_nodes()
        
        return {
            "total_nodes": len(self._trajectory_tree.nodes),
            "state_nodes": len(state_nodes),
            "candidate_nodes": len(candidate_nodes),
            "selected_nodes": len(selected_nodes),
            "total_candidates": sum(len(n.candidates) for n in state_nodes)
        }





