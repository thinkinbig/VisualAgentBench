from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import logging
import uuid
import os
import json
import re

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
    RewardRequest,
    RewardResponse,
    AggregateInfo,
    CheckpointInfo,
    TrajRoot,
    TrajectoryTree,
    TrajNode,
    TrajEdge,
    NodeStatus,
    SnapShot,
    SnapshotMeta,
    SnapshotCandidate,
    SnapshotRequest,
    SnapshotResponse,
    SnapshotMatch,
    SnapshotRound,
    SnapshotWinner,
)


class RuntimeManager:
    """Encapsulates all AgentRuntime mutations and environment-bridging state updates."""

    def __init__(self, runtime: Optional[AgentRuntime] = None, env: Optional[Union["ScriptBrowserEnv", "AsyncScriptBrowserEnv"]] = None, max_steps: int = 30) -> None:
        self._runtime: AgentRuntime = runtime or AgentRuntime()
        self._env: Optional[Union[ScriptBrowserEnv, AsyncScriptBrowserEnv]] = env
        self._max_steps: int = max_steps
        self._task_ended: bool = False

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

        # Initialize trajectory tree root once per run (meta can be left empty)
        try:
            if getattr(self._runtime, "trajectory_tree", None) is None:
                run_id = uuid.uuid4().hex
                root = TrajRoot(run_id=run_id, intent=intent or "", meta={})
                self._runtime.trajectory_tree = TrajectoryTree(root=root)
                # Ensure a root node (step=0) exists for the main path
                try:
                    root_node = TrajNode(
                        node_id="root",
                        parent_id=None,
                        step=0,
                        url=patched_meta.get("start_url"),
                        observation_hash=None,
                        checkpoint=None,
                        labels={"type": "root"},
                        candidates=[],
                    )
                    self._runtime.trajectory_tree.add_node(root_node)
                except Exception:
                    pass
        except Exception:
            pass
        

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

    def compose_observation_from_nodes(self, nodes: Optional[Dict[str, Any]]) -> str:
        """Compose observation text from accessibility tree nodes."""
        if not isinstance(nodes, dict) or not nodes:
            return ""
        lines: List[str] = []
        try:
            for _, node in nodes.items():
                t = str(node.get("text", ""))
                if t:
                    lines.append(t)
        except Exception:
            pass
        return "\n".join(lines)

    def compose_trajectory_from_meta(self) -> str:
        """Compose trajectory text from runtime metadata."""
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

    def _describe_action(self, action_str: str) -> str:
        """Human-readable action meaning by looking up AXTREE id label from obs_nodes_info."""
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
            
            # Special handling for type actions - extract the content being typed
            if verb == "type":
                # Extract content from type [id] content format
                type_match = re.match(r"^type\s*\[([^\]]+)\]\s+(.+)$", s, re.IGNORECASE)
                if type_match:
                    content = type_match.group(2).strip()
                    nodes = self.get_obs_nodes_info()
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
            
            nodes = self.get_obs_nodes_info()
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

        # Save screenshot if available
        screenshot_path = None
        try:
            if hasattr(self._env, 'page') and self._env.page:
                # Create screenshots directory
                screenshots_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)
                
                # Generate screenshot filename
                run_id = getattr(self._runtime.trajectory_tree.root, "run_id", "run") if self._runtime.trajectory_tree else "run"
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
            observation=observation_text,
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

        # Update trajectory graph: add EXECUTED edge and new node on the main path
        try:
            self._record_executed_transition(thought=thought, action=action_str)
            # Mark the executed action's PLANNED edge as EXECUTED and mark others as ABORTED
            self._update_node_status(action_str)
        except Exception:
            pass

        # Informational log after action execution
        try:
            if self._task_ended:
                logging.getLogger("reward_guided_logger").info("action performed - TASK ENDED")
            else:
                logging.getLogger("reward_guided_logger").info("action performed")
        except Exception:
            pass

        # Auto-save trajectory if task has ended
        self._auto_save_trajectory_if_ended()

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
        """检查任务是否已结束"""
        return self._task_ended

    def set_max_steps(self, max_steps: int) -> None:
        """设置最大步数限制"""
        self._max_steps = max_steps

    def _check_task_completion(self, terminated: bool, info: Dict[str, Any]) -> bool:
        """检查任务是否完成（达到最大步数或环境终止）"""
        # 检查是否达到最大步数
        if self.get_step() >= self._max_steps:
            try:
                logging.getLogger("reward_guided_logger").info(f"[TASK_END] Reached max steps: {self.get_step()}/{self._max_steps}")
            except Exception:
                pass
            return True
        
        # 检查环境是否终止
        if terminated:
            try:
                logging.getLogger("reward_guided_logger").info(f"[TASK_END] Environment terminated at step {self.get_step()}")
            except Exception:
                pass
            return True
        
        # 检查是否有其他终止信号
        if isinstance(info, dict):
            # 检查是否有成功/失败信号
            if info.get("success", False) or info.get("terminated", False):
                try:
                    logging.getLogger("reward_guided_logger").info(f"[TASK_END] Task completed successfully at step {self.get_step()}")
                except Exception:
                    pass
                return True
        
        return False

    def _auto_save_trajectory_if_ended(self) -> None:
        """如果任务已结束，自动保存轨迹"""
        if self._task_ended:
            try:
                saved_path = self.save_final_trajectory()
                if saved_path:
                    logging.getLogger("reward_guided_logger").info(f"[AUTO_SAVE] Trajectory automatically saved: {saved_path}")
                else:
                    logging.getLogger("reward_guided_logger").warning("[AUTO_SAVE] Failed to save trajectory")
            except Exception as e:
                try:
                    logging.getLogger("reward_guided_logger").error(f"[AUTO_SAVE] Error saving trajectory: {e}")
                except Exception:
                    pass

    # --------- Trajectory graph helpers ---------

    def _ensure_tree(self) -> TrajectoryTree:
        if getattr(self._runtime, "trajectory_tree", None) is None:
            run_id = uuid.uuid4().hex
            self._runtime.trajectory_tree = TrajectoryTree(root=TrajRoot(run_id=run_id, intent=self.get_intent() or "", meta={}))
            try:
                root_node = TrajNode(
                    node_id="root",
                    parent_id=None,
                    step=0,
                    url=self.get_start_url(),
                    observation_hash=None,
                    checkpoint=None,
                    labels={"type": "root"},
                    candidates=[],
                )
                self._runtime.trajectory_tree.add_node(root_node)
            except Exception:
                pass
        return self._runtime.trajectory_tree  # type: ignore[return-value]

    def _latest_main_node_id(self) -> str:
        tree = self._ensure_tree()
        if not tree.nodes:
            return "root"
        latest = max(tree.nodes, key=lambda n: n.step)
        return latest.node_id

    def _record_executed_transition(self, thought: str, action: str, meaning: Optional[str] = None) -> None:
        tree = self._ensure_tree()
        parent_id = self._latest_main_node_id()
        new_node_id = uuid.uuid4().hex
        
        # Get screenshot path from the latest checkpoint
        screenshot_path = None
        try:
            checkpoint = self.get_checkpoint()
            if checkpoint and hasattr(checkpoint, 'screenshot_path'):
                screenshot_path = checkpoint.screenshot_path
        except Exception:
            pass
        
        try:
            node = TrajNode(
                node_id=new_node_id,
                parent_id=parent_id if parent_id != "root" else "root",
                step=self.get_step(),
                url=self.get_current_url(),
                observation_hash=None,
                checkpoint=self.get_checkpoint(),
                screenshot_path=screenshot_path,
                obs_nodes_info=self.get_obs_nodes_info(),
                labels={},
                candidates=[],
            )
            tree.add_node(node)
        except Exception:
            return
        try:
            # 如果没有提供meaning，则计算一个
            if meaning is None:
                meaning = self._describe_action(action)
            
            edge = TrajEdge(
                edge_id=uuid.uuid4().hex,
                parent_id=parent_id,
                child_id=new_node_id,
                thought=thought,
                action=action,
                meaning=meaning,
                reward=None,
                notes={"step": self.get_step()},
            )
            tree.add_edge(edge)
        except Exception:
            pass

    def record_candidates(self, candidates: List[BlockInfo]) -> None:
        """记录当前步骤的候选动作到轨迹树中，为每个candidate创建对应的节点。"""
        tree = self._ensure_tree()
        current_node_id = self._latest_main_node_id()
        
        # 更新当前节点的候选动作
        try:
            tree.set_candidates_at_node(current_node_id, candidates)
            
            # 为每个candidate创建对应的节点
            for i, candidate in enumerate(candidates):
                candidate_node_id = f"candidate_{current_node_id}_{i}_{uuid.uuid4().hex[:8]}"
                
                # 创建candidate节点
                candidate_node = TrajNode(
                    node_id=candidate_node_id,
                    parent_id=current_node_id,
                    step=self.get_step(),
                    url=self.get_current_url(),
                    observation_hash=None,
                    checkpoint=None,
                    screenshot_path=None,
                    obs_nodes_info=None,
                    labels={"type": "candidate"},
                    candidates=[],
                )
                tree.add_node(candidate_node)
                
                # 创建从当前节点到candidate节点的edge
                meaning = self._describe_action(candidate.action)
                candidate_edge = TrajEdge(
                    edge_id=f"candidate_{current_node_id}_{i}_{uuid.uuid4().hex[:8]}",
                    parent_id=current_node_id,
                    child_id=candidate_node_id,
                    thought=candidate.thought,
                    action=candidate.action,
                    meaning=meaning,
                    reward=None,
                    notes={
                        "candidate_index": i,
                        "step": self.get_step(),
                        "is_candidate": True
                    }
                )
                tree.add_edge(candidate_edge)
                
        except Exception:
            pass

    def _update_node_status(self, executed_action: str) -> None:
        """更新节点状态：将执行的动作对应的节点标记为SELECTED。"""
        tree = self._ensure_tree()
        
        try:
            if len(tree.nodes) >= 2:
                # 找到root节点（执行动作的节点）
                root_node = tree.get_node("root")
                if root_node:
                    root_node.status = NodeStatus.SELECTED
                    
                    # 新创建的节点保持candidate状态，直到它执行动作
                    current_node = tree.nodes[-1]
                    current_node.status = NodeStatus.CANDIDATE
                    
                    # 根据实际执行的动作找到对应的candidate节点
                    for edge in tree.edges:
                        if (edge.parent_id == root_node.node_id and 
                            edge.notes.get("is_candidate", False) and
                            edge.action == executed_action):
                            # 找到对应的candidate节点
                            candidate_node = tree.get_node(edge.child_id)
                            if candidate_node:
                                candidate_node.status = NodeStatus.SELECTED
                                # 标记这个edge为已执行
                                edge.notes["is_executed"] = True
                            break
        except Exception:
            pass

    def save_final_trajectory(self, out_dir: Optional[str] = None) -> Optional[str]:
        """保存最终的轨迹树（JSON + Graphviz + Interactive HTML）。"""
        tree = getattr(self._runtime, "trajectory_tree", None)
        if tree is None:
            return None
        try:
            if out_dir is None:
                # default to VAB-WebArena-Lite/outputs/trajectory relative to this file
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "trajectory"))
            else:
                base_dir = out_dir
            os.makedirs(base_dir, exist_ok=True)
            run_id = getattr(tree.root, "run_id", "run")
            
            # Save JSON
            json_path = os.path.join(base_dir, f"trajectory_{run_id}_final.json")
            try:
                # Pydantic v2
                payload = tree.model_dump_json(indent=2)  # type: ignore[attr-defined]
            except Exception:
                try:
                    # Pydantic v1
                    payload = tree.json(indent=2, ensure_ascii=False)  # type: ignore[attr-defined]
                except Exception:
                    # Last resort
                    payload = json.dumps(getattr(tree, "dict", lambda: {})(), indent=2, ensure_ascii=False)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(payload)
            
            # Save Graphviz DOT
            dot_path = os.path.join(base_dir, f"trajectory_{run_id}_final.dot")
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write(tree.to_graphviz())
            
            # Save Interactive HTML
            html_path = os.path.join(base_dir, f"trajectory_{run_id}_interactive.html")
            tree.to_interactive_html(html_path)
            
            try:
                logging.getLogger("reward_guided_logger").info(f"[TRAJ_FINAL_SAVED] JSON: {json_path}, DOT: {dot_path}, HTML: {html_path}")
            except Exception:
                pass
            return json_path
        except Exception:
            return None


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

    # ---------------------------- Snapshot Management ----------------------------
    def _describe_action(self, action_str: str) -> str:
        """Human-readable action meaning by looking up AXTREE id label from obs_nodes_info."""
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
            
            # Special handling for type actions - extract the content being typed
            if verb == "type":
                # Extract content from type [id] content format
                type_match = re.match(r"^type\s*\[([^\]]+)\]\s+(.+)$", s, re.IGNORECASE)
                if type_match:
                    content = type_match.group(2).strip()
                    nodes = self.get_obs_nodes_info()
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
            
            nodes = self.get_obs_nodes_info()
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

    def _build_snapshot(self) -> SnapShot:
        """Assemble a SnapShot object from current runtime state."""
        meta = self.get_meta()
        checkpoint = self.get_checkpoint()
        # 从当前节点获取candidates
        tree = self._runtime.trajectory_tree
        current_node_id = self._latest_main_node_id()
        cands = tree.get_candidates_at_node(current_node_id) if tree else []
        matches = self.get_tournament_history()
        winner = self.get_selected_block()

        # 预构建 candidates 映射（index -> SnapshotCandidate）
        snap_candidates: List[SnapshotCandidate] = []
        for i, b in enumerate(cands):
            snap_candidates.append(SnapshotCandidate(
                index=i,
                thought=b.thought,
                action=b.action,
                meaning=self._describe_action(b.action),
            ))

        # 便捷函数：安全取候选条目（按 index）
        def _cand_entry(i: int) -> SnapshotCandidate:
            if 0 <= i < len(snap_candidates):
                return snap_candidates[i]
        
            return SnapshotCandidate(index=i, thought=None, action=None, meaning=None)

        # round → pairs 聚合
        from collections import defaultdict
        rounds_map: Dict[int, List[SnapshotMatch]] = defaultdict(list)
        for m in matches:
            # 胜者解析（兼容 undecided）
            w: Optional[int] = None
            try:
                if m.reward_response.winner in (1, 2):
                    w = m.reward_response.winner
                else:
                    dec = getattr(m.reward_response.decision, "value", str(m.reward_response.decision))
                    if isinstance(dec, str) and "response_1" in dec:
                        w = 1
                    elif isinstance(dec, str) and "response_2" in dec:
                        w = 2
            except Exception:
                w = None

            snap_match = SnapshotMatch(
                a=_cand_entry(m.index_a),
                b=_cand_entry(m.index_b),
                request=SnapshotRequest(
                    intent=m.reward_request.intent,
                    observation=m.reward_request.observation,
                    trajectory=m.reward_request.trajectory,
                    start_url=m.reward_request.start_url,
                    current_url=m.reward_request.current_url,
                ),
                response=SnapshotResponse(
                    raw=m.reward_response.raw_response,
                    decision=getattr(m.reward_response.decision, "value", str(m.reward_response.decision)),
                    winner=w,
                    is_valid=bool(m.reward_response.is_valid),
                    parse_errors=list(m.reward_response.parse_errors or []),
                    criteria=m.reward_response.criteria,
                    analysis=m.reward_response.analysis,
                    think=m.reward_response.think,
                ),
            )
            rounds_map[m.round_index].append(snap_match)

        snap_rounds: List[SnapshotRound] = [
            SnapshotRound(round_index=ri, pairs=pairs)
            for ri, pairs in sorted(rounds_map.items(), key=lambda x: x[0])
        ]

        # winner 汇总（可解析到 index）
        winner_index = None
        if winner is not None:
            for i, b in enumerate(cands):
                if b.action == winner.action and b.thought == winner.thought:
                    winner_index = i
                    break

        snapshot = SnapShot(
            meta=SnapshotMeta(
                intent=meta.intent,
                start_url=meta.start_url,
                current_url=meta.current_url,
                step=self.get_step(),
            ),
            checkpoint=checkpoint,   # 直接强类型注入 CheckpointInfo
            candidates=snap_candidates,
            rounds=snap_rounds,
            winner={
                "index": winner_index,
                "thought": getattr(winner, "thought", None),
                "action": getattr(winner, "action", None),
                "meaning": (self._describe_action(winner.action) if winner else None),
            } if winner is not None else None,
        )
        return snapshot

    def save_snapshot(self, out_path: str) -> None:
        """Serialize the current tournament state to a SnapShot JSON file."""
        try:
            snap = self._build_snapshot()
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            # pydantic v2：model_dump_json；v1：json()
            try:
                payload = snap.model_dump_json(indent=2, ensure_ascii=False)  # type: ignore[attr-defined]
            except Exception:
                payload = snap.json(indent=2, ensure_ascii=False)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(payload)
            try:
                logging.getLogger("reward_guided_logger").info(f"[SNAPSHOT_SAVED] {out_path}")
            except Exception:
                pass
        except Exception as e:
            try:
                logging.getLogger("reward_guided_logger").error(f"[SNAPSHOT_SAVE_FAILED] {e}")
            except Exception:
                pass




