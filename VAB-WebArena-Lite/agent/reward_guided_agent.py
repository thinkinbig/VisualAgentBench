import logging
import os
import json
import re
import random
from typing import Any, Optional, List, Dict, Tuple

from beartype import beartype

from .agent import Agent
from browser_env.trajectory import Trajectory
from browser_env.actions import (
    Action,
    create_none_action,
)
from llms import lm_config
from llms.json_validator import JSONResponseValidator
from .types import (
    Meta,
    AgentRuntime,
    PolicyRequest,
    PolicyResponse,
    RewardRequest,
    RewardResponse,
    PairwiseDecision,
    CheckpointInfo,
    AggregateInfo,
)
from .prompts.p import (
    role as pairwise_role,
    evaluation_summarized_v3 as pairwise_eval_criteria,
    context_rft_v3 as pairwise_user_template,
)


def _hr(title: str) -> str:
    return f"\n===== {title} =====\n"


class RewardGuidedAgent(Agent):
    """Reward-guided agent (staged skeleton) with human-readable logging.

    Stage 1 (Policy): Build PolicyRequest and log the policy prompt.
    Stage 2 (Reward): Run pairwise knockout. For each match, build RewardRequest
                      and log the reward prompt and decision.

    Note: This skeleton does not call LLMs; prompts are printed for inspection.
    """

    def __init__(
        self,
        action_set_tag: str,
        policy_lm_config: Optional[lm_config.LMConfig] = None,
        reward_lm_config: Optional[lm_config.LMConfig] = None,
        captioning_fn: Optional[Any] = None,
        num_samples: int = 8,
        best_of: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger("reward_guided_logger")
        self.action_set_tag = action_set_tag
        self.policy_lm_config = policy_lm_config
        self.reward_lm_config = reward_lm_config
        self.captioning_fn = captioning_fn
        self.num_samples = max(1, int(num_samples))
        self.best_of = max(1, int(best_of))
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self.validator = JSONResponseValidator()
        self.runtime = AgentRuntime()
        self._policy_prompt: Optional[Dict[str, Any]] = None
        self._load_policy_prompt()

    # ---------------------------- Utilities ----------------------------
    def _update_meta(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> None:
        m: Meta = self.runtime.meta
        m.intent = intent
        m.trajectory = trajectory or []
        # Accept environment-fed context when available
        m.start_url = meta_data.get("start_url", m.start_url)
        m.current_url = meta_data.get("current_url", m.current_url)
        # Try to capture structured node metadata from browser_env
        m.obs_nodes_info = None
        try:
            if isinstance(meta_data.get("obs_nodes_info"), dict):
                m.obs_nodes_info = meta_data.get("obs_nodes_info")
            elif isinstance(meta_data.get("text"), dict) and isinstance(meta_data["text"].get("obs_nodes_info"), dict):
                m.obs_nodes_info = meta_data["text"].get("obs_nodes_info")
            elif isinstance(meta_data.get("image"), dict) and isinstance(meta_data["image"].get("obs_nodes_info"), dict):
                m.obs_nodes_info = meta_data["image"].get("obs_nodes_info")
            elif isinstance(meta_data.get("observation_metadata"), dict):
                md = meta_data["observation_metadata"]
                if isinstance(md.get("text"), dict) and isinstance(md["text"].get("obs_nodes_info"), dict):
                    m.obs_nodes_info = md["text"].get("obs_nodes_info")
        except Exception:
            m.obs_nodes_info = None

    # ---------------------------- Stage 1: Policy ----------------------------
    def _build_policy_request(self) -> PolicyRequest:
        m = self.runtime.meta
        previous_action_text = "None"
        if self.runtime.previous_action is not None and hasattr(self.runtime.previous_action, "action_type"):
            # Minimal printable previous action
            at = self.runtime.previous_action.action_type.value
            eid = self.runtime.previous_action.element_id or ""
            previous_action_text = f"{at}{f' [{eid}]' if eid else ''}"

        trajectory_str = "\n".join(
            [f"{{THOUGHT: {tap.thought}, ACTION: {tap.action}}}" for tap in m.trajectory]
        )

        req = PolicyRequest(
            intent=m.intent or "",
            observation=m.observation or "",
            trajectory=trajectory_str,
            current_url=m.current_url or "",
            previous_action=previous_action_text,
            start_url=m.start_url,
            discovery_context=None,
        )
        self.runtime.policy_request = req
        return req

    def _format_policy_prompt(self, req: PolicyRequest) -> Tuple[str, str]:
        # If enhanced prompt JSON is available, use it to format system/user
        if self._policy_prompt:
            intro = self._policy_prompt.get("intro", "")
            examples = self._policy_prompt.get("examples", [])
            template = self._policy_prompt.get("template", "{objective}\n{observation}\n{url}")
            guidelines = self._policy_prompt.get("output_guidelines", "")

            # Build examples block
            examples_block: List[str] = []
            for i, pair in enumerate(examples):
                if isinstance(pair, list) and len(pair) == 2:
                    input_ex, output_ex = pair
                    examples_block.append(f"Example {i+1}:\n{input_ex}\n\n{output_ex}")
            system_text = intro
            if examples_block:
                system_text += "\n\## Examples:\n" + "\n\n".join(examples_block)
            if guidelines:
                system_text += "\n\n" + guidelines

            # Fill user template
            last_cp = self.runtime.last_checkpoint.dict(by_alias=True) if self.runtime.last_checkpoint else {}
            last_ag = self.runtime.last_aggregate.dict(by_alias=True) if self.runtime.last_aggregate else {}
            open_tabs = "[]"  # placeholder if not tracked
            # Build observation text from structured nodes if available
            observation_text = req.observation
            nodes = self.runtime.meta.obs_nodes_info
            if not observation_text and isinstance(nodes, dict):
                try:
                    lines: List[str] = []
                    # Keep stable order
                    for node_id, node in nodes.items():
                        t = str(node.get("text", ""))
                        if t:
                            lines.append(t)
                    observation_text = "\n".join(lines)
                except Exception:
                    observation_text = ""

            user_text = template.format(
                objective=req.intent,
                observation=observation_text or "",
                url=req.current_url,
                open_tabs=open_tabs,
                previous_action=req.previous_action,
                last_checkpoint=json.dumps(last_cp, ensure_ascii=False),
                last_aggregate=json.dumps(last_ag, ensure_ascii=False),
            )
            return system_text, user_text

    def _log_policy_prompt(self, system_text: str, user_text: str) -> None:
        self.logger.info(_hr("====POLICY SYSTEM PROMPT====\n") + system_text)
        self.logger.info(_hr("====POLICY USER PROMPT====\n") + user_text)

    def _parse_ax_observation(self, observation: str) -> List[Dict[str, str]]:
        elems: List[Dict[str, str]] = []
        if not observation:
            return elems
        for line in observation.splitlines():
            m = re.match(r"^\s*\[(?P<id>[-A-Za-z0-9_]+)\]\s+(?P<role>[A-Za-z]+)(?:\s+'(?P<name>[^']*)')?", line)
            if m:
                elems.append({
                    "id": m.group("id"),
                    "role": m.group("role").lower(),
                    "name": (m.group("name") or "").strip(),
                })
        return elems

    def _build_action_pool_from_obs(self, req: PolicyRequest) -> List[str]:
        # Prefer structured node info from browser_env if available
        elems: List[Dict[str, str]] = []
        nodes = self.runtime.meta.obs_nodes_info
        if isinstance(nodes, dict) and nodes:
            for node_id, node in nodes.items():
                # node.get('text') like "[id] role 'name' ..." from processors
                text = str(node.get("text", ""))
                m = re.match(r"^\s*\[(?P<id>[-A-Za-z0-9_]+)\]\s+(?P<role>[A-Za-z]+)(?:\s+'(?P<name>[^']*)')?", text)
                if m:
                    elems.append({
                        "id": m.group("id"),
                        "role": m.group("role").lower(),
                        "name": (m.group("name") or "").strip(),
                    })
        else:
            elems = self._parse_ax_observation(req.observation)
        pool: List[str] = []
        clickable_roles = {"link", "button", "image", "img", "checkbox", "radio"}
        type_roles = {"input", "textbox", "searchbox", "textarea", "combobox"}
        # Element-derived actions
        for e in elems:
            eid = e["id"]
            role = e["role"]
            if role in clickable_roles:
                pool.append(f"```click [{eid}]```")
            if role in type_roles:
                # Minimal placeholder content; press_enter_after defaults to 1
                pool.append(f"```type [{eid}] [test] [1]```")
            if role in {"link", "image", "img"}:
                pool.append(f"```hover [{eid}]```")
        # Generic actions
        pool.extend(["```scroll [down]```", "```scroll [up]```"])
        # Deduplicate while preserving order
        seen = set()
        unique_pool: List[str] = []
        for a in pool:
            if a not in seen:
                seen.add(a)
                unique_pool.append(a)
        return unique_pool

    def _rank_action(self, action_str: str) -> int:
        s = action_str.lower()
        order = [
            "click", "type", "hover", "scroll",
        ]
        for idx, verb in enumerate(order):
            if s.startswith(f"```{verb}") or s.startswith(verb):
                return idx
        return len(order)

    def _sample_policy_candidates(self, req: PolicyRequest) -> List[PolicyResponse]:
        # Best-of sampling from current observation and URL-derived candidates
        action_pool = self._build_action_pool_from_obs(req)
        if not action_pool:
            self.logger.info("No element-derived actions from observation; falling back to scroll only.")
            action_pool = ["```scroll [down]```", "```scroll [up]```"]

        chosen_actions: List[str] = []
        for slot in range(self.num_samples):
            k = min(len(action_pool), max(1, self.best_of))
            sampled = random.sample(action_pool, k=k)
            sampled_sorted = sorted(sampled, key=self._rank_action)
            best = sampled_sorted[0]
            try:
                self.logger.info(
                    _hr(f"POLICY BEST-OF (slot {slot+1}/{self.num_samples}, best_of={self.best_of})") +
                    "\n".join([f"- {s}" for s in sampled_sorted]) +
                    f"\nChosen: {best}"
                )
            except Exception:
                pass
            chosen_actions.append(best)

        url = self.runtime.meta.current_url or req.current_url or ""
        cp_list: List[PolicyResponse] = []
        for idx, action_str in enumerate(chosen_actions):
            checkpoint = CheckpointInfo(
                step=max(1, self.runtime.step),
                url=url,
                tab={"id": 1, "stack": []},
                objective=self.runtime.meta.intent or req.intent,
                env_flags={},
                state_hash=f"cp_{self.runtime.step}_{idx}"
            )
            aggregate = AggregateInfo(
                facts=[], entities=[], evidence=[], plan_next_1to3=[action_str], risks=[], stop_condition=""
            )
            # Build minimal BLOCK-like structure via pydantic alias
            pr = PolicyResponse(CHECKPOINT=checkpoint, AGGREGATE=aggregate, BLOCK={
                "thought": "Sampled next step from observation.",
                "action": action_str,
            })
            cp_list.append(pr)

        try:
            self.logger.info(_hr("POLICY CANDIDATES") + "\n".join([f"#{i}: {c.block.action}" for i, c in enumerate(cp_list)]))
        except Exception:
            pass
        return cp_list

    # ---------------------------- Stage 2: Reward ----------------------------
    def _build_reward_request(self, base: PolicyRequest, a: PolicyResponse, b: PolicyResponse) -> RewardRequest:
        thought1 = a.block.thought if a and a.block else ""
        action1 = a.block.action if a and a.block else ""
        thought2 = b.block.thought if b and b.block else ""
        action2 = b.block.action if b and b.block else ""
        return RewardRequest(
            intent=base.intent,
            observation=base.observation,
            trajectory=base.trajectory,
            start_url=base.start_url or "",
            current_url=base.current_url,
            thought1=thought1,
            action1=action1,
            thought2=thought2,
            action2=action2,
        )

    def _format_reward_prompt(self, rr: RewardRequest) -> Tuple[str, str]:
        system_text = f"{pairwise_role}\n{pairwise_eval_criteria}"
        user_text = pairwise_user_template.format(
            intent=rr.intent,
            observation=rr.observation,
            trajectory=rr.trajectory or "(empty)",
            start_url=rr.start_url,
            current_url=rr.current_url,
            thought1=rr.thought1,
            action1=rr.action1,
            thought2=rr.thought2,
            action2=rr.action2,
        )
        return system_text, user_text

    # ---------------------------- Init helpers ----------------------------
    def _load_policy_prompt(self) -> None:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "prompts", "jsons", "enhanced_actree.json")
            with open(json_path, "r", encoding="utf-8") as f:
                self._policy_prompt = json.load(f)
            self.logger.info("Loaded policy prompt JSON: %s", json_path)
        except Exception as e:
            self._policy_prompt = None
            self.logger.warning("Failed to load policy prompt JSON: %s", e)

    def _log_reward_prompt(self, system_text: str, user_text: str, round_idx: int, pair_idx: int) -> None:
        self.logger.info(_hr(f"====REWARD SYSTEM PROMPT (Round {round_idx}, Pair {pair_idx})====\n") + system_text)
        self.logger.info(_hr(f"====REWARD USER PROMPT (Round {round_idx}, Pair {pair_idx})====\n") + user_text)

    def _score_pair(self, rr: RewardRequest, round_idx: int, pair_idx: int) -> RewardResponse:
        system_text, user_text = self._format_reward_prompt(rr)
        self._log_reward_prompt(system_text, user_text, round_idx, pair_idx)
        # Placeholder decision (undecided). Real impl would call reward LLM and parse.
        return RewardResponse(
            raw_response="",
            decision=PairwiseDecision.UNDECIDED,
            winner=None,
            think=None,
            criteria=None,
            analysis=None,
            is_valid=False,
            parse_errors=[],
        )

    def _knockout(self, base_req: PolicyRequest, candidates: List[PolicyResponse]) -> Optional[PolicyResponse]:
        if not candidates:
            self.logger.info("No policy candidates; skipping knockout.")
            return None
        if len(candidates) == 1:
            self.logger.info("Single candidate; knockout skipped. Winner = #0")
            return candidates[0]

        indices = list(range(len(candidates)))
        round_idx = 0
        while len(indices) > 1:
            self.logger.info(_hr(f"====KNOCKOUT ROUND {round_idx + 1}===="))
            next_round: List[int] = []
            i = 0
            pair_idx = 0
            while i < len(indices):
                if i == len(indices) - 1:
                    self.logger.info(f"Odd participant gets bye: #{indices[i]}")
                    next_round.append(indices[i])
                    break
                ia = indices[i]
                ib = indices[i + 1]
                ra = candidates[ia]
                rb = candidates[ib]
                rr = self._build_reward_request(base_req, ra, rb)
                self.logger.info(f"Pair {pair_idx}: #{ia} vs #{ib}")
                score = self._score_pair(rr, round_idx + 1, pair_idx)
                if score.winner == 2 or score.decision == PairwiseDecision.RESPONSE_2:
                    self.logger.info(f"Winner: #{ib}")
                    next_round.append(ib)
                else:
                    self.logger.info(f"Winner: #{ia} (tie -> left wins by default)")
                    next_round.append(ia)
                pair_idx += 1
                i += 2
            indices = next_round
            round_idx += 1

        winner_idx = indices[0]
        self.logger.info(_hr("KNOCKOUT RESULT") + f"Winner index: #{winner_idx}")
        return candidates[winner_idx]

    # ---------------------------- Public API ----------------------------
    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
        images: Optional[List[Any]] = None,
        output_response: bool = False,
    ) -> Action:
        # Step 0: Update runtime/meta state
        self.runtime.step += 1
        self._update_meta(trajectory, intent, meta_data)

        # Stage 1: Build policy request and log prompt
        policy_req = self._build_policy_request()
        sys_txt, usr_txt = self._format_policy_prompt(policy_req)
        self._log_policy_prompt(sys_txt, usr_txt)

        # Stage 1 sampling (placeholder)
        candidates = self._sample_policy_candidates(policy_req)
        self.runtime.policy_candidates = candidates

        # Stage 2: Knockout (placeholder scoring)
        winner = self._knockout(policy_req, candidates)
        self.runtime.selected_policy = winner

        # Attach last checkpoint/aggregate if available
        if winner and winner.checkpoint:
            self.runtime.last_checkpoint = winner.checkpoint
        if winner and winner.aggregate:
            self.runtime.last_aggregate = winner.aggregate

        # Return a no-op action for now; include a readable summary
        action = create_none_action()
        summary = "no-candidates" if not candidates else (
            (winner.block.action if winner and winner.block else "winner-without-block")
        )
        action["raw_prediction"] = f"reward-guided-skeleton step={self.runtime.step} summary={summary}"
        return action


