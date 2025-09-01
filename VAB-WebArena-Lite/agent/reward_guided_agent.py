import logging
import json
import re
from typing import Optional, List, Dict, Any

from beartype import beartype

from .agent import Agent
from browser_env.trajectory import Trajectory
from browser_env.actions import (
    Action,
    create_id_based_action,
)
from llms import lm_config
from llms.utils import call_llm, build_api_input_for_text
from .types import (
    PlanRequest,
    BlockInfo,
    RewardRequest,
    RewardResponse,
    PairwiseDecision,
    AggregateInfo,
)
from .runtime_manager import RuntimeManager
 
from .prompts.sample_p import (
    intro as sample_intro,
    output_guidelines as sample_guidelines,
    render_prompt as render_sample_prompt,
)
from .prompts.note_p import (
    role_note_taker_agg,
    note_taker_rules_v1,
    context_note_taker_v1,
)


class RewardGuidedAgent(Agent):
    """Reward-guided agent with BLOCK-only policy and post-reward AGGREGATE update.

    Flow per turn:
    1) Build PolicyRequest from runtime meta.
    2) Sample BLOCK candidates (thought + bracket action string).
    3) Run pairwise knockout using reward prompt (stubbed scorer by default).
    4) After a winner is chosen, update runtime.aggregate.plan_next to winner.action.
    5) Return parsed Action created from winner.action.
    """

    def __init__(
        self,
        action_set_tag: str,
        policy_lm_config: Optional[lm_config.LMConfig] = None,
        reward_lm_config: Optional[lm_config.LMConfig] = None,
        note_lm_config: Optional[lm_config.LMConfig] = None,
        num_samples: int = 16,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger("reward_guided_logger")
        self.action_set_tag = action_set_tag
        self.policy_lm_config = policy_lm_config
        self.reward_lm_config = reward_lm_config
        self.note_lm_config = note_lm_config
        self.num_samples = max(1, int(num_samples))

        self.rt = RuntimeManager()

    # ---------------------------- Utilities ----------------------------
    def _compose_observation_from_nodes(self, nodes: Optional[Dict[str, Any]]) -> str:
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

    def _compose_trajectory_from_meta(self) -> str:
        m = self.rt.get_meta()
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


    # ---------------------------- Stage 1: Policy (BLOCK only) ----------------------------
    def _build_sample_request(self):
        # Build a lightweight container with required attributes for prompts
        from types import SimpleNamespace
        req = SimpleNamespace(
            intent=self.rt.get_intent(),
            observation=self._compose_observation_from_nodes(self.rt.get_obs_nodes_info()),
            current_url=self.rt.get_current_url(),
            action=self.rt.get_checkpoint().action if self.rt.get_checkpoint() else None,
            start_url=self.rt.get_start_url(),
        )
        return req

    def _format_policy_prompt(self, req: PlanRequest) -> tuple[str, str]:
        prev_action = "None"
        try:
            cp = self.rt.get_checkpoint()
            if cp and isinstance(cp.action, str) and cp.action.strip():
                prev_action = cp.action.strip()
        except Exception:
            pass
        aggregate_json = "{}"
        try:
            ag = self.rt.get_aggregate()
            if ag is not None:
                aggregate_json = json.dumps(ag.dict(), ensure_ascii=False)
        except Exception:
            pass
        system_text = f"{sample_intro}\n{sample_guidelines}"
        user_text = render_sample_prompt(
            objective=req.intent or "",
            observation=req.observation or "",
            url=req.current_url or "",
            previous_action=prev_action,
            aggregate=aggregate_json,
        )
        return system_text, user_text

    def _parse_block_from_response(self, raw: str) -> Optional[BlockInfo]:
        if not isinstance(raw, str) or not raw.strip():
            return None
        # Try JSON first
        try:
            import re as _re
            m = _re.search(r"\{[\s\S]*\}", raw)
            if m:
                data = json.loads(m.group(0))
                blk = data.get("BLOCK") if isinstance(data, dict) else None
                if isinstance(blk, dict):
                    thought = blk.get("thought") or ""
                    action = blk.get("action") or ""
                    # Strip backticks if present
                    if isinstance(action, str):
                        s = action.strip()
                        if s.startswith("```") and s.endswith("```"):
                            s = s[3:-3].strip()
                        action = s
                    if isinstance(thought, str) and isinstance(action, str) and action:
                        return BlockInfo(thought=thought, action=action)
        except Exception:
            pass
        # Fallback: extract action from code fence, thought from a simple key
        try:
            import re as _re
            act_m = _re.search(r"```([\s\S]*?)```", raw)
            thought_m = _re.search(r"\"thought\"\s*:\s*\"([^\"]*)\"", raw)
            action = act_m.group(1).strip() if act_m else ""
            thought = thought_m.group(1).strip() if thought_m else ""
            if action:
                return BlockInfo(thought=thought, action=action)
        except Exception:
            pass
        return None

    def _parse_ax_observation(self, observation: Optional[str]) -> List[Dict[str, str]]:
        elems: List[Dict[str, str]] = []
        if not isinstance(observation, str) or not observation.strip():
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

    def _format_reward_prompt(self, rr: RewardRequest) -> tuple[str, str]:
        """Build system/user texts for reward LLM."""
        system_text = (
            "You are an expert judge for web navigation actions.\n"
            "Choose which candidate action better progresses the objective given the current page.\n"
            "Output strictly JSON: {\"decision\": \"response_1|response_2|undecided\"}."
        )

        user_parts = [
            f"## OBJECTIVE\n{rr.intent}",
            f"## URL\n{rr.current_url}",
            f"## START_URL\n{rr.start_url}",
            f"## AXTREE\n{rr.observation}",
            f"## TRAJECTORY\n{rr.trajectory}",
            "## CANDIDATES",
            f"### RESPONSE_1\nTHOUGHT: {rr.thought1}\nACTION: {rr.action1}",
            f"### RESPONSE_2\nTHOUGHT: {rr.thought2}\nACTION: {rr.action2}",
            "## INSTRUCTIONS\nReturn strictly one JSON object: {\"decision\": \"response_1|response_2|undecided\"}"
        ]
        user_text = "\n\n".join(user_parts)
        return system_text, user_text

    def _parse_reward_decision(self, raw: str) -> PairwiseDecision:
        try:
            if not isinstance(raw, str) or not raw.strip():
                return PairwiseDecision.UNDECIDED
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                data = json.loads(m.group(0))
                dec = (data.get("decision") or "").strip().lower()
                if dec in (PairwiseDecision.RESPONSE_1.value, PairwiseDecision.RESPONSE_2.value, PairwiseDecision.UNDECIDED.value):
                    return PairwiseDecision(dec)
        except Exception:
            pass
        return PairwiseDecision.UNDECIDED

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
            nodes = self.rt.get_obs_nodes_info()
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

    def _sample_block_candidates(self, req: PlanRequest) -> List[BlockInfo]:
        if self.policy_lm_config is None:
            raise ValueError("Policy LLM config is required for LLM-based sampling")
        sys_txt, usr_txt = self._format_policy_prompt(req)
        # Build provider-correct API input (list of messages for chat models)
        prompt = build_api_input_for_text(self.policy_lm_config, sys_txt, usr_txt)
        # Log policy prompt (system and user) once per turn
        try:
            self.logger.info(f"[PROMPT_POLICY_SYSTEM]\n{sys_txt}")
            self.logger.info(f"[PROMPT_POLICY_USER]\n{usr_txt}")
        except Exception:
            pass
        seen_actions = set()
        results: List[BlockInfo] = []
        # Bounded sampling: cap attempts to avoid slow infinite rolling
        target_samples = max(1, int(getattr(self, "num_samples", 16)))
        max_attempts = max(10, target_samples * 3)
        attempts = 0
        # Keep sampling until target unique actions or attempt cap
        while len(results) < target_samples and attempts < max_attempts:
            try:
                raw = call_llm(self.policy_lm_config, prompt)
            except Exception:
                raw = ""
            # Log raw policy LLM response
            try:
                self.logger.info(f"[RAW_POLICY] {raw}")
            except Exception:
                pass
            blk = self._parse_block_from_response(raw)
            if blk is None or not isinstance(blk.action, str) or not blk.action.strip():
                attempts += 1
                continue
            action_str = blk.action.strip().strip("`")
            if action_str in seen_actions:
                attempts += 1
                continue
            seen_actions.add(action_str)
            # Normalize action field without backticks
            blk = BlockInfo(thought=blk.thought or "", action=action_str)
            results.append(blk)
            attempts += 1
        # Log thought with action (and brief meaning)
        try:
            for i, b in enumerate(results):
                try:
                    meaning = self._describe_action(b.action)
                except Exception:
                    meaning = b.action
                self.logger.info(f"[THOUGHT] #{i}: {b.thought} | [ACTION] {b.action} | [MEANING] {meaning}")
        except Exception:
            pass
        return results

    # ---------------------------- Stage 2: Reward (pairwise knockout) ----------------------------
    def _build_reward_request(self, base: PlanRequest, a: BlockInfo, b: BlockInfo) -> RewardRequest:
        return RewardRequest(
            intent=base.intent,
            observation=base.observation,
            trajectory=self._compose_trajectory_from_meta(),
            start_url=base.start_url,
            current_url=base.current_url,
            thought1=a.thought,
            action1=a.action,
            thought2=b.thought,
            action2=b.action,
        )

    def _score_pair(self, rr: RewardRequest, round_idx: int, pair_idx: int) -> RewardResponse:
        # If reward LLM is configured, use it; otherwise heuristic
        if self.reward_lm_config is not None:
            sys_txt, usr_txt = self._format_reward_prompt(rr)
            # Log reward prompt
            try:
                self.logger.info(f"[PROMPT_REWARD_SYSTEM]\n{sys_txt}")
                self.logger.info(f"[PROMPT_REWARD_USER]\n{usr_txt}")
            except Exception:
                pass
            raw = ""
            try:
                api_input = build_api_input_for_text(self.reward_lm_config, sys_txt, usr_txt)
                raw = call_llm(self.reward_lm_config, api_input)
            except Exception:
                raw = ""
            try:
                self.logger.info(f"[RAW_REWARD] {raw}")
            except Exception:
                pass
            decision = self._parse_reward_decision(raw)
            winner = None
            if decision == PairwiseDecision.RESPONSE_1:
                winner = 1
            elif decision == PairwiseDecision.RESPONSE_2:
                winner = 2
            return RewardResponse(
                raw_response=raw,
                decision=decision,
                winner=winner,
                think=None,
                criteria=None,
                analysis=None,
                is_valid=decision != PairwiseDecision.UNDECIDED,
                parse_errors=[] if decision != PairwiseDecision.UNDECIDED else ["undecided"],
            )

        # No heuristic fallback — if reward LLM is not configured or failed, stay undecided
        return RewardResponse(
            raw_response="",
            decision=PairwiseDecision.UNDECIDED,
            winner=None,
            think=None,
            criteria=None,
            analysis=None,
            is_valid=False,
            parse_errors=["no reward_lm_config"],
        )

    def _knockout(self, base_req: PlanRequest, candidates: List[BlockInfo]) -> Optional[BlockInfo]:
        """Single-elimination tournament bracket with pairwise matches each round.

        - Use up to 16 candidates (first 16).
        - Pair them two-by-two: (0 vs 1), (2 vs 3), ...
        - Winners advance to the next round. If odd count, last advances by bye.
        - Continue until one winner remains.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        current: List[BlockInfo] = candidates[:16]
        round_idx = 0
        while len(current) > 1:
            next_round: List[BlockInfo] = []
            pair_idx = 0
            i = 0
            while i < len(current):
                if i + 1 >= len(current):
                    # Bye for the last unpaired candidate
                    next_round.append(current[i])
                    break
                a = current[i]
                b = current[i + 1]
                rr = self._build_reward_request(base_req, a, b)
                # Log pairwise comparison context
                try:
                    a_desc = self._describe_action(a.action)
                    b_desc = self._describe_action(b.action)
                    self.logger.info(f"[KO_PAIR] round={round_idx} pair={pair_idx}\nA: {a_desc}\nB: {b_desc}")
                except Exception:
                    pass
                score = self._score_pair(rr, round_idx=round_idx, pair_idx=pair_idx)
                winner_tag = "A"
                if score.winner == 2 or score.decision == PairwiseDecision.RESPONSE_2:
                    next_round.append(b)
                    winner_tag = "B"
                else:
                    next_round.append(a)
                pair_idx += 1
                # Log decision summary
                try:
                    self.logger.info(
                        f"[KO_DECISION] round={round_idx} pair={pair_idx-1} winner={winner_tag} decision={score.decision}\n"
                        f"  A: {a_desc}\n  B: {b_desc}"
                    )
                except Exception:
                    pass
                i += 2
            current = next_round
            round_idx += 1
        return current[0]

    # ---------------------------- Step 3: Notes/AGGREGATE Update ----------------------------
    def _update_aggregate_via_llm(self, winner_action: str) -> None:
        # If note LLM is not configured, just set plan_next and return
        try:
            if self.note_lm_config is None:
                prev = self.rt.get_aggregate() or AggregateInfo(note=[], evidence=[], plan_next="", answer_ready=False)
                prev.plan_next = winner_action
                self.rt.set_aggregate(prev)
                return
        except Exception:
            pass

        # System and user prompts
        system_text = f"{role_note_taker_agg}\n{note_taker_rules_v1}"
        last_agg_json = "{}"
        try:
            ag = self.rt.get_aggregate()
            if ag is not None:
                last_agg_json = json.dumps(ag.dict(), ensure_ascii=False)
        except Exception:
            pass

        user_text = context_note_taker_v1.format(
            intent=self.rt.get_intent() or "",
            observation=self._compose_observation_from_nodes(self.rt.get_obs_nodes_info()),
            last_aggregate=last_agg_json,
            action=winner_action or "None",
            start_url=self.rt.get_start_url() or "",
            current_url=self.rt.get_current_url() or "",
        )

        # Log note prompt (system and user)
        try:
            self.logger.info(f"[PROMPT_NOTE_SYSTEM]\n{system_text}")
            self.logger.info(f"[PROMPT_NOTE_USER]\n{user_text}")
        except Exception:
            pass

        # Build API input and call LLM
        raw = ""
        try:
            api_input = build_api_input_for_text(self.note_lm_config, system_text, user_text)
            raw = call_llm(self.note_lm_config, api_input)
        except Exception:
            raw = ""
        # Log raw note LLM response
        try:
            self.logger.info(f"[RAW_NOTE] {raw}")
        except Exception:
            pass

        # Parse AGGREGATE JSON
        new_agg: AggregateInfo | None = None
        if isinstance(raw, str) and raw.strip():
            try:
                import re as _re
                m = _re.search(r"\{[\s\S]*\}", raw)
                if m:
                    data = json.loads(m.group(0))
                    payload = data.get("AGGREGATE", data)
                    if isinstance(payload, dict):
                        # Normalize fields
                        note = payload.get("note") or []
                        evidence = payload.get("evidence") or []
                        plan_next = payload.get("plan_next") or ""
                        answer_ready = payload.get("answer_ready") if isinstance(payload.get("answer_ready"), bool) else False
                        new_agg = AggregateInfo(
                            note=list(note) if isinstance(note, list) else [],
                            evidence=list(evidence) if isinstance(evidence, list) else [],
                            plan_next=str(plan_next),
                            answer_ready=bool(answer_ready),
                        )
            except Exception:
                new_agg = None

        # Apply update or fallback
        try:
            if new_agg is None:
                prev = self.rt.get_aggregate() or AggregateInfo(note=[], evidence=[], plan_next="", answer_ready=False)
                prev.plan_next = winner_action
                self.rt.set_aggregate(prev)
            else:
                self.rt.set_aggregate(new_agg)
        except Exception:
            pass

    # ---------------------------- Public API ----------------------------
    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
        output_response: bool = False,
    ) -> Action:

        # Stage 1: BLOCK candidates
        policy_req = self._build_sample_request()
        candidates = self._sample_block_candidates(policy_req)
        self.rt.set_block_candidates(candidates)

        # Stage 2: Knockout
        winner = self._knockout(policy_req, candidates)
        self.rt.set_selected_block(winner)

        if not winner:
            raise ValueError("No candidate actions available")

        
        # Defer trajectory updates until after environment executes the action

        # Minimal thought-only log for the winner
        try:
            self.logger.info(f"winner: [THOUGHT] {winner.thought} [ACTION] {winner.action}")
        except Exception as e:
            raise ValueError(f"Failed to log winner thought: {winner.thought}") from e


        # Parse action
        try:
            action = create_id_based_action(winner.action)
        except Exception as e:
            raise ValueError(f"Failed to parse winner action: {winner.action}") from e

        # Execute in browser environment and update runtime
        if self.rt.has_environment():
            try:
                self.rt.execute_action(action, thought=winner.thought, action_str=winner.action)
            except Exception as e:
                # Log and continue to avoid aborting the run
                try:
                    self.logger.error(f"Environment execution failed: {e}")
                except Exception:
                    pass
        
        # Step 3: Update Aggregate with LLM (never fail the turn)
        try:
            self._update_aggregate_via_llm(winner.action)
        except Exception as e:
            try:
                self.logger.error(f"Note update failed: {e}")
            except Exception:
                pass

        return action