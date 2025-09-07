import logging
import json
import re
import os
import random
from typing import Optional, List, Dict, Any, Tuple

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
    PolicyRequest,
    PolicyResponse,
    BlockInfo,
    RewardRequest,
    RewardResponse,
    PairwiseDecision,
    PairwiseMatch,
)
from .parsers import BlockParser, RewardParser, ActionValidator, ObservationParser
from .sampling import NucleusSampler, CandidateSelector


from .runtime_manager import RuntimeManager
 
from .prompts.sample_p import (
    intro as sample_intro,
    render_prompt as render_sample_prompt,
)
from .prompts.reward_p import (
    role as reward_role,
    evaluation_summarized_v3 as reward_criteria,
    context_rft_v3 as reward_context_template,
)


class RewardGuidedAgent(Agent):
    """Reward-guided agent with BLOCK-only policy and pairwise knockout selection.

    Flow per turn:
    1) Build PolicyRequest from runtime meta.
    2) Sample BLOCK candidates (thought + bracket action string).
    3) Run pairwise knockout using reward prompt (stubbed scorer by default).
    4) Return parsed Action created from winner.action.
    """

    def __init__(
        self,
        action_set_tag: str,
        policy_lm_config: lm_config.LMConfig,
        reward_lm_config: lm_config.LMConfig,
        num_samples: int = 16,
        nucleus_sampler: Optional["NucleusSampler"] = None,
        max_steps: int = 30,
        num_calls: int = 5,
        max_tournament_candidates: int = 16,
        max_obs_length: int = 2048,
        max_retry: int = 3,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger("reward_guided_logger")
        self.action_set_tag = action_set_tag
        self.policy_lm_config = policy_lm_config
        self.reward_lm_config = reward_lm_config
        self.num_samples = max(1, int(num_samples))
        self.max_steps = max_steps
        self.num_calls = max(1, int(num_calls))
        self.max_tournament_candidates = max(1, int(max_tournament_candidates))
        self.max_obs_length = max(1, int(max_obs_length))
        self.max_retry = max(1, int(max_retry))

        # Use provided nucleus sampler or create default one
        self.nucleus_sampler = nucleus_sampler

        self.rt = RuntimeManager(max_steps=max_steps)


    # ---------------------------- Stage 1: Policy (BLOCK only) ----------------------------
    def _build_sample_request(self) -> PolicyRequest:
        # Build policy request with only the essential attributes
        return PolicyRequest(
            intent=self.rt.get_intent() or "",
            observation=self.rt.compose_observation_from_nodes(self.rt.get_obs_nodes_info()),
            current_url=self.rt.get_current_url(),
        )

    def _format_policy_prompt(self, req: PolicyRequest) -> tuple[str, str]:
        prev_thought = "(None)"
        prev_action = "(None)"
        try:
            cp = self.rt.get_checkpoint()
            blk = getattr(cp, "block", None) if cp else None
            if blk and isinstance(blk.action, str) and blk.action.strip():
                if isinstance(getattr(blk, "thought", None), str) and blk.thought.strip():
                    prev_thought = blk.thought.strip()
                prev_action = blk.action.strip()
        except Exception:
            pass
        
        # Build system text with examples
        system_text = f"{sample_intro}"
        
        user_text = render_sample_prompt(
            objective=req.intent or "",
            observation=req.observation or "",
            url=req.current_url or "",
            previous_thought=prev_thought,
            previous_action=prev_action,
        )
        return system_text, user_text



    def _sample_block_candidates(self, req: PolicyRequest) -> PolicyResponse:
        if self.policy_lm_config is None:
            raise ValueError("Policy LLM config is required for LLM-based sampling")
        
        # Multi-block sampling: generate multiple diverse blocks per call
        target_samples = self.num_samples
        
        # Generate fewer calls but each call produces more blocks
        num_calls = self.num_calls
        all_candidates: List[BlockInfo] = []
        seen_actions: set = set()  # Track seen actions for deduplication
        
        self.logger.info(f"[MULTI_BLOCK] Generating {num_calls} calls for best-of-{target_samples} selection")
        
        # Log policy prompt once (same for all calls)
        sys_txt, usr_txt = self._format_policy_prompt(req)
        try:
            self.logger.info(f"[PROMPT_POLICY_SYSTEM]\n{sys_txt}")
            self.logger.info(f"[PROMPT_POLICY_USER]\n{usr_txt}")
        except Exception:
            pass
        
        # Generate multiple diverse blocks per call
        for call_idx in range(num_calls):
            # Use aggressive sampling parameters for diversity
            temperature, top_p = self.nucleus_sampler.get_sampling_params(call_idx, num_calls)
            
            # Create dynamic LM config for this sampling attempt
            dynamic_config = self.nucleus_sampler.create_dynamic_lm_config(self.policy_lm_config, temperature, top_p)
            
            # Build prompt for this sampling attempt (reuse the same prompt)
            prompt = build_api_input_for_text(dynamic_config, sys_txt, usr_txt)
            
            self.logger.info(f"[MULTI_BLOCK] Call {call_idx + 1}/{num_calls}, temp={temperature:.2f}, top_p={top_p:.2f}")
            
            # call_llm already has retry mechanism built-in
            try:
                raw = call_llm(dynamic_config, prompt)
                
                # Log the raw response for this call
                try:
                    self.logger.info(f"[MULTI_BLOCK_RESPONSE] Call {call_idx + 1} raw response:\n{raw}")
                except Exception:
                    pass
                
                # Parse multiple BLOCKs from response
                blocks = BlockParser.parse_multiple_blocks(raw)
                if not blocks:
                    # If no blocks parsed, this is a parsing issue, not LLM issue
                    # Since call_llm already retried, we should continue to next call
                    self.logger.warning(f"[MULTI_BLOCK] No valid blocks parsed from call {call_idx + 1}")
                    continue
                    
            except Exception as e:
                # If call_llm fails after retries, log and continue to next call
                self.logger.error(f"[MULTI_BLOCK] Call {call_idx + 1} failed: {e}")
                continue
            
            # Log parsed blocks for this call
            try:
                self.logger.info(f"[MULTI_BLOCK_PARSED] Call {call_idx + 1} parsed {len(blocks)} blocks:")
                for i, blk in enumerate(blocks):
                    self.logger.info(f"  Block {i + 1}: {blk.action} | {blk.thought}")
            except Exception:
                pass
            
            # Process each block
            for blk in blocks:
                action_str = (blk.action or "").strip()
                # Hard-filter invalid ids/urls
                if not ActionValidator.is_valid_action(action_str, self.rt.get_obs_nodes_info()):
                    continue
                
                # Check for duplicates - skip if we've seen this action before
                if action_str in seen_actions:
                    continue
                
                # Normalize action field without backticks
                blk = BlockInfo(thought=blk.thought, action=action_str)
                all_candidates.append(blk)
                seen_actions.add(action_str)  # Track this action
                
                # Stop if we have enough candidates
                if len(all_candidates) >= target_samples * 2:  # Generate 2x target for selection
                    break
            
            # Stop if we have enough candidates
            if len(all_candidates) >= target_samples * 2:
                break
        
        self.logger.info(f"[MULTI_BLOCK] Generated {len(all_candidates)} unique candidates from {num_calls} calls")
        
        # Keep all candidates instead of limiting to target_samples
        selected_candidates = all_candidates
        
        self.logger.info(f"[MULTI_BLOCK] Selected {len(selected_candidates)} best candidates from {len(all_candidates)} total")
        
        # Log selected candidates
        for i, candidate in enumerate(selected_candidates):
            try:
                meaning = self.rt._describe_action(candidate.action)
                self.logger.info(f"[THOUGHT] #{i}: {candidate.thought} | [ACTION] {candidate.action} | [MEANING] {meaning}")
            except Exception:
                pass
        
        # Build and return PolicyResponse
        return PolicyResponse(
            candidates=selected_candidates,
            total_generated=len(all_candidates),
            unique_actions=len(set(c.action for c in selected_candidates)),
            is_valid=len(selected_candidates) > 0
        )

    # ---------------------------- Stage 2: Reward (pairwise knockout) ----------------------------
    def _build_reward_request(self, base: PolicyRequest, a: BlockInfo, b: BlockInfo) -> RewardRequest:
        return RewardRequest(
            intent=base.intent,
            observation=base.observation,
            trajectory=self.rt.compose_trajectory_from_meta(),
            start_url=self.rt.get_start_url(),
            current_url=base.current_url,
            thought1=a.thought,
            action1=a.action,
            thought2=b.thought,
            action2=b.action,
        )

    def _score_pair(self, rr: RewardRequest) -> RewardResponse:
        # Format reward prompt directly
        system_text = f"{reward_role}\n{reward_criteria}"
        user_text = reward_context_template.format(
            intent=rr.intent,
            observation=rr.observation,
            trajectory=rr.trajectory,
            start_url=rr.start_url,
            current_url=rr.current_url,
            thought1=rr.thought1,
            action1=rr.action1,
            thought2=rr.thought2,
            action2=rr.action2
        )
        
        # Log reward prompt
        try:
            self.logger.info(f"[PROMPT_REWARD_SYSTEM]\n{system_text}")
            self.logger.info(f"[PROMPT_REWARD_USER]\n{user_text}")
        except Exception:
            pass
        
        # call_llm already has retry mechanism built-in
        try:
            api_input = build_api_input_for_text(self.reward_lm_config, system_text, user_text)
            raw = call_llm(self.reward_lm_config, api_input)
            
            try:
                self.logger.info(f"[RAW_REWARD] {raw}")
            except Exception:
                pass
            
            # This will raise ValueError if parsing fails (call_llm already retried)
            try:
                decision = RewardParser.parse_decision(raw)
                winner = None
                if decision == PairwiseDecision.RESPONSE_1:
                    winner = 1
                elif decision == PairwiseDecision.RESPONSE_2:
                    winner = 2
                
                return RewardResponse(
                    raw_response=raw,
                    decision=decision,
                    winner=winner,
                    is_valid=True,
                )
            except ValueError as e:
                # If parsing fails, return invalid response
                self.logger.error(f"Failed to parse reward decision: {e}")
                return RewardResponse(
                    raw_response=raw,
                    decision=PairwiseDecision.RESPONSE_1,  # Default fallback
                    winner=1,
                    is_valid=False,
                )
                    
        except Exception as e:
            # If call_llm fails after retries, return invalid response
            self.logger.error(f"Reward LLM call failed: {e}")
            return RewardResponse(
                raw_response="",
                decision=PairwiseDecision.RESPONSE_1,  # Default fallback
                winner=1,
                is_valid=False,
            )

    def _knockout(self, base_req: PolicyRequest, candidates: List[BlockInfo]) -> Optional[BlockInfo]:
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

        current: List[BlockInfo] = candidates[:self.max_tournament_candidates]
        round_idx = 0
        
        # Log tournament start
        try:
            self.logger.info(f"[TOURNAMENT_START] Starting knockout tournament with {len(current)} candidates")
        except Exception:
            pass
            
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
                    a_desc = self.rt._describe_action(a.action)
                    b_desc = self.rt._describe_action(b.action)
                    self.logger.info(f"[TOURNAMENT_MATCH] Round {round_idx + 1}, Match {pair_idx + 1}:\n  Candidate A: {a_desc}\n  Candidate B: {b_desc}")
                except Exception:
                    pass
                score = self._score_pair(rr)
                # Record the pairwise match for visualization
                self.rt.record_pair(round_idx, i, i + 1, rr, score)
                
                winner_tag = "A"
                if score.winner == 2 or score.decision == PairwiseDecision.RESPONSE_2:
                    next_round.append(b)
                    winner_tag = "B"
                else:
                    next_round.append(a)
                pair_idx += 1
                # Log decision summary
                try:
                    decision_desc = {
                        PairwiseDecision.RESPONSE_1: "Selected Candidate A",
                        PairwiseDecision.RESPONSE_2: "Selected Candidate B"
                    }.get(score.decision, str(score.decision))
                    
                    self.logger.info(
                        f"[TOURNAMENT_RESULT] Round {round_idx + 1}, Match {pair_idx} Result: {decision_desc}\n"
                        f"  Winner: Candidate {winner_tag}\n"
                        f"  Candidate A: {a_desc}\n"
                        f"  Candidate B: {b_desc}"
                    )
                except Exception:
                    pass
                i += 2
            current = next_round
            round_idx += 1
            
            # Log round completion
            try:
                self.logger.info(f"[TOURNAMENT_ROUND] Round {round_idx} completed, {len(current)} candidates advance to next round")
            except Exception:
                pass
                
        return current[0]


    # ---------------------------- Public API ----------------------------
    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
        output_response: bool = False,
    ) -> Action:
        # Clear previous round samples at the start of each turn
        self.rt.clear_current_round_samples()
        
        # Ensure runtime has fresh intent/meta before building prompts
        try:
            # Single entrypoint to set meta and ensure initial AXTree when missing
            self.rt.bootstrap_turn(trajectory=trajectory, intent=intent, meta_data=meta_data)
        except Exception:
            pass

        # Stage 1: BLOCK candidates
        policy_req = self._build_sample_request()
        policy_resp = self._sample_block_candidates(policy_req)
        candidates = policy_resp.candidates
        
        # Record candidates in trajectory tree
        try:
            self.rt.record_candidates(candidates)
        except Exception:
            pass

        # Log all candidates in a consolidated list (Thought | Action | Meaning)
        try:
            self.logger.info(f"[CANDIDATES_GENERATED] Generated {len(candidates)} candidate actions:")
            for idx, c in enumerate(candidates):
                meaning = self.rt._describe_action(c.action)
                self.logger.info(f"  Candidate {idx + 1}: {meaning}")
                self.logger.info(f"    Thought: {c.thought}")
                self.logger.info(f"    Action: {c.action}")
        except Exception:
            pass

        # Stage 2: Knockout
        winner = self._knockout(policy_req, candidates)
        self.rt.set_selected_block(winner)

        if not winner:
            raise ValueError("No candidate actions available")

        
        # Defer trajectory updates until after environment executes the action

        # Log the final winner
        try:
            winner_meaning = self.rt._describe_action(winner.action)
            self.logger.info(f"[TOURNAMENT_WINNER] Selected action: {winner_meaning}")
            self.logger.info(f"  Thought: {winner.thought}")
            self.logger.info(f"  Action: {winner.action}")
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
        


        # Note: Trajectory tree is kept in memory for final visualization

        return action



