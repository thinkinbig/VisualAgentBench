import argparse
import logging
import json
import copy
from typing import Any, Optional, List, Dict, Tuple
from types import SimpleNamespace
import time

from beartype import beartype
from PIL import Image
from llms.utils import build_api_input_for_text, call_llm

from agent.prompts import *
from agent.config_schema import load_and_validate_instruction
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_webrl_id_based_action,
    create_stop_action,
)
from browser_env.utils import StateInfo
from llms import (
    call_llm,
    lm_config,
)
from llms.tokenizers import Tokenizer


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
        planner_ip = None
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.planner_ip = planner_ip
        
        # Check if the model is multimodal.
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model or lm_config.provider in ["api", "finetune"]) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print(
                    "WARNING: Input image provided but no image captioner available."
                )

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            if self.planner_ip is not None and self.planner_ip != "":
                response = call_llm(lm_config, prompt, 'EMPTY', self.planner_ip)
            else:
                response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == 'webrl_id':
                    action = create_webrl_id_based_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    # Optionally read instruction JSON to pick up defaults for policy model and action_set_tag
    instruction_obj: dict[str, Any] | None = None
    try:
        instruction_obj = json.loads(load_and_validate_instruction(args.instruction_path).json())
    except Exception:
        instruction_obj = None

    # Build policy (action) model config, allowing JSON overrides under policy_model
    if isinstance(instruction_obj, dict) and isinstance(instruction_obj.get("policy_model"), dict):
        policy_section = instruction_obj["policy_model"]
        policy_provider = policy_section.get("provider") or args.provider
        policy_model = policy_section.get("model") or args.model
        policy_mode = policy_section.get("mode") or args.mode
        policy_model_endpoint = policy_section.get("model_endpoint", getattr(args, "model_endpoint", None))

        policy_args = SimpleNamespace(
            provider=policy_provider,
            model=policy_model,
            mode=policy_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            context_length=args.context_length,
            max_tokens=args.max_tokens,
            stop_token=args.stop_token,
            max_obs_length=args.max_obs_length,
            max_retry=args.max_retry,
            model_endpoint=policy_model_endpoint,
        )
        # Map optional per-model gen overrides into args before constructing LMConfig
        gen_overrides = policy_section.get("gen") or {}
        if isinstance(gen_overrides, dict):
            for k, v in gen_overrides.items():
                if v is not None:
                    setattr(policy_args, k, v)
        llm_config = lm_config.construct_llm_config(policy_args)  # type: ignore[arg-type]
    else:
        llm_config = lm_config.construct_llm_config(args)

    # Allow JSON root-level default for action_set_tag if user didn't explicitly change it
    if isinstance(instruction_obj, dict) and "action_set_tag" in instruction_obj:
        try:
            # Heuristic: only override if args has default value
            if getattr(args, "action_set_tag", "id_accessibility_tree") == "id_accessibility_tree":
                args.action_set_tag = instruction_obj["action_set_tag"]
        except Exception:
            pass
    
    # Reward model config will be constructed later for reward_guided agent using instruction JSON

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        # Build tokenizer from the computed policy llm_config
        tokenizer = Tokenizer(llm_config.provider, llm_config.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
            planner_ip=args.planner_ip
        )
    elif args.agent_type == "reward_guided":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        # Build tokenizer from the computed policy llm_config
        tokenizer = Tokenizer(llm_config.provider, llm_config.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        # Build reward model config from instruction JSON (new format only) with CLI fallbacks
        instruction_obj = getattr(prompt_constructor, "instruction", {})
        reward_model_section = instruction_obj.get("reward_model") if isinstance(instruction_obj, dict) else None

        reward_provider = reward_model_section.get("provider") or getattr(args, "reward_provider", None) or args.provider
        reward_model = reward_model_section.get("model") or getattr(args, "reward_model", None) or args.model
        reward_mode = reward_model_section.get("mode") or getattr(args, "mode", None)
        reward_model_endpoint = reward_model_section.get("model_endpoint") or getattr(args, "reward_model_endpoint", None) or getattr(args, "model_endpoint", None)

        reward_args = SimpleNamespace(
            provider=reward_provider,
            model=reward_model,
            mode=reward_mode or args.mode,
            temperature=args.temperature,
            top_p=args.top_p,
            context_length=args.context_length,
            max_tokens=args.max_tokens,
            stop_token=args.stop_token,
            max_obs_length=args.max_obs_length,
            max_retry=args.max_retry,
            model_endpoint=reward_model_endpoint,
        )
        # Map optional per-model gen overrides into args before constructing LMConfig
        if isinstance(reward_model_section, dict):
            gen_overrides = reward_model_section.get("gen") or {}
            if isinstance(gen_overrides, dict):
                for k, v in gen_overrides.items():
                    if v is not None:
                        setattr(reward_args, k, v)
        reward_llm_config = lm_config.construct_llm_config(reward_args)  # type: ignore[arg-type]
        # Read optional defaults from instruction file (new format only: root-level fields)
        # Use instruction values when present unless CLI explicitly overrides (CLI wins).
        def _prefer_instruction_root(arg_value, key, builtin_default):
            inst_value = instruction_obj.get(key) if isinstance(instruction_obj, dict) else None
            if inst_value is not None and arg_value == builtin_default:
                return inst_value
            return arg_value if arg_value is not None else (inst_value if inst_value is not None else builtin_default)

        eff_num_samples = _prefer_instruction_root(getattr(args, "num_samples", 20), "num_samples", 20)
        eff_temperature = _prefer_instruction_root(getattr(args, "temperature", 1.0), "temperature", 1.0)
        eff_top_p = _prefer_instruction_root(getattr(args, "top_p", 0.9), "top_p", 0.9)
        eff_max_refinements = _prefer_instruction_root(getattr(args, "max_refinements", 2), "max_refinements", 2)
        agent = RewardGuidedAgent(
            action_set_tag=args.action_set_tag,
            policy_lm_config=llm_config,
            reward_lm_config=reward_llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
            planner_ip=args.planner_ip,
            num_samples=eff_num_samples,
            temperature=eff_temperature,
            top_p=eff_top_p,
            max_refinements=eff_max_refinements,
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent


class RewardGuidedAgent(Agent):
    """Reward-guided Trajectory Search Agent as described in the paper"""
    
    def __init__(
        self,
        action_set_tag: str,
        policy_lm_config: lm_config.LMConfig,
        reward_lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn=None,
        planner_ip=None,
        num_samples: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_refinements: int = 2
    ) -> None:
        super().__init__()
        self.action_set_tag = action_set_tag
        self.policy_lm_config = policy_lm_config
        self.reward_lm_config = reward_lm_config
        self.prompt_constructor = prompt_constructor
        self.captioning_fn = captioning_fn
        self.planner_ip = planner_ip
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_refinements = max_refinements
        self.logger = logging.getLogger("reward_guided_logger")
        
        # Check if the model is multimodal
        if ("gemini" in policy_lm_config.model or 
            "gpt-4" in policy_lm_config.model and "vision" in policy_lm_config.model or 
            policy_lm_config.provider in ["api", "finetune"]) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

        # Strong-convergence controls (instance-local; no external metadata required)
        self._patience_counter: int = 0
        self._prev_effective_score: float | None = None
        # Defaults; optionally tuned from instruction if present
        try:
            instruction_obj = getattr(prompt_constructor, "instruction", {})
            meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
            self._patience_limit: int = int(meta_cfg.get("patience_limit", 5))
            self._min_progress_epsilon: float = float(meta_cfg.get("min_progress_epsilon", 0.1))
        except Exception:
            self._patience_limit = 5
            self._min_progress_epsilon = 0.1
    
    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag
    
    def _generate_action_candidates(
        self, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any], 
        images: Optional[List[Image.Image]] = None
    ) -> List[Tuple[str, Action]]:
        """Generate action candidates using nucleus sampling"""
        candidates = []
        self.logger.info("Generating action candidates (num_samples=%d)", self.num_samples)
        
        # Create page screenshot image for multimodal models
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(page_screenshot_arr)
        
        # Caption the input image, if provided
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print("WARNING: Input image provided but no image captioner available.")
        
        # Generate multiple samples with nucleus sampling (with early-exit)
        # Early-exit controls from instruction.meta_data
        try:
            instruction_obj = getattr(self.prompt_constructor, "instruction", {})
            meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
            early_unique_target = int(meta_cfg.get("target_unique_candidates", 3))
            early_majority_threshold = int(meta_cfg.get("early_majority_threshold", 2))
            sample_time_budget_sec = float(meta_cfg.get("sample_time_budget_sec", 0.0))  # 0 means no budget
        except Exception:
            early_unique_target = 3
            early_majority_threshold = 2
            sample_time_budget_sec = 0.0

        unique_action_keys: set[str] = set()
        action_key_counts: Dict[str, int] = {}
        sample_loop_start = time.time()

        for sample_idx in range(self.num_samples):
            try:
                if self.multimodal_inputs:
                    prompt = self.prompt_constructor.construct(
                        trajectory, intent, page_screenshot_img, images, meta_data
                    )
                else:
                    prompt = self.prompt_constructor.construct(
                        trajectory, intent, meta_data
                    )
                
                # Create config with sampling parameters
                sample_config = copy.deepcopy(self.policy_lm_config)
                sample_config.gen_config["temperature"] = self.temperature
                sample_config.gen_config["top_p"] = self.top_p
                
                if self.planner_ip is not None and self.planner_ip != "":
                    response = call_llm(sample_config, prompt, 'EMPTY', self.planner_ip)
                else:
                    response = call_llm(sample_config, prompt)
                
                force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")
                response = f"{force_prefix}{response}"
                self.logger.debug("[Sample %d] Raw LLM response: %r", sample_idx, response)
                
                # Parse action (with fallbacks to be robust to formatting)
                parsed_response = self.prompt_constructor.extract_action(response)
                self.logger.debug("[Sample %d] Extracted action string: %s", sample_idx, parsed_response)
                try:
                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(parsed_response)
                    elif self.action_set_tag == "som":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == 'webrl_id':
                        # Primary: parse WebRL-style
                        action = create_webrl_id_based_action(parsed_response)
                    else:
                        raise ValueError(f"Unknown action type {self.action_set_tag}")
                except Exception:
                    # Fallbacks: many prompts still emit id-style like "goto [...]"
                    try:
                        action = create_id_based_action(parsed_response)
                    except Exception:
                        # As a last resort, try playwright
                        action = create_playwright_action(parsed_response)
                
                action["raw_prediction"] = response
                candidates.append((response, action))
                self.logger.debug("[Sample %d] Parsed action OK: %s", sample_idx, str(action))
                # Early-exit checks
                action_key = str(action)
                action_key_counts[action_key] = action_key_counts.get(action_key, 0) + 1
                unique_action_keys.add(action_key)
                if early_unique_target > 0 and len(unique_action_keys) >= early_unique_target:
                    self.logger.debug(
                        "Early-exit: reached target_unique_candidates=%d at sample %d",
                        early_unique_target, sample_idx,
                    )
                    break
                if early_majority_threshold > 0 and action_key_counts[action_key] >= early_majority_threshold:
                    self.logger.debug(
                        "Early-exit: action %s reached majority threshold=%d at sample %d",
                        action_key, early_majority_threshold, sample_idx,
                    )
                    break
                if sample_time_budget_sec > 0.0 and (time.time() - sample_loop_start) >= sample_time_budget_sec:
                    self.logger.debug(
                        "Early-exit: sampling time budget %.2fs exceeded at sample %d",
                        sample_time_budget_sec, sample_idx,
                    )
                    break
                
            except Exception as e:
                self.logger.warning("Error generating sample %d: %s", sample_idx, e, exc_info=True)
                continue
        
        return candidates
    
    def _score_action_with_reward_model(
        self, 
        action: Action, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any]
    ) -> float:
        """Score an action using the reward model"""
        try:
            # Create reward evaluation prompt
            reward_prompt = self._create_reward_prompt(action, trajectory, intent, meta_data)
            self.logger.debug("Reward prompt (text):\n%s", reward_prompt)
            
            # Get reward score from reward model (build provider-specific API input)
            api_input = build_api_input_for_text(
                self.reward_lm_config,
                "You are a reward model for web agents. Read the user's content and output only: Score: X.X",
                reward_prompt,
            )
            response = call_llm(self.reward_lm_config, api_input)
            self.logger.debug("Reward model response: %r", response)
            
            # Extract reward score from response (configurable via instruction meta_data)
            try:
                # Try to extract numerical score
                import re
                instruction_obj = getattr(self.prompt_constructor, "instruction", {})
                meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
                score_regex = meta_cfg.get("reward_score_regex", r'Score:\s*([\d.]+)')
                score_match = re.search(score_regex, response)
                if score_match:
                    score = float(score_match.group(1))
                    self.logger.debug("Extracted reward score: %s", score)
                    return score
                
                # Fallback: try to find any number in the response
                numbers = re.findall(r'[\d.]+', response)
                if numbers:
                    score = float(numbers[0])
                    self.logger.debug("Fallback extracted numeric score: %s", score)
                    return score
                
                # If no score found, return a default score
                self.logger.debug("No numeric score found, defaulting to 0.0")
                return 0.0
                
            except (ValueError, IndexError):
                return 0.0
                
        except Exception as e:
            self.logger.warning("Error scoring action with reward model: %s", e, exc_info=True)
            return 0.0

    def _score_actions_with_reward_model(
        self,
        actions: List[Action],
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
    ) -> List[float]:
        """Batch score multiple actions in a single reward-model call.
        Returns list of scores aligned with input actions. Falls back to per-action if parsing fails.
        """
        try:
            # Build batch prompt
            current_text = trajectory[-1]["observation"].get("text", "No text observation")
            proposals: List[str] = []
            for idx, act in enumerate(actions, start=1):
                raw_pred = act.get("raw_prediction", str(act))
                if isinstance(raw_pred, str) and "```" in raw_pred:
                    try:
                        raw_pred = self.prompt_constructor.extract_action(raw_pred)
                    except Exception:
                        raw_pred = raw_pred.replace("```", "").strip()
                proposals.append(f"{idx}. {raw_pred}")

            user_text = (
                f"Goal: {intent}\n\n"
                f"Current State: {current_text}\n\n"
                "Proposed Actions (each line starts with index.):\n" +
                "\n".join(proposals) +
                "\n\nFor each action i, reply on a separate line strictly as: i: Score: X.X"
            )
            system_text = (
                "You are a reward model for web agents. Evaluate each proposed next action independently and output scores only."
            )
            api_input = build_api_input_for_text(self.reward_lm_config, system_text, user_text)
            response = call_llm(self.reward_lm_config, api_input)
            self.logger.debug("Batch reward model response: %r", response)

            import re
            pattern = re.compile(r"^(\d+)\s*:\s*Score:\s*([\d.]+)", re.MULTILINE)
            matches = pattern.findall(response)
            scores: List[float] = [0.0] * len(actions)
            for idx_str, score_str in matches:
                try:
                    i = int(idx_str)
                    if 1 <= i <= len(actions):
                        scores[i - 1] = float(score_str)
                except Exception:
                    continue

            # If parsing failed for all, fallback to per-action scoring
            if all(s == 0.0 for s in scores):
                return [
                    self._score_action_with_reward_model(a, trajectory, intent, meta_data)
                    for a in actions
                ]
            return scores
        except Exception as e:
            self.logger.warning("Batch reward scoring failed, falling back to single: %s", e, exc_info=True)
            return [
                self._score_action_with_reward_model(a, trajectory, intent, meta_data)
                for a in actions
            ]

    def _get_turn_index(self, meta_data: dict[str, Any], trajectory: Trajectory) -> int:
        """Infer the current turn index (0-based) from metadata or trajectory."""
        try:
            if isinstance(meta_data, dict) and isinstance(meta_data.get("action_history"), list):
                # First entry is a placeholder (e.g., "None")
                return max(0, len(meta_data["action_history"]) - 1)
        except Exception:
            pass
        # Fallback: estimate from trajectory length (state, action, state, ...)
        try:
            return max(0, (len(trajectory) - 1) // 2)
        except Exception:
            return 0

    def _compute_turn_penalty(self, turn_index: int) -> float:
        """Monotonically increasing penalty with turns to discourage dithering.
        Linear schedule: base 0.2 per turn, capped at 5.0.
        """
        try:
            penalty = 0.2 * float(turn_index)
        except Exception:
            penalty = 0.0
        return float(min(5.0, max(0.0, penalty)))
    
    def _create_reward_prompt(
        self, 
        action: Action, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any]
    ) -> str:
        """Create a prompt for the reward model to evaluate an action"""
        # Prefer a clean action string without code fences
        raw_pred = action.get("raw_prediction", str(action))
        proposed_action = raw_pred
        if isinstance(raw_pred, str) and "```" in raw_pred:
            try:
                proposed_action = self.prompt_constructor.extract_action(raw_pred)
            except Exception:
                proposed_action = raw_pred.replace("```", "").strip()

        current_text = trajectory[-1]["observation"].get("text", "No text observation")

        # Allow instruction meta_data to override reward prompt template
        try:
            instruction_obj = getattr(self.prompt_constructor, "instruction", {})
            meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
        except Exception:
            meta_cfg = {}
        template = meta_cfg.get("reward_prompt_template") if isinstance(meta_cfg, dict) else None
        if isinstance(template, str) and template:
            try:
                return template.format(
                    objective=intent,
                    current_state=current_text,
                    proposed_action=proposed_action,
                )
            except Exception:
                pass

        # Default prompt
        prompt = (
            "You are a reward model that evaluates web agent actions. Given the current state and an action, "
            "predict how good this action is for achieving the goal.\n\n"
            f"Goal: {intent}\n\n"
            f"Current State: {current_text}\n\n"
            f"Proposed Action: {proposed_action}\n\n"
            "Please evaluate this action and provide a score from 0.0 to 10.0, where:\n"
            "- 0.0: Completely wrong action that will not help achieve the goal\n"
            "- 5.0: Neutral action that neither helps nor hurts\n"
            "- 10.0: Perfect action that directly helps achieve the goal\n\n"
            "Respond with: Score: X.X"
        )
        return prompt
    
    def _refine_action(
        self, 
        action: Action, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any],
        refinement_feedback: str
    ) -> Tuple[str, Action]:
        """Refine an action based on feedback from the reward model"""
        try:
            # Create refinement prompt (use the provided feedback string)
            instruction_obj = getattr(self.prompt_constructor, "instruction", {})
            meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
            context_template = meta_cfg.get("refinement_context_template") if isinstance(meta_cfg, dict) else None

            if isinstance(context_template, str) and context_template:
                try:
                    refinement_context = context_template.format(
                        objective=intent,
                        current_state=trajectory[-1]["observation"].get("text", "No text observation"),
                        original_action=action.get("raw_prediction", str(action)),
                        feedback=refinement_feedback,
                    )
                except Exception:
                    refinement_context = None
            else:
                refinement_context = None

            if not refinement_context:
                refinement_context = (
                    "You are a web agent that needs to refine an action based on feedback.\n\n"
                    f"Goal: {intent}\n\n"
                    f"Current State: {trajectory[-1]['observation'].get('text', 'No text observation')}\n\n"
                    f"Original Action: {action.get('raw_prediction', str(action))}\n\n"
                    f"Feedback: {refinement_feedback}\n\n"
                    "Please provide a refined action that addresses the feedback and better achieves the goal."
                )
            
            if self.multimodal_inputs:
                page_screenshot_arr = trajectory[-1]["observation"]["image"]
                page_screenshot_img = Image.fromarray(page_screenshot_arr)
                prompt = self.prompt_constructor.construct(
                    trajectory, intent, page_screenshot_img, None, meta_data
                )
            else:
                prompt = self.prompt_constructor.construct(
                    trajectory, intent, meta_data
                )
            
            # Add refinement context
            prompt += f"\n\nRefinement Context:\n{refinement_context}"
            
            if self.planner_ip is not None and self.planner_ip != "":
                response = call_llm(self.policy_lm_config, prompt, 'EMPTY', self.planner_ip)
            else:
                response = call_llm(self.policy_lm_config, prompt)
            
            force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            
            # Parse refined action
            parsed_response = self.prompt_constructor.extract_action(response)
            if self.action_set_tag == "id_accessibility_tree":
                refined_action = create_id_based_action(parsed_response)
            elif self.action_set_tag == "playwright":
                refined_action = create_playwright_action(parsed_response)
            elif self.action_set_tag == "som":
                refined_action = create_id_based_action(parsed_response)
            elif self.action_set_tag == 'webrl_id':
                refined_action = create_webrl_id_based_action(parsed_response)
            else:
                raise ValueError(f"Unknown action type {self.action_set_tag}")
            
            refined_action["raw_prediction"] = response
            return response, refined_action
            
        except Exception as e:
            print(f"Error refining action: {e}")
            return action.get("raw_prediction", ""), action
    
    @beartype
    def next_action(
        self, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any], 
        images: Optional[List[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        """Generate next action using reward-guided trajectory search"""
        
        # Step 1: Generate action candidates using nucleus sampling
        candidates = self._generate_action_candidates(trajectory, intent, meta_data, images)
        self.logger.info("Generated %d candidate actions", len(candidates))
        
        if not candidates:
            # Fallback to default action
            action = create_none_action()
            action["raw_prediction"] = "Failed to generate candidates"
            return action
        
        # Step 2: Count action frequencies and select top candidates
        action_counts = {}
        for response, action in candidates:
            action_key = str(action)
            if action_key in action_counts:
                action_counts[action_key].append((response, action))
            else:
                action_counts[action_key] = [(response, action)]
        
        # Sort by frequency and select top candidates
        sorted_candidates = sorted(action_counts.items(), key=lambda x: len(x[1]), reverse=True)
        top_candidates = []
        for action_key, action_list in sorted_candidates[:3]:  # Top 3 most frequent for efficiency
            # Use the first occurrence of each unique action
            top_candidates.append(action_list[0])
        self.logger.debug("Top %d unique candidates selected for scoring", len(top_candidates))
        
        # Step 3: Score candidates using reward model (batch for speed)
        scored_candidates = []
        try:
            actions_for_batch = [a for _, a in top_candidates]
            scores = self._score_actions_with_reward_model(actions_for_batch, trajectory, intent, meta_data)
            for (response, action), score in zip(top_candidates, scores):
                scored_candidates.append((score, response, action))
                self.logger.info("Candidate scored: score=%.3f, action=%s", score, str(action))
        except Exception:
            # Fallback: score one by one
            for response, action in top_candidates:
                score = self._score_action_with_reward_model(action, trajectory, intent, meta_data)
                scored_candidates.append((score, response, action))
                self.logger.info("Candidate scored: score=%.3f, action=%s", score, str(action))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Step 4: Select best action
        best_score, best_response, best_action = scored_candidates[0]
        self.logger.info("Best action selected: score=%.3f, action=%s", best_score, str(best_action))
        
        if output_response:
            print(f'Agent: {best_response}', flush=True)
            print(f'Reward Score: {best_score}', flush=True)
        
        # Apply turn-based penalty for thresholds/decisions
        turn_index = self._get_turn_index(meta_data, trajectory)
        penalty = self._compute_turn_penalty(turn_index)
        effective_best_score = max(0.0, best_score - penalty)
        self.logger.debug("Turn=%d, penalty=%.2f, effective_best_score=%.2f", turn_index, penalty, effective_best_score)

        # Early rejection for very low effective scores
        try:
            instruction_obj = getattr(self.prompt_constructor, "instruction", {})
            meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
            reject_threshold = float(meta_cfg.get("reject_threshold", 2.0))
            low_score_behavior = str(meta_cfg.get("low_score_behavior", "resample")).strip().lower()
        except Exception:
            reject_threshold = 2.0
            low_score_behavior = "stop"

        if effective_best_score < reject_threshold:
            self.logger.info(
                "Best score %.3f below reject_threshold %.3f; behavior=%s",
                effective_best_score, reject_threshold, low_score_behavior,
            )
            if low_score_behavior == "stop":
                action = create_stop_action(f"Low confidence (eff_score={effective_best_score:.2f}, penalty={penalty:.2f})")
                action["raw_prediction"] = best_response
                return action
            elif low_score_behavior == "none":
                action = create_none_action()
                action["raw_prediction"] = f"Low confidence (eff_score={effective_best_score:.2f}, penalty={penalty:.2f})"
                return action
            else:
                # resample once with slightly higher diversity
                old_temp, old_top_p = self.temperature, self.top_p
                try:
                    self.temperature = min(self.temperature + 0.2, 1.3)
                    self.top_p = min(self.top_p + 0.1, 0.95)
                    candidates2 = self._generate_action_candidates(trajectory, intent, meta_data, images)
                finally:
                    self.temperature, self.top_p = old_temp, old_top_p

                action_counts2 = {}
                for response2, action2 in candidates2:
                    key2 = str(action2)
                    action_counts2.setdefault(key2, []).append((response2, action2))
                sorted2 = sorted(action_counts2.items(), key=lambda x: len(x[1]), reverse=True)
                top2 = [lst[0] for _, lst in sorted2[:3]]

                scored2: list[tuple[float, str, Action]] = []
                if top2:
                    try:
                        actions_for_batch2 = [a for _, a in top2]
                        scores2 = self._score_actions_with_reward_model(actions_for_batch2, trajectory, intent, meta_data)
                        for (resp2, act2), sc2 in zip(top2, scores2):
                            scored2.append((sc2, resp2, act2))
                    except Exception:
                        for resp2, act2 in top2:
                            sc2 = self._score_action_with_reward_model(act2, trajectory, intent, meta_data)
                            scored2.append((sc2, resp2, act2))

                if scored2:
                    scored2.sort(key=lambda x: x[0], reverse=True)
                    new_best_score, new_best_resp, new_best_action = scored2[0]
                    new_eff = max(0.0, new_best_score - penalty)
                    self.logger.info("Resample eff_score=%.3f (prev_eff=%.3f)", new_eff, effective_best_score)
                    if new_eff >= reject_threshold:
                        best_score, best_response, best_action = new_best_score, new_best_resp, new_best_action
                        effective_best_score = new_eff
                        if output_response:
                            print(f"Resample accepted: eff_score {new_eff:.1f}", flush=True)
                    else:
                        # still low: terminate decisively
                        action = create_stop_action(
                            f"Low confidence after resample (eff_score={new_eff:.2f}, penalty={penalty:.2f})"
                        )
                        action["raw_prediction"] = best_response
                        return action
                else:
                    action = create_stop_action("Failed to generate candidates on resample")
                    action["raw_prediction"] = best_response
                    return action

        # Optional: skip refinement if score already high enough
        try:
            instruction_obj = getattr(self.prompt_constructor, "instruction", {})
            meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
            skip_threshold = float(meta_cfg.get("skip_refinement_threshold", 8.5))
        except Exception:
            skip_threshold = 8.5
        if effective_best_score >= skip_threshold:
            if output_response:
                print(f"Skipping refinement (eff_score {effective_best_score:.1f} >= {skip_threshold}, penalty {penalty:.2f})", flush=True)
            return best_action

        # Step 5: Refinement process (up to max_refinements times)
        current_action = best_action
        current_score = best_score
        refinement_count = 0
        
        while refinement_count < self.max_refinements:
            # Get feedback from reward model (excluding the actual score)
            feedback_prompt = self._create_reward_prompt(current_action, trajectory, intent, meta_data)
            try:
                api_input_fb = build_api_input_for_text(
                    self.reward_lm_config,
                    "You are a reward model for web agents. Read the user's content and output critique and a single action in the same format, wrapped in code fences.",
                    feedback_prompt,
                )
                feedback_response = call_llm(self.reward_lm_config, api_input_fb)
            except Exception as e:
                self.logger.warning("Reward model failed during refinement feedback: %s", e, exc_info=True)
                break
            
            # Remove score information to get pure feedback
            feedback = feedback_response.replace(f"Score: {current_score}", "").strip()
            self.logger.debug("Refinement feedback: %r", feedback)
            
            # Refine the action
            refined_response, refined_action = self._refine_action(
                current_action, trajectory, intent, meta_data, feedback
            )
            self.logger.debug("Refined response: %r", refined_response)
            self.logger.debug("Refined action: %s", str(refined_action))
            
            # Score the refined action
            refined_score = self._score_action_with_reward_model(
                refined_action, trajectory, intent, meta_data
            )
            self.logger.info("Refined action scored: %.3f (prev=%.3f)", refined_score, current_score)
            
            if output_response:
                print(f'Refinement {refinement_count + 1}: {refined_response}', flush=True)
                print(f'Refined Score: {refined_score}', flush=True)
            
            # Only accept refinement if it improves the score by a minimum margin
            try:
                instruction_obj = getattr(self.prompt_constructor, "instruction", {})
                meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
                min_improve = float(meta_cfg.get("min_refine_improvement", 0.5))
            except Exception:
                min_improve = 0.5
            if refined_score >= current_score + min_improve:
                current_action = refined_action
                current_score = refined_score
                refinement_count += 1
                
                if output_response:
                    print(f'Refinement accepted! New score: {refined_score} (Δ≥{min_improve})', flush=True)
                self.logger.info("Refinement %d accepted: new_score=%.3f (min_improve=%.2f)", refinement_count, refined_score, min_improve)
            else:
                if output_response:
                    print(f'Refinement rejected. Δ<{min_improve}.', flush=True)
                self.logger.info("Refinement %d rejected: refined_score=%.3f < current_score+min_improve=%.3f", refinement_count + 1, refined_score, current_score + min_improve)
                break
        
        # Strong convergence: patience on lack of improvement across turns
        # Recompute effective score for the (possibly refined) current action
        effective_current_score = max(0.0, current_score - penalty)
        if self._prev_effective_score is None:
            self._prev_effective_score = effective_current_score
        else:
            improvement = effective_current_score - self._prev_effective_score
            if improvement < self._min_progress_epsilon:
                self._patience_counter += 1
            else:
                self._patience_counter = 0
            self._prev_effective_score = effective_current_score
            if self._patience_counter >= self._patience_limit:
                stop_action = create_stop_action(
                    f"No progress for {self._patience_counter} turns (Δ<{self._min_progress_epsilon:.2f}); stopping"
                )
                stop_action["raw_prediction"] = best_response
                return stop_action

        return current_action
    
    def reset(self, test_config_file: str) -> None:
        pass
