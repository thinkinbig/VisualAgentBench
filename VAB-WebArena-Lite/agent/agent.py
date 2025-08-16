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
        num_samples: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,

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

        self.logger = logging.getLogger("reward_guided_logger")
        
        # Check if the model is multimodal
        if ("gemini" in policy_lm_config.model or 
            "gpt-4" in policy_lm_config.model and "vision" in policy_lm_config.model or 
            policy_lm_config.provider in ["api", "finetune"]) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

        # Simple early stopping controls
        try:
            instruction_obj = getattr(prompt_constructor, "instruction", {})
            meta_cfg = instruction_obj.get("meta_data", {}) if isinstance(instruction_obj, dict) else {}
            self._reject_threshold: float = float(meta_cfg.get("reject_threshold", 2.0))
        except Exception:
            self._reject_threshold = 2.0
    
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
        
        # Generate multiple samples with nucleus sampling
        unique_action_keys: set[str] = set()
        action_key_counts: Dict[str, int] = {}

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
        
        # Build action history from trajectory
        action_history = []
        for i in range(1, len(trajectory), 2):  # Skip states, get actions
            if i < len(trajectory):
                action_info = trajectory[i]
                if isinstance(action_info, dict) and "action" in action_info:
                    action_str = str(action_info["action"])
                    # Clean up action string for readability
                    if "raw_prediction" in action_info:
                        action_str = action_info["raw_prediction"]
                    action_history.append(f"Step {len(action_history)+1}: {action_str}")
        
        action_history_text = "\n".join(action_history) if action_history else "No previous actions"

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
                    action_history=action_history_text,
                )
            except Exception:
                pass

        # Default prompt with action history
        prompt = (
            "You are a reward model that evaluates web agent actions. Given the current state, action history, and a proposed action, "
            "predict how good this action is for achieving the goal.\n\n"
            f"Goal: {intent}\n\n"
            f"Action History:\n{action_history_text}\n\n"
            f"Current State: {current_text}\n\n"
            f"Proposed Action: {proposed_action}\n\n"
            "Please evaluate this action considering the context of previous actions and provide a score from 0.0 to 10.0, where:\n"
            "- 0.0: Completely wrong action that will not help achieve the goal\n"
            "- 5.0: Neutral action that neither helps nor hurts\n"
            "- 10.0: Perfect action that directly helps achieve the goal\n\n"
            "Consider:\n"
            "- Does this action build logically on previous actions?\n"
            "- Is it a natural next step in the task progression?\n"
            "- Does it avoid repeating ineffective previous actions?\n\n"
            "Respond with: Score: X.X"
        )
        return prompt
    
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
        
        # Early rejection for very low scores
        if best_score < self._reject_threshold:
            self.logger.info(
                "Best score %.3f below reject_threshold %.3f, stopping",
                best_score, self._reject_threshold,
            )
            action = create_stop_action(f"Low confidence (score={best_score:.2f})")
            action["raw_prediction"] = best_response
            return action

        return best_action
    
    def reset(self, test_config_file: str) -> None:
        pass
