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

    

    # Allow JSON root-level default for action_set_tag if user didn't explicitly change it
    if isinstance(instruction_obj, dict) and "action_set_tag" in instruction_obj:
        try:
            # Heuristic: only override if args has default value
            if getattr(args, "action_set_tag", "id_accessibility_tree") == "id_accessibility_tree":
                args.action_set_tag = instruction_obj["action_set_tag"]
        except Exception:
            pass
    # Ensure we have a default action_set_tag even if instruction JSON is invalid or missing
    if not hasattr(args, "action_set_tag") or args.action_set_tag is None:
        try:
            with open(args.instruction_path) as _f:
                _raw = json.load(_f)
                args.action_set_tag = _raw.get("action_set_tag", "id_accessibility_tree")
        except Exception:
            args.action_set_tag = "id_accessibility_tree"
    
    # Reward model config will be constructed later for reward_guided agent using instruction JSON

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]

        # Build policy llm_config for prompt agent using legacy schema or CLI args
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
            gen_overrides = policy_section.get("gen") or {}
            if isinstance(gen_overrides, dict):
                for k, v in gen_overrides.items():
                    if v is not None:
                        setattr(policy_args, k, v)
            _policy_llm_config = lm_config.construct_llm_config(policy_args)  # type: ignore[arg-type]
        else:
            _policy_llm_config = lm_config.construct_llm_config(args)

        # Build tokenizer from the policy llm_config
        tokenizer = Tokenizer(_policy_llm_config.provider, _policy_llm_config.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=_policy_llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=_policy_llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
            planner_ip=args.planner_ip
        )
    elif args.agent_type == "reward_guided":
        # Load instruction data to get configuration
        with open(args.instruction_path) as f:
            instruction_data = json.load(f)

        # Build policy model config using new policy_lm_config if provided; else fall back to legacy or args
        policy_cfg = instruction_data.get("policy_lm_config") if isinstance(instruction_data, dict) else None
        if isinstance(policy_cfg, dict):
            policy_provider = policy_cfg.get("provider") or args.provider
            policy_model = policy_cfg.get("model") or args.model
            policy_mode = policy_cfg.get("mode") or args.mode
            policy_model_endpoint = policy_cfg.get("model_endpoint", getattr(args, "model_endpoint", None))

            gen_overrides = policy_cfg.get("gen") or policy_cfg.get("gen_config") or {}
            temperature = gen_overrides.get("temperature") if gen_overrides.get("temperature") is not None else args.temperature
            top_p_val = gen_overrides.get("top_p") if gen_overrides.get("top_p") is not None else args.top_p
            max_tokens_val = gen_overrides.get("max_tokens") if gen_overrides.get("max_tokens") is not None else args.max_tokens

            policy_args = SimpleNamespace(
                provider=policy_provider,
                model=policy_model,
                mode=policy_mode,
                temperature=temperature,
                top_p=top_p_val,
                context_length=args.context_length,
                max_tokens=max_tokens_val,
                stop_token=args.stop_token,
                max_obs_length=args.max_obs_length,
                max_retry=args.max_retry,
                model_endpoint=policy_model_endpoint,
            )
            policy_llm_config = lm_config.construct_llm_config(policy_args)  # type: ignore[arg-type]
        elif isinstance(instruction_obj, dict) and isinstance(instruction_obj.get("policy_model"), dict):
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
            gen_overrides = policy_section.get("gen") or {}
            if isinstance(gen_overrides, dict):
                for k, v in gen_overrides.items():
                    if v is not None:
                        setattr(policy_args, k, v)
            policy_llm_config = lm_config.construct_llm_config(policy_args)  # type: ignore[arg-type]
        else:
            policy_llm_config = lm_config.construct_llm_config(args)

        # Build reward model config using instruction JSON if provided; else fall back to args
        reward_cfg = instruction_data.get("reward_lm_config") if isinstance(instruction_data, dict) else None
        if isinstance(reward_cfg, dict):
            reward_provider = reward_cfg.get("provider") or getattr(args, "reward_provider", None) or args.provider
            reward_model = reward_cfg.get("model") or getattr(args, "reward_model", None) or args.model
            reward_mode = reward_cfg.get("mode") or getattr(args, "mode", None) or args.mode
            reward_model_endpoint = reward_cfg.get("model_endpoint", getattr(args, "reward_model_endpoint", None) or getattr(args, "model_endpoint", None))

            # Defaults: temperature 0.0, max_tokens 100 if not specified
            gen_overrides = reward_cfg.get("gen") or reward_cfg.get("gen_config") or {}
            temperature = gen_overrides.get("temperature") if gen_overrides.get("temperature") is not None else 0.0
            top_p_val = gen_overrides.get("top_p") if gen_overrides.get("top_p") is not None else args.top_p
            max_tokens_val = gen_overrides.get("max_tokens") if gen_overrides.get("max_tokens") is not None else 100

            reward_args = SimpleNamespace(
                provider=reward_provider,
                model=reward_model,
                mode=reward_mode,
                temperature=temperature,
                top_p=top_p_val,
                context_length=args.context_length,
                max_tokens=max_tokens_val,
                stop_token=args.stop_token,
                max_obs_length=args.max_obs_length,
                max_retry=args.max_retry,
                model_endpoint=reward_model_endpoint,
            )
        else:
            reward_provider = getattr(args, "reward_provider", None) or args.provider
            reward_model = getattr(args, "reward_model", None) or args.model
            reward_mode = getattr(args, "mode", None) or args.mode
            reward_model_endpoint = getattr(args, "reward_model_endpoint", None) or getattr(args, "model_endpoint", None)

            reward_args = SimpleNamespace(
                provider=reward_provider,
                model=reward_model,
                mode=reward_mode,
                temperature=0.0 if getattr(args, "temperature", None) is None else args.temperature,
                top_p=args.top_p,
                context_length=args.context_length,
                max_tokens=100 if getattr(args, "max_tokens", None) is None else args.max_tokens,
                stop_token=args.stop_token,
                max_obs_length=args.max_obs_length,
                max_retry=args.max_retry,
                model_endpoint=reward_model_endpoint,
            )
        reward_llm_config = lm_config.construct_llm_config(reward_args)  # type: ignore[arg-type]

        # Read parameters from instruction JSON file
        eff_num_samples = instruction_data.get("num_samples", 20)
        eff_temperature = instruction_data.get("temperature", 1.0)
        eff_top_p = instruction_data.get("top_p", 0.9)

        # Import RewardGuidedAgent (self-contained prompt loading)
        from .reward_guided_agent import RewardGuidedAgent

        agent = RewardGuidedAgent(
            action_set_tag=args.action_set_tag,
            policy_lm_config=policy_llm_config,
            reward_lm_config=reward_llm_config,
            captioning_fn=captioning_fn,
            num_samples=eff_num_samples,
            temperature=eff_temperature,
            top_p=eff_top_p,
            use_batch_reward=getattr(args, "use_batch_reward", False),
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent
