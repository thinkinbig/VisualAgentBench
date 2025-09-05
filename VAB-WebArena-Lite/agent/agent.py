import argparse
import json
from typing import Any

import tiktoken
from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
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
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ) -> Action:
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
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


def construct_agent(args: argparse.Namespace) -> Agent:
    agent: Agent
    if args.agent_type == "reward_guided":
        # Defer import to avoid heavy deps at module import time
        from agent.reward_guided_agent import RewardGuidedAgent
        # Load configuration from file
        with open(args.instruction_path, "r") as f:
            cfg = json.load(f)
        
        # Build LM configs from JSON
        def build_lm(cfg_obj: dict) -> lm_config.LMConfig:
            return lm_config.LMConfig(
                provider=cfg_obj.get("provider"),
                model=cfg_obj.get("model"),
                mode=cfg_obj.get("mode"),
                gen_config=cfg_obj.get("gen_config", {}),
            )
        
        # Extract all configuration parameters
        policy_lm = build_lm(cfg.get("policy_lm_config", {}))
        reward_lm = build_lm(cfg.get("reward_lm_config", {}))
        action_set_tag = cfg.get("action_set_tag", "id_accessibility_tree")
        num_samples = int(cfg.get("num_samples", 16))
        max_steps = getattr(args, 'max_steps', None) or cfg.get('max_steps', 30)
        num_calls = cfg.get('num_calls', 5)
        max_tournament_candidates = cfg.get('max_tournament_candidates', 16)
        max_obs_length = cfg.get('max_obs_length', 2048)
        max_retry = cfg.get('max_retry', 3)
        
        # Create NucleusSampler from config
        from agent.sampling import NucleusSampler
        nucleus_sampling_config = cfg.get("nucleus_sampling_config", None)
        nucleus_sampler = NucleusSampler(nucleus_sampling_config)
        
        agent = RewardGuidedAgent(
            action_set_tag=action_set_tag,
            policy_lm_config=policy_lm,
            reward_lm_config=reward_lm,
            num_samples=num_samples,
            nucleus_sampler=nucleus_sampler,
            max_steps=max_steps,
            num_calls=num_calls,
            max_tournament_candidates=max_tournament_candidates,
            max_obs_length=max_obs_length,
            max_retry=max_retry,
        )
    else:
        # For other agent types, build LLM config first
        llm_config = lm_config.construct_llm_config(args)
        
        if args.agent_type == "teacher_forcing":
            agent = TeacherForcingAgent()
        elif args.agent_type == "prompt":
            with open(args.instruction_path) as f:
                constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
            tokenizer = Tokenizer(args.provider, args.model)
            prompt_constructor = eval(constructor_type)(
                args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
            )
            agent = PromptAgent(
                action_set_tag=args.action_set_tag,
                lm_config=llm_config,
                prompt_constructor=prompt_constructor,
            )
        else:
            raise NotImplementedError(
                f"agent type {args.agent_type} not implemented"
            )
    return agent