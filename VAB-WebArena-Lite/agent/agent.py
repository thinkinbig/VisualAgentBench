import argparse
import json
import copy
from typing import Any, Optional, List, Dict, Tuple

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_webrl_id_based_action
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
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
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
            captioning_fn=captioning_fn,
            planner_ip=args.planner_ip
        )
    elif args.agent_type == "reward_guided":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = RewardGuidedAgent(
            action_set_tag=args.action_set_tag,
            policy_lm_config=llm_config,
            reward_lm_config=llm_config,  # Can be configured separately
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
            planner_ip=args.planner_ip
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
        
        # Check if the model is multimodal
        if ("gemini" in policy_lm_config.model or 
            "gpt-4" in policy_lm_config.model and "vision" in policy_lm_config.model or 
            policy_lm_config.provider in ["api", "finetune"]) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False
    
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
        for _ in range(self.num_samples):
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
                
                # Parse action
                parsed_response = self.prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == 'webrl_id':
                    action = create_webrl_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                
                action["raw_prediction"] = response
                candidates.append((response, action))
                
            except Exception as e:
                print(f"Error generating sample {_}: {e}")
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
            
            # Get reward score from reward model
            response = call_llm(self.reward_lm_config, reward_prompt)
            
            # Extract reward score from response
            # Assuming the reward model returns a score in the format "Score: X.X" or similar
            try:
                # Try to extract numerical score
                import re
                score_match = re.search(r'Score:\s*([\d.]+)', response)
                if score_match:
                    return float(score_match.group(1))
                
                # Fallback: try to find any number in the response
                numbers = re.findall(r'[\d.]+', response)
                if numbers:
                    return float(numbers[0])
                
                # If no score found, return a default score
                return 0.0
                
            except (ValueError, IndexError):
                return 0.0
                
        except Exception as e:
            print(f"Error scoring action with reward model: {e}")
            return 0.0
    
    def _create_reward_prompt(
        self, 
        action: Action, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any]
    ) -> str:
        """Create a prompt for the reward model to evaluate an action"""
        # This is a simplified reward prompt - you may want to customize this
        prompt = f"""
        You are a reward model that evaluates web agent actions. Given the current state and an action, 
        predict how good this action is for achieving the goal.
        
        Goal: {intent}
        
        Current State: {trajectory[-1]["observation"]["text"] if "text" in trajectory[-1]["observation"] else "No text observation"}
        
        Proposed Action: {action.get("raw_prediction", str(action))}
        
        Please evaluate this action and provide a score from 0.0 to 10.0, where:
        - 0.0: Completely wrong action that will not help achieve the goal
        - 5.0: Neutral action that neither helps nor hurts
        - 10.0: Perfect action that directly helps achieve the goal
        
        Respond with: Score: X.X
        """
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
            # Create refinement prompt
            refinement_prompt = f"""
            You are a web agent that needs to refine an action based on feedback.
            
            Goal: {intent}
            
            Current State: {trajectory[-1]["observation"]["text"] if "text" in trajectory[-1]["observation"] else "No text observation"}
            
            Original Action: {action.get("raw_prediction", str(action))}
            
            Feedback: {refinement_prompt}
            
            Please provide a refined action that addresses the feedback and better achieves the goal.
            """
            
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
            prompt += f"\n\nRefinement Context:\n{refinement_prompt}"
            
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
        for action_key, action_list in sorted_candidates[:5]:  # Top 5 most frequent
            # Use the first occurrence of each unique action
            top_candidates.append(action_list[0])
        
        # Step 3: Score candidates using reward model
        scored_candidates = []
        for response, action in top_candidates:
            score = self._score_action_with_reward_model(action, trajectory, intent, meta_data)
            scored_candidates.append((score, response, action))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Step 4: Select best action
        best_score, best_response, best_action = scored_candidates[0]
        
        if output_response:
            print(f'Agent: {best_response}', flush=True)
            print(f'Reward Score: {best_score}', flush=True)
        
        # Step 5: Refinement process (up to max_refinements times)
        current_action = best_action
        current_score = best_score
        refinement_count = 0
        
        while refinement_count < self.max_refinements:
            # Get feedback from reward model (excluding the actual score)
            feedback_prompt = self._create_reward_prompt(current_action, trajectory, intent, meta_data)
            feedback_response = call_llm(self.reward_lm_config, feedback_prompt)
            
            # Remove score information to get pure feedback
            feedback = feedback_response.replace(f"Score: {current_score}", "").strip()
            
            # Refine the action
            refined_response, refined_action = self._refine_action(
                current_action, trajectory, intent, meta_data, feedback
            )
            
            # Score the refined action
            refined_score = self._score_action_with_reward_model(
                refined_action, trajectory, intent, meta_data
            )
            
            if output_response:
                print(f'Refinement {refinement_count + 1}: {refined_response}', flush=True)
                print(f'Refined Score: {refined_score}', flush=True)
            
            # Only accept refinement if it improves the score
            if refined_score > current_score:
                current_action = refined_action
                current_score = refined_score
                refinement_count += 1
                
                if output_response:
                    print(f'Refinement accepted! New score: {refined_score}', flush=True)
            else:
                if output_response:
                    print(f'Refinement rejected. Score did not improve.', flush=True)
                break
        
        return current_action
    
    def reset(self, test_config_file: str) -> None:
        pass
