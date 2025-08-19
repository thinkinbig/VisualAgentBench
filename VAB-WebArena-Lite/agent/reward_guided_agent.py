import logging
import re
from typing import Any, Optional, List, Dict, Tuple
from PIL import Image

from beartype import beartype
from llms.utils import build_api_input_for_text, call_llm

from .agent import Agent
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_webrl_id_based_action,
)
from llms import lm_config


class RewardGuidedAgent(Agent):
    """Reward-guided Trajectory Search Agent using enhanced prompts"""
    
    def __init__(
        self,
        action_set_tag: str,
        policy_lm_config: lm_config.LMConfig,
        reward_lm_config: lm_config.LMConfig,
        captioning_fn=None,
        num_samples: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
        use_batch_reward: bool = False,
    ) -> None:
        super().__init__()
        self.action_set_tag = action_set_tag
        self.policy_lm_config = policy_lm_config
        self.reward_lm_config = reward_lm_config
        self.captioning_fn = captioning_fn
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.use_batch_reward = use_batch_reward

        self.logger = logging.getLogger("reward_guided_logger")
        
        # Initialize discovery context for storing thoughts from send_msg actions
        self.discovery_context = []
        
        # Heuristic: determine if policy model expects multimodal inputs
        model_lower = (policy_lm_config.model or "").lower()
        provider_lower = (policy_lm_config.provider or "").lower()
        self.multimodal_inputs = (
            ("gemini" in model_lower)
            or ("gpt-4" in model_lower and "vision" in model_lower)
            or (provider_lower in ["api", "finetune"])
        )
        
        # Load enhanced prompt for policy LM
        self._load_enhanced_prompt()
        
        # Load base prompt for reward LM
        self._load_reward_prompt()
        
        # Load batch reward prompt for batch scoring
        self._load_batch_reward_prompt()
    
    def _load_enhanced_prompt(self) -> None:
        """Load the enhanced prompt for policy LM from raw Python file"""
        try:
            from agent.prompts.raw.p_cot_id_actree_3s_enhanced import prompt
            self.enhanced_prompt = prompt
            self.logger.info("Loaded enhanced prompt for policy LM from raw file")
        except ImportError as e:
            self.logger.warning(f"Failed to import enhanced prompt from raw file: {e}")
            self.enhanced_prompt = None
    
    def _load_reward_prompt(self) -> None:
        """Load the reward evaluation prompt from raw Python file"""
        try:
            from agent.prompts.raw.reward_evaluation_prompt import reward_evaluation_prompt
            self.reward_prompt = reward_evaluation_prompt
            self.logger.info("Loaded reward evaluation prompt from raw file")
        except ImportError as e:
            self.logger.warning(f"Failed to import reward evaluation prompt from raw file: {e}")
            self.reward_prompt = None
    
    def _load_batch_reward_prompt(self) -> None:
        """Load the batch reward evaluation prompt from raw Python file"""
        try:
            from agent.prompts.raw.batch_reward_evaluation_prompt import batch_reward_evaluation_prompt
            self.batch_reward_prompt = batch_reward_evaluation_prompt
            self.logger.info("Loaded batch reward evaluation prompt from raw file")
        except ImportError as e:
            self.logger.warning(f"Failed to import batch reward evaluation prompt from raw file: {e}")
            self.batch_reward_prompt = None
    
    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag
    
    def _extract_action_from_backticks(self, response: str) -> str:
        """Extract action between triple backticks; fallback to best-effort cleanup."""
        try:
            pattern = r"```([\s\S]*?)```"
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
            # Best-effort: look for answer phrase then take remainder
            answer_phrase = "In summary, the next action I will perform is"
            if answer_phrase in response:
                tail = response.split(answer_phrase, 1)[1].strip()
                # Strip any trailing non-action chars
                tail = re.sub(r"[\s\S]*?```", "", tail) or tail
                tail = tail.strip("` ")
                return tail.strip()
            return response.strip()
        except Exception:
            return response.strip()
    
    def _extract_thoughts_from_response(self, response: str) -> Dict:
        """Extract thoughts and action from the enhanced prompt response format (single-step)."""
        try:
            if "In summary, the next action I will perform is" in response:
                summary_start = response.find("In summary, the next action I will perform is")
                reasoning_part = response[:summary_start].strip()
                action = self._extract_action_from_backticks(response)
                return {
                    "thought": reasoning_part,
                    "action": action,
                    "raw_response": response
                }
            else:
                extracted_action = self._extract_action_from_backticks(response)
                return {
                    "thought": "",
                    "action": extracted_action,
                    "raw_response": response
                }
        except Exception as e:
            self.logger.warning("Failed to parse response: %s", e)
            try:
                extracted_action = self._extract_action_from_backticks(response)
                return {
                    "thought": "",
                    "action": extracted_action,
                    "raw_response": response
                }
            except Exception:
                return {
                    "thought": "",
                    "action": "",
                    "raw_response": response
                }
    
    def _update_discovery_context(self, action: Action, thoughts: Dict) -> None:
        """Update discovery context with thoughts from send_msg actions"""
        if not thoughts or not thoughts.get("action"):
            return
            
        action_str = thoughts.get("action")
        
        # Check if this is a send_msg action
        if action_str.startswith("send_msg(") or "send_msg" in action_str:
            # Extract the message content - handle both single and double quotes
            message_match = re.search(r'send_msg\(["\']([^"\']+)["\']\)', action_str)
            if message_match:
                message_content = message_match.group(1)
                
                # Create discovery context entry
                self.discovery_context.append({
                    "timestamp": len(self.discovery_context),
                    "message": message_content,
                    "thought": thoughts.get("thought")
                })
                self.logger.info("Added to discovery context: %s", message_content)
    
    def _get_discovery_context_string(self) -> str:
        """Convert discovery context to a string for prompt inclusion"""
        if not self.discovery_context:
            return ""
        
        context_parts = []
        for entry in self.discovery_context[-5:]:  # Keep last 5 entries
            context_parts.append(f"Discovery {entry['timestamp'] + 1}: {entry['message']}")
            if entry.get("thought"):
                context_parts.append(f"  Reasoning: {entry['thought'][:200]}...")  # Limit length
        
        return "\n".join(context_parts)
    
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
        
        # Add discovery context to intent if available
        discovery_context_str = self._get_discovery_context_string()
        if discovery_context_str:
            intent_with_context = f"{intent}\n\nPrevious Discoveries:\n{discovery_context_str}"
        else:
            intent_with_context = intent
        
        # Generate multiple samples with nucleus sampling
        unique_action_keys: set[str] = set()
        action_key_counts: Dict[str, int] = {}

        for sample_idx in range(self.num_samples):
            try:
                # Use enhanced prompt if available, otherwise fall back to original
                if self.enhanced_prompt and "intro" in self.enhanced_prompt:
                    # Create enhanced prompt manually
                    enhanced_intro = self.enhanced_prompt["intro"]
                    enhanced_examples = self.enhanced_prompt.get("examples", [])
                    enhanced_template = self.enhanced_prompt.get("template", "")
                    
                    # Build histories
                    # Action history from meta_data
                    action_hist_list = []
                    try:
                        if isinstance(meta_data, dict):
                            action_hist_list = list(meta_data.get("action_history", []))
                    except Exception:
                        action_hist_list = []
                    # Drop the initial placeholder if present
                    if action_hist_list and action_hist_list[0] == "None":
                        action_hist_list = action_hist_list[1:]
                    action_hist_str = "\n".join([f"- {a}" for a in action_hist_list[-10:]]) if action_hist_list else "(empty)"
                    
                    # Thought history derived from past actions in trajectory
                    thought_hist_entries: list[str] = []
                    try:
                        for item in trajectory:
                            if isinstance(item, dict) and item.get("action_type") is not None:
                                t = item.get("thoughts", {}).get("thought")
                                if t:
                                    thought_hist_entries.append(t)
                    except Exception:
                        pass
                    thought_hist_str = "\n".join([f"- {t}" for t in thought_hist_entries[-10:]]) if thought_hist_entries else "(empty)"
                    
                    # Get current observation and URL
                    current_obs = trajectory[-1]["observation"].get("text", "No observation")
                    current_url = trajectory[-1]["info"].get("page", {}).url if trajectory[-1]["info"].get("page") else "No URL"
                    previous_action = meta_data.get("action_history", ["None"])[-1] if meta_data.get("action_history") else "None"
                    
                    # Format the template
                    current_input = enhanced_template.format(
                        observation=current_obs,
                        url=current_url,
                        objective=intent_with_context,
                        previous_action=previous_action,
                        action_history=action_hist_str,
                        thought_history=thought_hist_str
                    )
                    
                    # System prompt: intro + few-shot examples
                    system_text = enhanced_intro
                    if enhanced_examples:
                        examples_block = []
                        for i, (input_ex, output_ex) in enumerate(enhanced_examples):
                            examples_block.append(f"Example {i+1}:\nInput: {input_ex}\nOutput: {output_ex}")
                        system_text = system_text + "\n\nExamples:\n" + "\n\n".join(examples_block)

                    # User prompt: ONLY current filled template
                    user_text = f"{current_input}\n\nPlease think step-by-step and then provide your action."

                    # Build API input
                    api_input = build_api_input_for_text(
                        self.policy_lm_config,
                        system_text,
                        user_text
                    )
                    # Log prompts only once per step (first sample) for inspection
                    if sample_idx == 0:
                        try:
                            self.logger.debug("=== POLICY SYSTEM PROMPT ===\n%s", system_text)
                            self.logger.debug("=== POLICY USER PROMPT ===\n%s", user_text)
                        except Exception:
                            pass
                    response = call_llm(self.policy_lm_config, api_input)
            
                self.logger.debug("[Sample %d] Raw LLM response: %r", sample_idx, response)
            
                # Extract thoughts and action from the new format
                thoughts = self._extract_thoughts_from_response(response)
                parsed_response = thoughts.get("action")
                
                if not parsed_response:
                    self.logger.warning("[Sample %d] No action extracted from response", sample_idx)
                    continue
                
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
                action["thoughts"] = thoughts  # Store thoughts dict for later use
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
            
            # Log the complete reward prompt for debugging
            self.logger.debug("=== REWARD PROMPT ===\n%s", reward_prompt)
            
            # Get reward score from reward model (build provider-specific API input)
            api_input = build_api_input_for_text(
                self.reward_lm_config,
                "You are a reward model for web agents. Read the user's content and output only: Score: X.X",
                reward_prompt,
            )
            response = call_llm(self.reward_lm_config, api_input)
            self.logger.debug("Reward model response: %r", response)
            
            # Log the complete reward model interaction
            self.logger.debug("=== REWARD MODEL RESPONSE ===\n%s", response)
            
            # Extract reward score from response
            try:
                # Try to extract numerical score using SCORE format
                score_match = re.search(r'SCORE:\s*(\d+)', response)
                if score_match:
                    score = int(score_match.group(1))
                    # Ensure score is within valid range (1 to 5)
                    score = max(1, min(5, score))
                    self.logger.debug("Extracted reward score: %s", score)
                    return float(score)
                
                # Fallback: try to find any number in the response
                numbers = re.findall(r'\d+', response)
                if numbers:
                    score = int(numbers[0])
                    # Ensure score is within valid range (1 to 5)
                    score = max(1, min(5, score))
                    self.logger.debug("Fallback extracted numeric score: %s", score)
                    return float(score)
                
                # If no score found, return a default score
                self.logger.debug("No numeric score found, defaulting to 1.0")
                return 1.0
                
            except (ValueError, IndexError):
                return 1.0
                
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
            current_url = self._get_current_url(trajectory)
            proposals: List[str] = []
            for idx, act in enumerate(actions, start=1):
                raw_pred = act.get("raw_prediction", str(act))
                # Handle enhanced prompt format: "Let's think step-by-step... In summary, the next action I will perform is ```action```"
                if isinstance(raw_pred, str) and "In summary, the next action I will perform is" in raw_pred:
                    try:
                        # Extract the action part (inside ``` ```) - focus on action for scoring
                        action_part = self._extract_action_from_backticks(raw_pred)
                        proposals.append(f"{idx}. {action_part}")
                    except Exception:
                        proposals.append(f"{idx}. {raw_pred}")
                else:
                    # Handle old format
                    if isinstance(raw_pred, str) and "```" in raw_pred:
                        try:
                            raw_pred = self._extract_action_from_backticks(raw_pred)
                        except Exception:
                            raw_pred = raw_pred.replace("```", "").strip()
                    proposals.append(f"{idx}. {raw_pred}")

            # Use batch reward prompt template if available
            if self.batch_reward_prompt:
                # Format the batch reward prompt with the proposals (proposals already have numbers)
                formatted_multiple_actions = "\n".join(proposals)
                
                # Get the first action's thoughts for the agent response section
                first_action = actions[0] if actions else None
                first_thoughts = first_action.get("thoughts", {}) if first_action else {}
                agent_thought = first_thoughts.get("thought", "No thought provided")
                agent_action = first_action.get("raw_prediction", str(first_action)) if first_action else "No action"
                
                complete_batch_prompt = self.batch_reward_prompt.format(
                    intent=intent,
                    trajectory=self._format_trajectory_for_prompt(trajectory),
                    current_url=current_url,
                    text_observation=current_text,
                    multiple_actions=formatted_multiple_actions,
                    thought=agent_thought,
                    action=agent_action
                )
                
                # Split into system and user parts
                if "# Action space:" in complete_batch_prompt:
                    system_text = complete_batch_prompt.split("# Action space:")[0].strip()
                    user_text = complete_batch_prompt.split("# Action space:")[1].strip()
                else:
                    # Fallback if template doesn't have expected structure
                    system_text = self.batch_reward_prompt.split("# Action space:")[0].strip()
                    user_text = f"Goal: {intent}\n\nCurrent State: {current_text}\nCurrent URL: {current_url}\n\nProposed Actions:\n" + "\n".join(proposals) + "\n\nFor each action i, reply on a separate line strictly as:\ni: REASON: [explanation]\nSCORE: [1-5]"
            else:
                # Fallback to manual construction if batch prompt not available
                system_text = self.reward_prompt.split("# Action space:")[0].strip()
                user_text = f"Goal: {intent}\n\nCurrent State: {current_text}\nCurrent URL: {current_url}\n\nProposed Actions:\n" + "\n".join(proposals) + "\n\nFor each action i, reply on a separate line strictly as:\ni: REASON: [explanation]\nSCORE: [1-5]"
            
            # Log the complete batch reward prompt for debugging
            complete_batch_prompt = system_text + "\n\n" + user_text
            self.logger.debug("=== COMPLETE BATCH REWARD PROMPT ===\n%s", complete_batch_prompt)
            self.logger.debug("=== BATCH REWARD SYSTEM PROMPT ===\n%s", system_text)
            self.logger.debug("=== BATCH REWARD USER PROMPT ===\n%s", user_text)
            
            api_input = build_api_input_for_text(self.reward_lm_config, system_text, user_text)
            response = call_llm(self.reward_lm_config, api_input)
            self.logger.debug("Batch reward model response: %r", response)
            
            # Log the complete batch reward model response
            self.logger.debug("=== BATCH REWARD MODEL RESPONSE ===\n%s", response)

            pattern = re.compile(r"^(\d+)\s*:\s*REASON:\s*.*?\nSCORE:\s*(\d+)", re.MULTILINE | re.DOTALL)
            matches = pattern.findall(response)
            scores: List[float] = [1.0] * len(actions)  # Default to 1.0 instead of 0.0
            for idx_str, score_str in matches:
                try:
                    i = int(idx_str)
                    if 1 <= i <= len(actions):
                        score = int(score_str)
                        # Ensure score is within valid range (1 to 5)
                        score = max(1, min(5, score))
                        scores[i - 1] = float(score)
                except Exception:
                    continue

            # If parsing failed for all, fallback to per-action scoring
            if all(s == 1.0 for s in scores):
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

    def _format_trajectory_for_prompt(self, trajectory: Trajectory) -> str:
        """Format trajectory for inclusion in reward prompt"""
        try:
            if not trajectory:
                return "No trajectory available"
            
            # Format recent trajectory steps (last 5 for brevity)
            formatted_steps = []
            for i, step in enumerate(trajectory[-5:], 1):
                if isinstance(step, dict):
                    action_info = step.get("action", {})
                    action_str = str(action_info) if action_info else "No action"
                    
                    # Get observation info
                    obs = step.get("observation", {})
                    obs_text = obs.get("text", "No observation")[:200]  # Limit length
                    
                    formatted_steps.append(f"Step {i}: Action: {action_str}\n  Observation: {obs_text}")
            
            return "\n".join(formatted_steps) if formatted_steps else "No trajectory steps available"
        except Exception as e:
            self.logger.warning("Error formatting trajectory: %s", e)
            return "Error formatting trajectory"
    
    def _get_current_url(self, trajectory: Trajectory) -> str:
        """Extract current URL from trajectory"""
        try:
            if trajectory and len(trajectory) > 0:
                last_step = trajectory[-1]
                if isinstance(last_step, dict):
                    page_info = last_step.get("info", {}).get("page", {})
                    if hasattr(page_info, 'url'):
                        return page_info.url
                    elif isinstance(page_info, dict):
                        return page_info.get("url", "No URL available")
            return "No URL available"
        except Exception as e:
            self.logger.warning("Error extracting URL: %s", e)
            return "Error extracting URL"
    
    def _create_reward_prompt(
        self, 
        action: Action, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any]
    ) -> str:
        """Create a prompt for the reward model to evaluate an action with thoughts and reasoning"""
        # Extract thoughts and action from the response
        raw_pred = action.get("raw_prediction", str(action))
        proposed_action = raw_pred
        thought = ""
        
        # Parse the enhanced prompt format: "Let's think step-by-step... In summary, the next action I will perform is ```action```"
        if isinstance(raw_pred, str) and "In summary, the next action I will perform is" in raw_pred:
            try:
                # Extract the reasoning part (everything before "In summary")
                summary_start = raw_pred.find("In summary, the next action I will perform is")
                reasoning_part = raw_pred[:summary_start].strip()
                
                # Extract the action part (inside ``` ```)
                proposed_action = self._extract_action_from_backticks(raw_pred)
                thought = reasoning_part
            except Exception:
                # Fallback to old format
                if "```" in raw_pred:
                    proposed_action = self._extract_action_from_backticks(raw_pred)
                else:
                    proposed_action = raw_pred
        else:
            # Fallback to old format
            if "```" in raw_pred:
                try:
                    proposed_action = self._extract_action_from_backticks(raw_pred)
                except Exception:
                    proposed_action = raw_pred.replace("```", "").strip()
            else:
                proposed_action = raw_pred

        current_text = trajectory[-1]["observation"].get("text", "No text observation")
        
        # Use loaded reward prompt template
        if self.reward_prompt:
            # Use the loaded reward evaluation prompt template
            return self.reward_prompt.format(
                intent=intent,
                trajectory=self._format_trajectory_for_prompt(trajectory),
                current_url=self._get_current_url(trajectory),
                text_observation=current_text,
                thought=thought,
                action=proposed_action
            )
        else:
            # Fallback to default reward prompt
            raise ValueError(
                "Failed to load reward prompt. Please ensure reward_evaluation_prompt.py exists and is valid."
            )
    
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
        
        # Step 3: Score candidates using reward model
        scored_candidates = []
        try:
            if self.use_batch_reward:
                self.logger.debug("Using batch reward scoring for %d candidates", len(top_candidates))
                actions_for_batch = [a for _, a in top_candidates]
                scores = self._score_actions_with_reward_model(actions_for_batch, trajectory, intent, meta_data)
                for (response, action), score in zip(top_candidates, scores):
                    scored_candidates.append((score, response, action))
                    self.logger.info("Candidate scored (batch): score=%.3f, action=%s", score, str(action))
            else:
                self.logger.debug("Using per-action reward scoring for %d candidates", len(top_candidates))
                for response, action in top_candidates:
                    score = self._score_action_with_reward_model(action, trajectory, intent, meta_data)
                    scored_candidates.append((score, response, action))
                    self.logger.info("Candidate scored (single): score=%.3f, action=%s", score, str(action))
        except Exception:
            # Fallback: score one by one
            for response, action in top_candidates:
                score = self._score_action_with_reward_model(action, trajectory, intent, meta_data)
                scored_candidates.append((score, response, action))
                self.logger.info("Candidate scored (fallback single): score=%.3f, action=%s", score, str(action))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Step 4: Select best action
        best_score, best_response, best_action = scored_candidates[0]
        self.logger.info("Best action selected: score=%.3f, action=%s", best_score, str(best_action))
        
        # Step 5: Update discovery context if this is a send_msg action
        best_thoughts = best_action.get("thoughts")  # Get the thoughts dict
        if best_thoughts:
            self._update_discovery_context(best_action, best_thoughts)
        
        if output_response:
            print(f'Agent: {best_response}', flush=True)
            print(f'Reward Score: {best_score}', flush=True)
        
        return best_action
    
    def reset(self, test_config_file: str) -> None:
        """Reset the agent state, including discovery context and reload prompts"""
        self.discovery_context = []
        
        # Reload prompts to ensure they're up to date
        self._load_enhanced_prompt()
        self._load_reward_prompt()
        self._load_batch_reward_prompt()
        
        self.logger.info("Agent reset: cleared discovery context and reloaded prompts")
    
    def get_discovery_context(self) -> List[Dict]:
        """Get the current discovery context for external use"""
        return self.discovery_context.copy()
