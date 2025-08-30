import logging
import re
import json
import os
from typing import Any, Optional, List, Dict, Tuple

from beartype import beartype
from llms.utils import build_api_input_for_text, call_llm

from .agent import Agent
from browser_env import Trajectory, ActionTypes
from browser_env.actions import (
    Action,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_webrl_id_based_action,
)
from browser_env.actions import is_equivalent
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
    ) -> None:
        super().__init__()
        self.action_set_tag = action_set_tag
        self.policy_lm_config = policy_lm_config
        self.reward_lm_config = reward_lm_config
        self.captioning_fn = captioning_fn
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p

        self.logger = logging.getLogger("reward_guided_logger")
        self.reward_score_max_retries = 3
        
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
    
    def _load_enhanced_prompt(self) -> None:
        """Load the enhanced prompt for policy LM from JSON file"""
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "prompts", "jsons", "enhanced_actree.json")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                self.enhanced_prompt = json.load(f)
            self.logger.info("Loaded enhanced prompt for policy LM from JSON file: %s", json_path)
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Failed to load enhanced prompt from JSON file: {e}")
            self.enhanced_prompt = None
    
    def _load_reward_prompt(self) -> None:
        """Load the reward evaluation prompt from JSON file"""
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "prompts", "jsons", "reward_evaluation_prompt.json")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                self.reward_prompt = json.load(f)
            self.logger.info("Loaded reward evaluation prompt from JSON file: %s", json_path)
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Failed to load reward evaluation prompt from JSON file: {e}")
            self.reward_prompt = None
    
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
            # Preferred format: explicit THOUGHT and ACTION sections
            thought_action_match = re.search(r"THOUGHT:\s*(.*?)\s*ACTION:\s*```([\s\S]*?)```", response, re.IGNORECASE | re.DOTALL)
            if thought_action_match:
                reasoning_part = thought_action_match.group(1).strip()
                action = thought_action_match.group(2).strip()
                return {
                    "thought": reasoning_part,
                    "action": action,
                    "raw_response": response
                }

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

    def _concise_action_string(self, action: Action) -> str:
        """Return a concise string like 'click [1234]' or 'type [5678]' for logging."""
        try:
            action_type = action.get("action_type")
            element_id = action.get("element_id", "")
            if action_type is None:
                return str(action)
            if isinstance(action_type, ActionTypes):
                action_name = str(action_type).split(".")[-1].lower()
            else:
                action_name = str(action_type).lower()
            if element_id:
                return f"{action_name} [{element_id}]"
            key_comb = action.get("key_comb", "")
            direction = action.get("direction", "")
            answer = action.get("answer", "")
            if action_name == "press" and key_comb:
                return f"press [{key_comb}]"
            if action_name == "scroll" and direction:
                return f"scroll [{direction}]"
            if action_name == "stop" and answer:
                return f"stop [{answer}]"
            return action_name
        except Exception:
            return str(action)

    def _is_valid_action(self, action: Action) -> bool:
        """Lightweight validity checks to filter unusable actions before scoring/dedup."""
        try:
            action_type = action.get("action_type")
            if action_type is None:
                return False

            # Always valid or no-op
            if action_type in (
                ActionTypes.NONE,
                ActionTypes.NEW_TAB,
                ActionTypes.GO_BACK,
                ActionTypes.GO_FORWARD,
                ActionTypes.PAGE_CLOSE,
            ):
                return True

            if action_type == ActionTypes.SCROLL:
                direction = (action.get("direction") or "").lower()
                return direction in {"up", "down"}

            if action_type == ActionTypes.KEY_PRESS:
                key_comb = action.get("key_comb") or ""
                return len(key_comb.strip()) > 0

            if action_type == ActionTypes.MOUSE_CLICK:
                coords = action.get("coords")
                try:
                    return coords is not None and len(coords) == 2
                except Exception:
                    return False

            if action_type == ActionTypes.KEYBOARD_TYPE:
                text = action.get("text")
                return bool(text)

            if action_type in (ActionTypes.CLICK, ActionTypes.HOVER, ActionTypes.TYPE):
                has_id = bool(action.get("element_id"))
                has_role_name = bool(action.get("element_role")) and bool(action.get("element_name"))
                has_pw = bool(action.get("pw_code"))
                if action_type == ActionTypes.TYPE and not action.get("text"):
                    return False
                return has_id or has_role_name or has_pw

            if action_type == ActionTypes.PAGE_FOCUS:
                try:
                    return int(action.get("page_number", -1)) >= 0
                except Exception:
                    return False

            if action_type == ActionTypes.GOTO_URL:
                url = action.get("url") or ""
                return url.startswith("http://") or url.startswith("https://")

            if action_type == ActionTypes.CHECK:
                return bool(action.get("pw_code"))

            if action_type == ActionTypes.STOP:
                return True

            if action_type == ActionTypes.SEND_MESSAGE_TO_USER:
                message = action.get("answer", "")
                return isinstance(message, str) and len(message) > 0

            # WebRL-specific
            if action_type == ActionTypes.SEARCH:
                return bool(action.get("element_id")) and bool(action.get("text"))
            if action_type == ActionTypes.SELECT_DROPDOWN_OPTION:
                return bool(action.get("element_id")) and bool(action.get("argument"))

            return True
        except Exception:
            return False
    
    def _update_discovery_context(self, action: Action, thoughts: Dict) -> None:
        """Update discovery context with thoughts from send_msg_to_user actions"""
        if not thoughts or not thoughts.get("action"):
            return
            
        action_str = thoughts.get("action")
        
        # Check if this is a send_msg_to_user action
        if action_str.startswith("send_msg_to_user(") or "send_msg_to_user" in action_str:
            # Extract the message content - handle both single and double quotes
            message_match = re.search(r'send_msg_to_user\(["\']([^"\']+)["\']\)', action_str)
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
        images: Optional[List[Any]] = None
    ) -> List[Tuple[str, Action]]:
        """Generate action candidates using nucleus sampling"""
        candidates = []
        self.logger.debug("Generating action candidates (num_samples=%d)", self.num_samples)
        
        # Note: no image processing; only pass through to optional captioner when provided
        
        # Caption the input image, if provided
        # Image inputs are intentionally ignored (no image support required)
        
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
                    
                    # Build trajectory entries as pairs of {THOUGHT, ACTION}
                    trajectory_entries: list[str] = []
                    try:
                        for step in trajectory:
                            if isinstance(step, dict) and step.get("action_type") is not None:
                                # Thought text (if any)
                                thought_text = ""
                                try:
                                    thought_text = (step.get("thoughts", {}) or {}).get("thought", "") or ""
                                except Exception:
                                    thought_text = ""
                                # Action string (prefer thoughts->action, then raw_prediction, then concise)
                                action_str = None
                                thoughts_obj = step.get("thoughts")
                                if isinstance(thoughts_obj, dict) and thoughts_obj.get("action"):
                                    action_str = str(thoughts_obj.get("action"))
                                else:
                                    raw_pred = step.get("raw_prediction")
                                    if isinstance(raw_pred, str):
                                        action_str = (
                                            self._extract_action_from_backticks(raw_pred)
                                            if "```" in raw_pred else raw_pred
                                        )
                                if not action_str:
                                    try:
                                        action_str = self._concise_action_string(step)
                                    except Exception:
                                        action_str = str(step)
                                trajectory_entries.append(
                                    f"{{THOUGHT: {thought_text}, ACTION: {action_str}}}"
                                )
                    except Exception:
                        pass
                    trajectory_str = "\n".join(trajectory_entries[-10:]) if trajectory_entries else "(empty)"
                    
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
                        trajectory=trajectory_str
                    )
                    
                    # System prompt: intro + few-shot examples
                    system_text = enhanced_intro
                    if enhanced_examples:
                        examples_block = []
                        for i, (input_ex, output_ex) in enumerate(enhanced_examples):
                            examples_block.append(f"Example {i+1}:\nInput: {input_ex}\nOutput: {output_ex}")
                        system_text = system_text + "\n\nExamples:\n" + "\n\n".join(examples_block)

                    # User prompt: ONLY current filled template (format instructions are in the prompt JSON)
                    user_text = f"{current_input}"

                    # Build API input
                    api_input = build_api_input_for_text(
                        self.policy_lm_config,
                        system_text,
                        user_text
                    )
                    # Log prompts once per environment step (first sample) at INFO level
                    if sample_idx == 0:
                        try:
                            self.logger.info("=== POLICY SYSTEM PROMPT ===\n%s", system_text)
                            self.logger.info("=== POLICY USER PROMPT ===\n%s", user_text)
                        except Exception:
                            pass
                    response = call_llm(self.policy_lm_config, api_input)
            
            
                # Extract thoughts and action from the new format
                thoughts = self._extract_thoughts_from_response(response)
                parsed_response = thoughts.get("action")
                
                if not parsed_response:
                    self.logger.warning("[Sample %d] No action extracted from response", sample_idx)
                    continue
                
                # Removed redundant extracted action log per sample; keep only Thought->Action below
                if thoughts and isinstance(thoughts, dict):
                    thought_preview = (thoughts.get("thought") or "").replace("\n", " ")[:200]
                    self.logger.debug("[Sample %d] Thought->Action: %s => %s", sample_idx, thought_preview, parsed_response)
                
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
            concise = self._concise_action_string(action)
            reward_prompt = self._create_reward_prompt(action, trajectory, intent, meta_data)
            self.logger.info("Scoring action: %s", concise)
            self.logger.debug("=== REWARD PROMPT ===\n%s", reward_prompt)

            for attempt in range(1, self.reward_score_max_retries + 1):
                # Use system + user_template from JSON prompt for reward model
                if self.reward_prompt and self.reward_prompt.get("system") and self.reward_prompt.get("user_template"):
                    system_text = self.reward_prompt["system"]

                    # Prefer thought/action stored on the action object; fallback to parsing raw_prediction
                    thought_text = ""
                    action_text = ""
                    try:
                        th = action.get("thoughts")
                        if isinstance(th, dict):
                            thought_text = (th.get("thought") or "")
                            action_text = (th.get("action") or "")
                    except Exception:
                        pass
                    if not thought_text:
                        thought_text = self._extract_thoughts_from_response(action.get("raw_prediction", "")).get("thought", "")
                    if not action_text:
                        raw_pred_val = action.get("raw_prediction", str(action))
                        action_text = self._extract_action_from_backticks(raw_pred_val if isinstance(raw_pred_val, str) else str(action))

                    user_text = self.reward_prompt["user_template"].format(
                        intent=intent,
                        trajectory=self._format_trajectory_for_prompt(trajectory),
                        current_url=self._get_current_url(trajectory),
                        text_observation=trajectory[-1]["observation"].get("text", "No text observation"),
                        thought=thought_text,
                        action=action_text
                    )
                    api_input = build_api_input_for_text(self.reward_lm_config, system_text, user_text)
                else:
                    api_input = build_api_input_for_text(
                        self.reward_lm_config,
                        "You are a reward model for web agents. Read the user's content and output only: SCORE: X",
                        reward_prompt,
                    )
                response = call_llm(self.reward_lm_config, api_input)
                # Avoid logging raw response to prevent duplicate 'SCORE: X' lines
                self.logger.debug("=== REWARD MODEL RESPONSE (attempt %d) ===", attempt)

                score_match = re.search(r'SCORE:\s*([1-5])', response)
                if score_match:
                    score = int(score_match.group(1))
                    self.logger.debug("Extracted reward score: %s", score)
                    return float(score)

                self.logger.warning("Reward model returned no valid SCORE on attempt %d; retrying...", attempt)

            self.logger.debug("No valid SCORE after %d attempts, defaulting to 1.0", self.reward_score_max_retries)
            return 1.0

        except Exception as e:
            self.logger.warning("Error scoring action with reward model: %s", e, exc_info=True)
            return 0.0

    def _format_trajectory_for_prompt(self, trajectory: Trajectory) -> str:
        """Format trajectory for inclusion in reward prompt"""
        try:
            if not trajectory:
                return "No trajectory available"

            # Build pairs of (action_step, following_observation_step)
            formatted_steps = []
            step_counter = 1
            # Only look back a reasonable window to keep prompt short
            start_idx = max(0, len(trajectory) - 20)
            i = start_idx
            while i < len(trajectory):
                step = trajectory[i]
                # An action step is stored directly as an action dict with key 'action_type'
                if isinstance(step, dict) and step.get("action_type") is not None:
                    action_info = step
                    # Prefer the concise action extracted from thoughts or from ```...```
                    action_str = None
                    thoughts = action_info.get("thoughts")
                    if isinstance(thoughts, dict) and thoughts.get("action"):
                        action_str = str(thoughts.get("action"))
                    else:
                        raw_pred = action_info.get("raw_prediction")
                        if isinstance(raw_pred, str):
                            action_str = (
                                self._extract_action_from_backticks(raw_pred)
                                if "```" in raw_pred else raw_pred
                            )
                    if not action_str:
                        action_str = str(action_info)
                    # Find the next observation following this action
                    obs_text = "No observation"
                    j = i + 1
                    while j < len(trajectory):
                        nxt = trajectory[j]
                        if isinstance(nxt, dict) and nxt.get("observation") is not None:
                            obs = nxt.get("observation", {})
                            obs_text = str(obs.get("text", "No observation"))[:200]
                            break
                        j += 1

                    formatted_steps.append(
                        f"Step {step_counter}: Action: {action_str}\n  Observation: {obs_text}"
                    )
                    step_counter += 1
                i += 1

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
        
        # Parse formats: THOUGHT/ACTION blocks or legacy "In summary...```action```"
        if isinstance(raw_pred, str) and re.search(r"THOUGHT:\s*", raw_pred, re.IGNORECASE):
            try:
                ta_match = re.search(r"THOUGHT:\s*(.*?)\s*ACTION:\s*```([\s\S]*?)```", raw_pred, re.IGNORECASE | re.DOTALL)
                if ta_match:
                    thought = ta_match.group(1).strip()
                    proposed_action = ta_match.group(2).strip()
                else:
                    proposed_action = self._extract_action_from_backticks(raw_pred)
            except Exception:
                proposed_action = self._extract_action_from_backticks(raw_pred) if "```" in raw_pred else raw_pred
        elif isinstance(raw_pred, str) and "In summary, the next action I will perform is" in raw_pred:
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
            # Preferred: user_template (pairs with separate system text)
            user_tmpl = self.reward_prompt.get("user_template", "")
            if user_tmpl:
                return user_tmpl.format(
                    intent=intent,
                    trajectory=self._format_trajectory_for_prompt(trajectory),
                    current_url=self._get_current_url(trajectory),
                    text_observation=current_text,
                    thought=thought,
                    action=proposed_action
                )
            # Backward compatibility: single prompt template
            prompt_template = self.reward_prompt.get("prompt", "")
            if prompt_template:
                return prompt_template.format(
                    intent=intent,
                    trajectory=self._format_trajectory_for_prompt(trajectory),
                    current_url=self._get_current_url(trajectory),
                    text_observation=current_text,
                    thought=thought,
                    action=proposed_action
                )
            else:
                raise ValueError("Reward prompt template not found in JSON file")
        else:
            # Fallback to default reward prompt
            raise ValueError(
                "Failed to load reward prompt. Please ensure reward_evaluation_prompt.json exists and is valid."
            )
    
    @beartype
    def next_action(
        self, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any], 
        images: Optional[List[Any]] = None,
        output_response: bool = False
    ) -> Action:
        """Generate next action using reward-guided trajectory search"""
        
        # Step 1: Generate action candidates using nucleus sampling
        candidates = self._generate_action_candidates(trajectory, intent, meta_data, images)
        self.logger.debug("Generated %d candidate actions", len(candidates))
        
        if not candidates:
            # Fallback to default action
            action = create_none_action()
            action["raw_prediction"] = "Failed to generate candidates"
            return action
        
        # Step 2: Validate and (for arena) keep ALL candidates even if actions are equivalent
        valid_candidates: list[tuple[str, Action]] = [
            (resp, act) for (resp, act) in candidates if self._is_valid_action(act)
        ]
        if len(valid_candidates) < len(candidates):
            self.logger.info(
                "Filtered %d invalid candidates", len(candidates) - len(valid_candidates)
            )
        top_candidates: list[tuple[str, Action]] = valid_candidates
        try:
            cand_strs = [self._concise_action_string(a) for (_, a) in top_candidates]
            # Derive current step: count actions already taken in trajectory + 1
            try:
                actions_so_far = sum(
                    1
                    for item in trajectory
                    if isinstance(item, dict) and item.get("action_type") is not None
                )
                step_num = actions_so_far + 1
            except Exception:
                step_num = 1
            self.logger.info(
                "Step %d candidates to score (%d): %s", step_num, len(cand_strs), ", ".join(cand_strs)
            )
        except Exception:
            pass
        
        # Step 3: Score candidates using reward model
        scored_candidates = []
        for response, action in top_candidates:
            score = self._score_action_with_reward_model(action, trajectory, intent, meta_data)
            scored_candidates.append((score, response, action))
            self.logger.info("Candidate scored: score=%.3f, action=%s", score, str(action))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Step 4: Select best action
        best_score, best_response, best_action = scored_candidates[0]
        self.logger.info("Best action selected: score=%.3f, action=%s", best_score, str(best_action))
        try:
            best_action["reward_score"] = float(best_score)
        except Exception:
            pass
        
        # Step 5: Update discovery context if this is a send_msg_to_user action
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
        
        self.logger.info("Agent reset: cleared discovery context and reloaded prompts")
    
    def get_discovery_context(self) -> List[Dict]:
        """Get the current discovery context for external use"""
        return self.discovery_context.copy()
