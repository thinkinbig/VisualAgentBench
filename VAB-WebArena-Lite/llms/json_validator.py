"""
JSON response validator for LLM outputs
"""
import json
import re
from typing import Dict, Any, Tuple, Optional
from pydantic import ValidationError
from .types import LLMResponse, PolicyResponse, ThoughtActionPair, ParsedAction, ActionType


class JSONResponseValidator:
    """Validates and fixes JSON responses from LLMs"""
    
    def __init__(self):
        self.action_patterns = {
            'click': r'click\s*\[([^\]]+)\]',
            'type': r'type\s*\[([^\]]+)\]\s*\[([^\]]+)\](?:\s*\[([01])\])?',
            'hover': r'hover\s*\[([^\]]+)\]',
            'press': r'press\s*\[([^\]]+)\]',
            'scroll': r'scroll\s*\[?(up|down)\]?',
            'goto': r'goto\s*\[([^\]]+)\]',
            'new_tab': r'new_tab',
            'tab_focus': r'tab_focus\s*\[(\d+)\]',
            'close_tab': r'close_tab',
            'go_back': r'go_back',
            'go_forward': r'go_forward',
            'send_msg_to_user': r'send_msg_to_user\s*\[([^\]]+)\]'
        }
    
    def validate_response(self, raw_response: str) -> LLMResponse:
        """Validate and parse LLM response"""
        response = LLMResponse(raw_response=raw_response)
        
        try:
            # Try to parse as structured JSON response first
            agent_response = self._parse_structured_response(raw_response)
            if agent_response:
                response.agent_response = agent_response
                response.is_valid = True
                return response
        except Exception as e:
            response.validation_errors.append(f"Structured parsing failed: {str(e)}")
        
        # Try to parse as thought-action pair
        try:
            thought_action = self._parse_thought_action(raw_response)
            if thought_action:
                response.thought_action = thought_action
                response.is_valid = True
                return response
        except Exception as e:
            response.validation_errors.append(f"Thought-action parsing failed: {str(e)}")
        
        # If all parsing fails, mark as invalid
        response.validation_errors.append("Unable to parse response in any expected format")
        return response
    
    def _parse_structured_response(self, response: str) -> Optional[PolicyResponse]:
        """Parse structured JSON response with CHECKPOINT/AGGREGATE/BLOCK"""
        # Look for JSON-like structure
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
        
        json_str = json_match.group(0)
        try:
            json_data = json.loads(json_str)
            return PolicyResponse.parse_obj(json_data)
        except (json.JSONDecodeError, ValidationError):
            return None
    
    def _parse_thought_action(self, response: str) -> Optional[ThoughtActionPair]:
        """Parse thought-action pair format"""
        # Look for THOUGHT: ... ACTION: ... pattern
        thought_action_match = re.search(
            r"THOUGHT:\s*(.*?)\s*ACTION:\s*```?([\s\S]*?)```?", 
            response, 
            re.IGNORECASE | re.DOTALL
        )
        
        if thought_action_match:
            thought = thought_action_match.group(1).strip()
            action = thought_action_match.group(2).strip()
            
            parsed_action = self._parse_action(action)
            
            return ThoughtActionPair(
                thought=thought,
                action=action,
                parsed_action=parsed_action
            )
        
        # Fallback: look for action in backticks
        action_match = re.search(r'```([\s\S]*?)```', response)
        if action_match:
            action = action_match.group(1).strip()
            parsed_action = self._parse_action(action)
            
            return ThoughtActionPair(
                thought="",
                action=action,
                parsed_action=parsed_action
            )
        
        return None
    
    def _parse_action(self, action_str: str) -> Optional[ParsedAction]:
        """Parse action string into structured format"""
        action_str = action_str.strip()
        
        for action_type, pattern in self.action_patterns.items():
            match = re.search(pattern, action_str, re.IGNORECASE)
            if match:
                try:
                    return self._create_parsed_action(action_type, match)
                except Exception:
                    continue
        
        return None
    
    def _create_parsed_action(self, action_type: str, match: re.Match) -> ParsedAction:
        """Create ParsedAction from regex match"""
        action_data = {"action_type": ActionType(action_type)}
        
        if action_type == "click":
            action_data["element_id"] = match.group(1)
        elif action_type == "type":
            action_data["element_id"] = match.group(1)
            action_data["content"] = match.group(2)
            if match.group(3):
                action_data["press_enter_after"] = match.group(3) == "1"
        elif action_type == "hover":
            action_data["element_id"] = match.group(1)
        elif action_type == "press":
            action_data["key_combination"] = match.group(1)
        elif action_type == "scroll":
            action_data["direction"] = match.group(1) if match.group(1) else "down"
        elif action_type == "goto":
            action_data["url"] = match.group(1)
        elif action_type == "tab_focus":
            action_data["tab_index"] = int(match.group(1))
        elif action_type == "send_msg_to_user":
            action_data["content"] = match.group(1)
        
        return ParsedAction(**action_data)
    
    def fix_json_format(self, response: str) -> str:
        """Attempt to fix common JSON formatting issues"""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response)
        
        # Fix common JSON issues
        response = re.sub(r'(\w+):', r'"\1":', response)  # Add quotes to keys
        response = re.sub(r':\s*([^"\[\{][^,\}\]]*)', r': "\1"', response)  # Quote unquoted values
        response = re.sub(r',\s*}', '}', response)  # Remove trailing commas
        response = re.sub(r',\s*]', ']', response)  # Remove trailing commas in arrays
        
        return response.strip()
    
    def validate_and_retry(self, response: str, max_attempts: int = 3) -> LLMResponse:
        """Validate response and attempt fixes if needed"""
        # First attempt: validate as-is
        result = self.validate_response(response)
        if result.is_valid:
            return result
        
        # Second attempt: try fixing JSON format
        if max_attempts > 1:
            fixed_response = self.fix_json_format(response)
            result = self.validate_response(fixed_response)
            if result.is_valid:
                return result
        
        # Third attempt: extract just the action part
        if max_attempts > 2:
            action_match = re.search(r'```([\s\S]*?)```', response)
            if action_match:
                action_only = action_match.group(1).strip()
                thought_action = ThoughtActionPair(
                    thought="",
                    action=action_only,
                    parsed_action=self._parse_action(action_only)
                )
                return LLMResponse(
                    raw_response=response,
                    thought_action=thought_action,
                    is_valid=thought_action.parsed_action is not None
                )
        
        return result
