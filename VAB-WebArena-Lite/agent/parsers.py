"""
Parsers for reward-guided agent responses and action validation.

This module centralizes parsing logic that was previously scattered across
the RewardGuidedAgent class, making the code more modular and maintainable.
"""

import json
import re
from typing import Optional, List, Dict, Any

from .types import BlockInfo, PairwiseDecision


class BlockParser:
    """Parser for BLOCK responses from policy LLM."""
    
    @staticmethod
    def parse_single_block(raw: str) -> Optional[BlockInfo]:
        """Parse a single BLOCK from LLM response."""
        if not isinstance(raw, str) or not raw.strip():
            return None
        
        # Try JSON format first
        try:
            import re as _re
            m = _re.search(r"\{[\s\S]*\}", raw)
            if m:
                json_str = m.group(0)
                # Clean trailing commas before parsing
                json_str = _re.sub(r',(\s*[}\]])', r'\1', json_str)
                data = json.loads(json_str)
                blk = data.get("BLOCK") if isinstance(data, dict) else None
                if isinstance(blk, dict):
                    thought = blk.get("thought") or ""
                    action = blk.get("action") or ""
                    # Strip backticks if present
                    if isinstance(action, str):
                        s = action.strip()
                        if s.startswith("```") and s.endswith("```"):
                            s = s[3:-3].strip()
                        # Normalize common missing-bracket case for goto
                        try:
                            mm = _re.match(r"^goto\s+(https?://[^\s\]]+)$", s, flags=_re.IGNORECASE)
                            if mm:
                                s = f"goto [{mm.group(1)}]"
                        except Exception:
                            pass
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

    @staticmethod
    def parse_multiple_blocks(raw: str) -> List[BlockInfo]:
        """Parse multiple BLOCKs from LLM response."""
        blocks = []
        if not isinstance(raw, str) or not raw.strip():
            return blocks
        
        try:
            import re as _re
            # Try to find JSON with BLOCKS array
            m = _re.search(r"\{[\s\S]*\"BLOCKS\"[\s\S]*\}", raw)
            if m:
                json_str = m.group(0)
                # Clean trailing commas before parsing
                json_str = _re.sub(r',(\s*[}\]])', r'\1', json_str)
                data = json.loads(json_str)
                blocks_data = data.get("BLOCKS", [])
                if isinstance(blocks_data, list):
                    for block_data in blocks_data:
                        if isinstance(block_data, dict):
                            thought = block_data.get("thought") or ""
                            action = block_data.get("action") or ""
                            # Strip backticks if present
                            if isinstance(action, str):
                                s = action.strip()
                                if s.startswith("```") and s.endswith("```"):
                                    s = s[3:-3].strip()
                                # Normalize common missing-bracket case for goto
                                try:
                                    mm = _re.match(r"^goto\s+(https?://[^\s\]]+)$", s, flags=_re.IGNORECASE)
                                    if mm:
                                        s = f"goto [{mm.group(1)}]"
                                except Exception:
                                    pass
                                action = s
                            if isinstance(thought, str) and isinstance(action, str) and action:
                                blocks.append(BlockInfo(thought=thought, action=action))
        except Exception:
            pass
        
        # If no BLOCKS found, try fallback to single BLOCK
        if not blocks:
            single_block = BlockParser.parse_single_block(raw)
            if single_block:
                blocks.append(single_block)
        
        return blocks


class RewardParser:
    """Parser for reward responses from reward LLM."""
    
    @staticmethod
    def parse_decision(raw: str) -> PairwiseDecision:
        """Parse pairwise decision from reward LLM response."""
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Empty or invalid LLM response for reward decision")
        
        # Try XML format first (from detailed prompt)
        xml_match = re.search(r"<Answer>\s*(Response\s*[12])\s*</Answer>", raw, re.IGNORECASE)
        if xml_match:
            response = xml_match.group(1).strip().lower()
            if "response 1" in response or "response1" in response:
                return PairwiseDecision.RESPONSE_1
            elif "response 2" in response or "response2" in response:
                return PairwiseDecision.RESPONSE_2
        
        # Fallback to JSON format (from simple prompt)
        try:
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                json_str = json_match.group(0)
                # Clean trailing commas before parsing
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                data = json.loads(json_str)
                dec = (data.get("decision") or "").strip().lower()
                if dec in (PairwiseDecision.RESPONSE_1.value, PairwiseDecision.RESPONSE_2.value):
                    return PairwiseDecision(dec)
        except Exception:
            pass
        
        # If we get here, the response format is invalid
        raise ValueError(f"Invalid LLM response format for reward decision: {raw[:200]}...")


class ActionValidator:
    """Validator for action strings before parsing."""
    
    @staticmethod
    def is_valid_action(action_str: str, obs_nodes_info: Optional[Dict[str, Any]] = None) -> bool:
        """Validate action string format and references."""
        try:
            s = (action_str or "").strip()
            
            # Special validation for type actions - must have proper bracket format
            if s.lower().startswith("type"):
                # Check for proper type action format: type [id] content (auto-enter)
                type_match = re.match(r"^type\s*\[([^\]]+)\]\s+(.+)$", s, flags=re.IGNORECASE)
                if type_match:
                    elem_id = type_match.group(1).strip()
                    if obs_nodes_info is not None:
                        return isinstance(obs_nodes_info, dict) and elem_id in obs_nodes_info
                    return True  # If no obs_nodes_info provided, assume valid format
                else:
                    # Reject malformed type actions (missing brackets around id or no content)
                    return False
            
            # click/hover must reference existing AXTREE id
            m = re.match(r"^(click|hover)\s*\[([^\]]+)\]", s, flags=re.IGNORECASE)
            if m:
                elem_id = m.group(2).strip()
                if obs_nodes_info is not None:
                    return isinstance(obs_nodes_info, dict) and elem_id in obs_nodes_info
                return True  # If no obs_nodes_info provided, assume valid format
            
            # goto must include an http(s) URL inside brackets
            g = re.match(r"^goto\s*\[([^\]]+)\]", s, flags=re.IGNORECASE)
            if g:
                url = g.group(1).strip().lower()
                return url.startswith("http://") or url.startswith("https://")
            
            # press/scroll/go_back/go_forward/send_msg_to_user are considered syntactically valid here
            return True
        except Exception:
            return False


class ObservationParser:
    """Parser for observation data from browser environment."""
    
    @staticmethod
    def parse_ax_observation(observation: Optional[str]) -> List[Dict[str, str]]:
        """Parse AXTREE observation into structured elements."""
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
