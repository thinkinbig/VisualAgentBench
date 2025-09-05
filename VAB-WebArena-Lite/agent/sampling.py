"""
Sampling utilities for reward-guided agent.

This module handles nucleus sampling and dynamic parameter generation
for diverse candidate generation.
"""

import random
from typing import Tuple, Dict, Any, List

from llms import lm_config
from .types import BlockInfo


class NucleusSampler:
    """Handles nucleus sampling parameter generation and configuration."""
    
    def __init__(self, nucleus_config: Dict[str, Any]):
        self.config = nucleus_config or {
            "enabled": False,
            "top_p_range": [0.3, 0.9],
            "temperature_range": [0.8, 1.2],
            "diversity_strategy": "random",
            "fallback_top_p": 0.9,
            "fallback_temperature": 1.0
        }
    
    def get_sampling_params(self, attempt: int, total_attempts: int) -> Tuple[float, float]:
        """Generate dynamic sampling parameters for nucleus sampling.
        
        Args:
            attempt: Current attempt number (0-based)
            total_attempts: Total number of attempts planned
            
        Returns:
            Tuple of (temperature, top_p) for this sampling attempt
        """
        if not self.config.get("enabled", False):
            # Fallback to original config values
            return (
                self.config.get("fallback_temperature", 1.0),
                self.config.get("fallback_top_p", 0.9)
            )
        
        strategy = self.config.get("diversity_strategy", "random")
        top_p_range = self.config.get("top_p_range", [0.3, 0.9])
        temp_range = self.config.get("temperature_range", [0.8, 1.2])
        
        if strategy == "random":
            # Random sampling within ranges
            temperature = random.uniform(temp_range[0], temp_range[1])
            top_p = random.uniform(top_p_range[0], top_p_range[1])
        elif strategy == "progressive":
            # Progressive sampling: start conservative, become more diverse
            progress = attempt / max(1, total_attempts - 1)
            temperature = temp_range[0] + progress * (temp_range[1] - temp_range[0])
            top_p = top_p_range[0] + progress * (top_p_range[1] - top_p_range[0])
        elif strategy == "alternating":
            # Alternating between conservative and diverse
            if attempt % 2 == 0:
                temperature = temp_range[0]  # Conservative
                top_p = top_p_range[0]
            else:
                temperature = temp_range[1]  # Diverse
                top_p = top_p_range[1]
        else:
            # Default to random
            temperature = random.uniform(temp_range[0], temp_range[1])
            top_p = random.uniform(top_p_range[0], top_p_range[1])
        
        return temperature, top_p

    def get_aggressive_sampling_params(self, attempt: int, total_attempts: int) -> Tuple[float, float]:
        """Generate more aggressive sampling parameters for best-of-n sampling.
        
        Args:
            attempt: Current attempt number (0-based)
            total_attempts: Total number of attempts planned
            
        Returns:
            Tuple of (temperature, top_p) for this sampling attempt
        """
        if not self.config.get("enabled", False):
            # Use more aggressive defaults for best-of-n
            return (1.5, 0.7)
        
        # More aggressive ranges for diversity
        temp_range = [1.0, 2.0]  # Higher temperature range
        top_p_range = [0.3, 0.8]  # Lower top_p for more diversity
        
        strategy = self.config.get("diversity_strategy", "random")
        
        if strategy == "random":
            # Random sampling within aggressive ranges
            temperature = random.uniform(temp_range[0], temp_range[1])
            top_p = random.uniform(top_p_range[0], top_p_range[1])
        elif strategy == "progressive":
            # Progressive sampling: start diverse, become more diverse
            progress = attempt / max(1, total_attempts - 1)
            temperature = temp_range[0] + progress * (temp_range[1] - temp_range[0])
            top_p = top_p_range[0] + progress * (top_p_range[1] - top_p_range[0])
        elif strategy == "alternating":
            # Alternating between moderate and very diverse
            if attempt % 2 == 0:
                temperature = temp_range[0]  # Moderate
                top_p = top_p_range[0]
            else:
                temperature = temp_range[1]  # Very diverse
                top_p = top_p_range[1]
        else:
            # Default to random with aggressive ranges
            temperature = random.uniform(temp_range[0], temp_range[1])
            top_p = random.uniform(top_p_range[0], top_p_range[1])
        
        return temperature, top_p

    def create_dynamic_lm_config(self, base_config: lm_config.LMConfig, 
                                temperature: float, top_p: float) -> lm_config.LMConfig:
        """Create a new LMConfig with dynamic sampling parameters.
        
        Args:
            base_config: Base configuration to modify
            temperature: New temperature value
            top_p: New top_p value
            
        Returns:
            New LMConfig with updated parameters
        """
        # Create a copy of the gen_config with updated parameters
        new_gen_config = base_config.gen_config.copy()
        new_gen_config["temperature"] = temperature
        new_gen_config["top_p"] = top_p
        
        # Create new LMConfig with updated parameters
        return lm_config.LMConfig(
            provider=base_config.provider,
            model=base_config.model,
            mode=base_config.mode,
            gen_config=new_gen_config
        )


class CandidateSelector:
    """Handles candidate selection and diversity optimization."""
    
    @staticmethod
    def select_best_candidates(candidates: List[BlockInfo], target_count: int) -> List[BlockInfo]:
        """Select the best candidates from a large pool using diversity and quality metrics.
        
        Args:
            candidates: List of all generated candidates
            target_count: Number of candidates to select
            
        Returns:
            List of selected best candidates
        """
        if len(candidates) <= target_count:
            return candidates
        
        # Group candidates by action to measure diversity
        action_groups: Dict[str, List[BlockInfo]] = {}
        for candidate in candidates:
            action = candidate.action or ""
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append(candidate)
        
        # Calculate diversity score (number of unique actions)
        unique_actions = len(action_groups)
        
        # If we have enough unique actions, select one from each group
        if unique_actions >= target_count:
            selected = []
            for action, group in action_groups.items():
                if len(selected) >= target_count:
                    break
                # Select the first (or best) candidate from each group
                selected.append(group[0])
            return selected
        
        # If not enough unique actions, use diversity + quality selection
        selected = []
        
        # First, select one from each unique action group
        for action, group in action_groups.items():
            selected.append(group[0])
        
        # If we still need more candidates, select from remaining groups
        remaining_needed = target_count - len(selected)
        if remaining_needed > 0:
            # Sort groups by size (larger groups might have better quality)
            sorted_groups = sorted(action_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            for action, group in sorted_groups:
                if len(selected) >= target_count:
                    break
                # Add additional candidates from larger groups
                for candidate in group[1:]:  # Skip first one (already selected)
                    if len(selected) >= target_count:
                        break
                    selected.append(candidate)
        
        return selected
