"""Reward functions for different task types in VisualAgentBench"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from browser_env.actions import Action
from browser_env.utils import StateInfo

logger = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Components of a reward function"""
    task_progress: float = 0.0
    efficiency: float = 0.0
    safety: float = 0.0
    user_experience: float = 0.0
    exploration: float = 0.0
    consistency: float = 0.0

class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "task_progress": 0.4,
            "efficiency": 0.2,
            "safety": 0.2,
            "user_experience": 0.1,
            "exploration": 0.05,
            "consistency": 0.05
        }
    
    @abstractmethod
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> float:
        """Compute reward for a state-action pair"""
        pass
    
    def compute_weighted_reward(self, components: RewardComponents) -> float:
        """Compute weighted reward from components"""
        
        total_reward = 0.0
        for component_name, weight in self.weights.items():
            if hasattr(components, component_name):
                component_value = getattr(components, component_name)
                total_reward += weight * component_value
        
        return max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]

class TaskCompletionReward(RewardFunction):
    """Reward function focused on task completion"""
    
    def __init__(self, 
                 target_urls: Optional[List[str]] = None,
                 target_elements: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__(weights)
        self.target_urls = target_urls or []
        self.target_elements = target_elements or []
    
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> float:
        """Compute reward based on task completion progress"""
        
        components = RewardComponents()
        
        # Task progress component
        components.task_progress = self._compute_task_progress(state, action, next_state, context)
        
        # Efficiency component
        components.efficiency = self._compute_efficiency(state, action, next_state)
        
        # Safety component
        components.safety = self._compute_safety(state, action, next_state)
        
        # User experience component
        components.user_experience = self._compute_user_experience(state, action, next_state)
        
        return self.compute_weighted_reward(components)
    
    def _compute_task_progress(self, 
                              state: StateInfo, 
                              action: Action, 
                              next_state: Optional[StateInfo],
                              context: Optional[Dict[str, Any]]) -> float:
        """Compute task progress component"""
        
        if not next_state:
            return 0.0
        
        progress = 0.0
        
        # Check if we're closer to target URLs
        current_url = state.get('url', '')
        next_url = next_state.get('url', '')
        
        for target_url in self.target_urls:
            if target_url in next_url:
                progress += 0.5
            elif target_url in current_url:
                progress += 0.3
        
        # Check if we're closer to target elements
        current_elements = state.get('elements', [])
        next_elements = next_state.get('elements', [])
        
        for target_element in self.target_elements:
            if any(target_element.lower() in str(elem).lower() for elem in next_elements):
                progress += 0.3
            elif any(target_element.lower() in str(elem).lower() for elem in current_elements):
                progress += 0.1
        
        # Check action relevance to task
        action_type = action.get('action_type', '')
        if context and 'task_intent' in context:
            intent = context['task_intent'].lower()
            
            if 'click' in action_type and any(keyword in intent for keyword in ['button', 'link', 'select']):
                progress += 0.2
            elif 'type' in action_type and any(keyword in intent for keyword in ['input', 'search', 'form']):
                progress += 0.2
            elif 'navigate' in action_type and any(keyword in intent for keyword in ['go', 'visit', 'page']):
                progress += 0.2
        
        return min(1.0, progress)
    
    def _compute_efficiency(self, 
                           state: StateInfo, 
                           action: Action, 
                           next_state: Optional[StateInfo]) -> float:
        """Compute efficiency component"""
        
        efficiency = 0.5  # Base efficiency
        
        # Prefer actions that change state meaningfully
        if next_state and next_state != state:
            efficiency += 0.2
        
        # Prefer actions that don't repeat previous actions
        action_type = action.get('action_type', '')
        if action_type != 'wait':
            efficiency += 0.1
        
        # Prefer actions that target specific elements
        target = action.get('target', '')
        if target and target != 'generic':
            efficiency += 0.2
        
        return min(1.0, efficiency)
    
    def _compute_safety(self, 
                        state: StateInfo, 
                        action: Action, 
                        next_state: Optional[StateInfo]) -> float:
        """Compute safety component"""
        
        safety = 0.8  # Base safety
        
        action_type = action.get('action_type', '')
        
        # Some actions are inherently safer
        if action_type in ['wait', 'scroll']:
            safety += 0.2
        elif action_type in ['click', 'type']:
            # Check if target is safe
            target = action.get('target', '')
            if 'delete' in target.lower() or 'remove' in target.lower():
                safety -= 0.3
            elif 'confirm' in target.lower() or 'submit' in target.lower():
                safety -= 0.1
        
        return max(0.0, min(1.0, safety))
    
    def _compute_user_experience(self, 
                                state: StateInfo, 
                                action: Action, 
                                next_state: Optional[StateInfo]) -> float:
        """Compute user experience component"""
        
        ux = 0.5  # Base UX
        
        action_type = action.get('action_type', '')
        
        # Prefer natural user interactions
        if action_type in ['click', 'type']:
            ux += 0.2
        
        # Avoid excessive waiting
        if action_type == 'wait':
            value = action.get('value', '1')
            try:
                wait_time = float(value)
                if wait_time <= 2:
                    ux += 0.1
                else:
                    ux -= 0.2
            except ValueError:
                ux -= 0.1
        
        return max(0.0, min(1.0, ux))

class EfficiencyReward(RewardFunction):
    """Reward function focused on efficiency and speed"""
    
    def __init__(self, 
                 max_steps: int = 10,
                 time_penalty: float = 0.1,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__(weights)
        self.max_steps = max_steps
        self.time_penalty = time_penalty
    
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> float:
        """Compute reward based on efficiency"""
        
        components = RewardComponents()
        
        # Efficiency component (primary focus)
        components.efficiency = self._compute_efficiency(state, action, next_state, context)
        
        # Task progress component
        components.task_progress = self._compute_task_progress(state, action, next_state, context)
        
        # Safety component
        components.safety = self._compute_safety(state, action, next_state)
        
        # Consistency component
        components.consistency = self._compute_consistency(state, action, next_state)
        
        return self.compute_weighted_reward(components)
    
    def _compute_efficiency(self, 
                           state: StateInfo, 
                           action: Action, 
                           next_state: Optional[StateInfo],
                           context: Optional[Dict[str, Any]]) -> float:
        """Compute efficiency component with focus on speed"""
        
        efficiency = 0.5  # Base efficiency
        
        # Prefer actions that make significant progress
        if next_state and next_state != state:
            efficiency += 0.3
            
            # Check if we're making progress toward goal
            if context and 'task_intent' in context:
                intent = context['task_intent'].lower()
                current_url = state.get('url', '')
                next_url = next_state.get('url', '')
                
                # URL changes often indicate progress
                if next_url != current_url:
                    efficiency += 0.2
        
        # Prefer actions that don't waste time
        action_type = action.get('action_type', '')
        if action_type == 'wait':
            value = action.get('value', '1')
            try:
                wait_time = float(value)
                if wait_time <= 1:
                    efficiency += 0.1
                else:
                    efficiency -= wait_time * self.time_penalty
            except ValueError:
                efficiency -= 0.1
        elif action_type in ['click', 'type']:
            efficiency += 0.1
        
        # Penalize actions that don't change state
        if next_state == state:
            efficiency -= 0.2
        
        return max(0.0, min(1.0, efficiency))
    
    def _compute_task_progress(self, 
                              state: StateInfo, 
                              action: Action, 
                              next_state: Optional[StateInfo],
                              context: Optional[Dict[str, Any]]) -> float:
        """Compute task progress component"""
        
        progress = 0.0
        
        if context and 'task_intent' in context:
            intent = context['task_intent'].lower()
            
            # Check if action is relevant to task
            action_type = action.get('action_type', '')
            target = action.get('target', '')
            
            if 'order' in intent and 'view' in intent:
                if 'order' in target.lower() or 'order' in str(next_state).lower():
                    progress += 0.5
            elif 'admin' in intent:
                if 'admin' in str(next_state).lower():
                    progress += 0.5
            elif 'login' in intent:
                if 'logout' in str(next_state).lower() or 'profile' in str(next_state).lower():
                    progress += 0.5
        
        return min(1.0, progress)
    
    def _compute_safety(self, 
                        state: StateInfo, 
                        action: Action, 
                        next_state: Optional[StateInfo]) -> float:
        """Compute safety component"""
        
        safety = 0.8  # Base safety
        
        action_type = action.get('action_type', '')
        target = action.get('target', '')
        
        # Penalize potentially dangerous actions
        if 'delete' in target.lower() or 'remove' in target.lower():
            safety -= 0.4
        elif 'confirm' in target.lower() and 'delete' in str(state).lower():
            safety -= 0.3
        
        return max(0.0, min(1.0, safety))
    
    def _compute_consistency(self, 
                            state: StateInfo, 
                            action: Action, 
                            next_state: Optional[StateInfo]) -> float:
        """Compute consistency component"""
        
        consistency = 0.7  # Base consistency
        
        # Prefer actions that follow a logical sequence
        action_type = action.get('action_type', '')
        
        if action_type in ['click', 'type']:
            # These are typically good follow-up actions
            consistency += 0.2
        elif action_type == 'wait':
            # Waiting is sometimes necessary but not always
            consistency += 0.1
        
        return min(1.0, consistency)

class SafetyFirstReward(RewardFunction):
    """Reward function that prioritizes safety over speed"""
    
    def __init__(self, 
                 dangerous_keywords: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__(weights)
        self.dangerous_keywords = dangerous_keywords or [
            'delete', 'remove', 'drop', 'trash', 'cancel', 'abort'
        ]
    
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> float:
        """Compute reward with safety as primary concern"""
        
        components = RewardComponents()
        
        # Safety component (primary focus)
        components.safety = self._compute_safety(state, action, next_state)
        
        # Task progress component
        components.task_progress = self._compute_task_progress(state, action, next_state, context)
        
        # Efficiency component
        components.efficiency = self._compute_efficiency(state, action, next_state)
        
        # User experience component
        components.user_experience = self._compute_user_experience(state, action, next_state)
        
        return self.compute_weighted_reward(components)
    
    def _compute_safety(self, 
                        state: StateInfo, 
                        action: Action, 
                        next_state: Optional[StateInfo]) -> float:
        """Compute safety component with high priority"""
        
        safety = 1.0  # Start with perfect safety
        
        action_type = action.get('action_type', '')
        target = action.get('target', '')
        value = action.get('value', '')
        
        # Check for dangerous keywords in target
        for keyword in self.dangerous_keywords:
            if keyword.lower() in target.lower():
                safety -= 0.5
                break
        
        # Check for dangerous keywords in value
        for keyword in self.dangerous_keywords:
            if keyword.lower() in str(value).lower():
                safety -= 0.3
                break
        
        # Check page context for dangerous elements
        if next_state:
            page_content = str(next_state).lower()
            for keyword in self.dangerous_keywords:
                if keyword in page_content:
                    safety -= 0.2
                    break
        
        # Some action types are inherently safer
        if action_type == 'wait':
            safety += 0.1
        elif action_type in ['scroll', 'navigate']:
            safety += 0.05
        
        return max(0.0, min(1.0, safety))
    
    def _compute_task_progress(self, 
                              state: StateInfo, 
                              action: Action, 
                              next_state: Optional[StateInfo],
                              context: Optional[Dict[str, Any]]) -> float:
        """Compute task progress component"""
        
        progress = 0.0
        
        if context and 'task_intent' in context:
            intent = context['task_intent'].lower()
            
            # Check if action advances the task safely
            action_type = action.get('action_type', '')
            target = action.get('target', '')
            
            if 'order' in intent and 'view' in intent:
                if 'order' in target.lower() and 'delete' not in target.lower():
                    progress += 0.4
            elif 'admin' in intent:
                if 'admin' in str(next_state).lower():
                    progress += 0.4
        
        return min(1.0, progress)
    
    def _compute_efficiency(self, 
                           state: StateInfo, 
                           action: Action, 
                           next_state: Optional[StateInfo]) -> float:
        """Compute efficiency component (lower priority for safety-first)"""
        
        efficiency = 0.5  # Base efficiency
        
        # Prefer actions that make progress
        if next_state and next_state != state:
            efficiency += 0.2
        
        # Avoid excessive waiting
        if action.get('action_type') == 'wait':
            try:
                wait_time = float(action.get('value', '1'))
                if wait_time <= 3:
                    efficiency += 0.1
                else:
                    efficiency -= 0.1
            except ValueError:
                pass
        
        return max(0.0, min(1.0, efficiency))
    
    def _compute_user_experience(self, 
                                state: StateInfo, 
                                action: Action, 
                                next_state: Optional[StateInfo]) -> float:
        """Compute user experience component"""
        
        ux = 0.6  # Base UX
        
        action_type = action.get('action_type', '')
        
        # Prefer natural interactions
        if action_type in ['click', 'type']:
            ux += 0.2
        
        # Avoid actions that might cause errors
        target = action.get('target', '')
        if any(keyword in target.lower() for keyword in self.dangerous_keywords):
            ux -= 0.3
        
        return max(0.0, min(1.0, ux))

class CompositeReward(RewardFunction):
    """Composite reward function that combines multiple reward functions"""
    
    def __init__(self, 
                 reward_functions: List[Tuple[RewardFunction, float]],
                 weights: Optional[Dict[str, float]] = None):
        super().__init__(weights)
        self.reward_functions = reward_functions  # List of (function, weight) tuples
    
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> float:
        """Compute composite reward from multiple functions"""
        
        total_reward = 0.0
        total_weight = 0.0
        
        for reward_func, weight in self.reward_functions:
            try:
                reward_value = reward_func.compute_reward(state, action, next_state, context)
                total_reward += weight * reward_value
                total_weight += weight
            except Exception as e:
                logger.warning(f"Error computing reward with {reward_func.__class__.__name__}: {e}")
        
        if total_weight > 0:
            return total_reward / total_weight
        else:
            return 0.0

def create_task_specific_reward(task_type: str, **kwargs) -> RewardFunction:
    """Factory function to create task-specific reward functions"""
    
    if task_type == "order_management":
        return TaskCompletionReward(
            target_urls=["order", "admin"],
            target_elements=["order", "view", "details"],
            weights={
                "task_progress": 0.5,
                "efficiency": 0.2,
                "safety": 0.2,
                "user_experience": 0.1
            }
        )
    
    elif task_type == "efficient_navigation":
        return EfficiencyReward(
            max_steps=8,
            time_penalty=0.15,
            weights={
                "efficiency": 0.5,
                "task_progress": 0.3,
                "safety": 0.1,
                "consistency": 0.1
            }
        )
    
    elif task_type == "safe_operations":
        return SafetyFirstReward(
            dangerous_keywords=["delete", "remove", "drop", "trash"],
            weights={
                "safety": 0.6,
                "task_progress": 0.2,
                "efficiency": 0.1,
                "user_experience": 0.1
            }
        )
    
    else:
        # Default reward function
        return TaskCompletionReward()
