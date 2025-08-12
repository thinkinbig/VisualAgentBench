"""Reward Model for guiding search in VisualAgentBench"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from browser_env.actions import Action
from browser_env.utils import StateInfo, Observation
from llms import generate_from_openai_chat_completion

from .models import RewardSignal
from .enums import RewardComponent, SafetyLevel

logger = logging.getLogger(__name__)

class RewardModel(ABC):
    """Abstract base class for reward models"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> RewardSignal:
        """Compute reward for a state-action pair"""
        pass
    
    @abstractmethod
    def compute_state_value(self, 
                           state: StateInfo,
                           context: Optional[Dict[str, Any]] = None) -> float:
        """Compute value of a state"""
        pass
    
    def save_model(self, path: str):
        """Save the reward model"""
        if hasattr(self, 'state_dict'):
            torch.save(self.state_dict(), path)
            logger.info(f"Reward model saved to {path}")
    
    def load_model(self, path: str):
        """Load the reward model"""
        if hasattr(self, 'state_dict'):
            self.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Reward model loaded from {path}")

class LLMBasedRewardModel(RewardModel):
    """Reward model based on LLM evaluation"""
    
    def __init__(self, 
                 llm_model: str = "gpt-4-1106-preview",
                 temperature: float = 0.0):
        super().__init__()
        self.llm_model = llm_model
        self.temperature = temperature
        
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> RewardSignal:
        """Compute reward using LLM evaluation"""
        
        # Construct prompt for reward evaluation
        prompt = self._construct_reward_prompt(state, action, next_state, context)
        
        try:
            response = generate_from_openai_chat_completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator assessing the quality of actions in a web automation task."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            # Parse the response to extract reward components
            reward_data = self._parse_reward_response(response)
            
            return RewardSignal(
                value=reward_data.get("total_reward", 0.0),
                components=reward_data.get("components", {}),
                confidence=reward_data.get("confidence", 0.5),
                metadata=reward_data
            )
            
        except Exception as e:
            logger.error(f"Error computing LLM-based reward: {e}")
            return RewardSignal(
                value=0.0,
                components={"error": 0.0},
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def compute_state_value(self, 
                           state: StateInfo,
                           context: Optional[Dict[str, Any]] = None) -> float:
        """Compute state value using LLM evaluation"""
        
        prompt = self._construct_state_value_prompt(state, context)
        
        try:
            response = generate_from_openai_chat_completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator assessing the value of states in a web automation task."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )
            
            # Parse response for state value
            try:
                value = float(response.strip())
                return max(0.0, min(1.0, value))  # Clamp to [0, 1]
            except ValueError:
                logger.warning(f"Could not parse state value from response: {response}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error computing state value: {e}")
            return 0.5
    
    def _construct_reward_prompt(self, 
                                state: StateInfo, 
                                action: Action, 
                                next_state: Optional[StateInfo],
                                context: Optional[Dict[str, Any]]) -> str:
        """Construct prompt for reward evaluation"""
        
        prompt = f"""Evaluate the quality of an action in a web automation task.

Current State:
- URL: {state.get('url', 'N/A')}
- Page Title: {state.get('page_title', 'N/A')}
- Available Elements: {len(state.get('elements', []))} elements

Action Taken:
- Type: {action.get('action_type', 'N/A')}
- Target: {action.get('target', 'N/A')}
- Value: {action.get('value', 'N/A')}

{f"Next State URL: {next_state.get('url', 'N/A')}" if next_state else "No next state available"}

Context: {context.get('task_intent', 'N/A') if context else 'N/A'}

Please evaluate this action and provide:
1. Total reward (0-1): Overall quality score
2. Component scores (0-1 each):
   - task_progress: How much this action advances the task
   - efficiency: How efficient this action is
   - safety: How safe this action is
   - user_experience: Impact on user experience
3. Confidence (0-1): How confident you are in this evaluation

Format your response as JSON:
{{
    "total_reward": 0.8,
    "components": {{
        "task_progress": 0.9,
        "efficiency": 0.7,
        "safety": 0.8,
        "user_experience": 0.8
    }},
    "confidence": 0.9
}}"""
        
        return prompt
    
    def _construct_state_value_prompt(self, 
                                    state: StateInfo,
                                    context: Optional[Dict[str, Any]]) -> str:
        """Construct prompt for state value evaluation"""
        
        prompt = f"""Evaluate the value of a state in a web automation task.

State Information:
- URL: {state.get('url', 'N/A')}
- Page Title: {state.get('page_title', 'N/A')}
- Available Elements: {len(state.get('elements', []))} elements

Context: {context.get('task_intent', 'N/A') if context else 'N/A'}

Rate this state from 0.0 to 1.0, where:
- 0.0: Completely unhelpful for the task
- 0.5: Neutral/uncertain
- 1.0: Perfect state for task completion

Provide only the numerical score (e.g., 0.8):"""
        
        return prompt
    
    def _parse_reward_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract reward data"""
        
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                logger.warning("Could not find JSON in response, using fallback")
                return {
                    "total_reward": 0.5,
                    "components": {"fallback": 0.5},
                    "confidence": 0.5
                }
                
        except Exception as e:
            logger.error(f"Failed to parse reward response: {e}")
            return {
                "total_reward": 0.5,
                "components": {"error": 0.5},
                "confidence": 0.0
            }

class NeuralRewardModel(RewardModel, nn.Module):
    """Neural network-based reward model"""
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 1):
        RewardModel.__init__(self)
        nn.Module.__init__(self)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network"""
        return self.network(x)
    
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> RewardSignal:
        """Compute reward using neural network"""
        
        # Convert state and action to feature vector
        features = self._extract_features(state, action, next_state, context)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reward_value = self.forward(features_tensor).item()
        
        return RewardSignal(
            value=reward_value,
            components={"neural_reward": reward_value},
            confidence=0.8,  # Neural models typically have high confidence
            metadata={"model_type": "neural"}
        )
    
    def compute_state_value(self, 
                           state: StateInfo,
                           context: Optional[Dict[str, Any]] = None) -> float:
        """Compute state value using neural network"""
        
        # Extract state features
        features = self._extract_state_features(state, context)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.forward(features_tensor).item()
        
        return value
    
    def _extract_features(self, 
                         state: StateInfo, 
                         action: Action, 
                         next_state: Optional[StateInfo],
                         context: Optional[Dict[str, Any]]) -> List[float]:
        """Extract features from state, action, and context"""
        
        features = []
        
        # State features
        features.extend(self._extract_state_features(state, context))
        
        # Action features
        features.extend(self._extract_action_features(action))
        
        # Next state features (if available)
        if next_state:
            features.extend(self._extract_state_features(next_state, context))
        else:
            features.extend([0.0] * 10)  # Placeholder features
        
        # Pad or truncate to fixed length
        target_length = 512
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
        
        return features
    
    def _extract_state_features(self, 
                               state: StateInfo,
                               context: Optional[Dict[str, Any]]) -> List[float]:
        """Extract features from state"""
        
        features = []
        
        # URL features (simplified)
        url = state.get('url', '')
        features.append(len(url) / 1000.0)  # Normalized URL length
        features.append(1.0 if 'admin' in url else 0.0)  # Admin page indicator
        
        # Page title features
        title = state.get('page_title', '')
        features.append(len(title) / 100.0)  # Normalized title length
        
        # Element count features
        elements = state.get('elements', [])
        features.append(len(elements) / 100.0)  # Normalized element count
        
        # Task context features
        if context:
            intent = context.get('task_intent', '')
            features.append(len(intent) / 200.0)  # Normalized intent length
        else:
            features.append(0.0)
        
        # Add more state features as needed
        features.extend([0.0] * 5)  # Placeholder for additional features
        
        return features
    
    def _extract_action_features(self, action: Action) -> List[float]:
        """Extract features from action"""
        
        features = []
        
        # Action type encoding
        action_type = action.get('action_type', '')
        action_types = ['click', 'type', 'navigate', 'wait', 'scroll']
        for at in action_types:
            features.append(1.0 if at in action_type else 0.0)
        
        # Target features
        target = action.get('target', '')
        features.append(len(target) / 100.0)  # Normalized target length
        
        # Value features
        value = action.get('value', '')
        features.append(len(value) / 100.0)  # Normalized value length
        
        # Add more action features as needed
        features.extend([0.0] * 3)  # Placeholder for additional features
        
        return features

class HybridRewardModel(RewardModel):
    """Hybrid reward model combining multiple approaches"""
    
    def __init__(self, 
                 llm_model: str = "gpt-4-1106-preview",
                 neural_model_path: Optional[str] = None,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.llm_model = LLMBasedRewardModel(llm_model)
        self.neural_model = NeuralRewardModel() if neural_model_path else None
        
        # Default weights for combining models
        self.weights = weights or {
            "llm": 0.7,
            "neural": 0.3
        }
        
        if neural_model_path:
            self.neural_model.load_model(neural_model_path)
    
    def compute_reward(self, 
                      state: StateInfo, 
                      action: Action, 
                      next_state: Optional[StateInfo] = None,
                      context: Optional[Dict[str, Any]] = None) -> RewardSignal:
        """Compute hybrid reward combining multiple models"""
        
        # Get LLM reward
        llm_reward = self.llm_model.compute_reward(state, action, next_state, context)
        
        # Get neural reward if available
        if self.neural_model:
            neural_reward = self.neural_model.compute_reward(state, action, next_state, context)
            
            # Combine rewards
            total_reward = (
                self.weights["llm"] * llm_reward.value +
                self.weights["neural"] * neural_reward.value
            )
            
            # Combine components
            combined_components = {
                "llm_reward": llm_reward.value,
                "neural_reward": neural_reward.value,
                "hybrid_reward": total_reward
            }
            
            # Average confidence
            avg_confidence = (
                self.weights["llm"] * llm_reward.confidence +
                self.weights["neural"] * neural_reward.confidence
            )
            
            return RewardSignal(
                value=total_reward,
                components=combined_components,
                confidence=avg_confidence,
                metadata={
                    "model_type": "hybrid",
                    "llm_metadata": llm_reward.metadata,
                    "neural_metadata": neural_reward.metadata
                }
            )
        else:
            # Only LLM reward available
            return llm_reward
    
    def compute_state_value(self, 
                           state: StateInfo,
                           context: Optional[Dict[str, Any]] = None) -> float:
        """Compute hybrid state value"""
        
        llm_value = self.llm_model.compute_state_value(state, context)
        
        if self.neural_model:
            neural_value = self.neural_model.compute_state_value(state, context)
            
            # Combine values
            total_value = (
                self.weights["llm"] * llm_value +
                self.weights["neural"] * neural_value
            )
            
            return total_value
        else:
            return llm_value
