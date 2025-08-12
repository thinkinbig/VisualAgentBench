"""Enums for Reward Guided Search module"""

from enum import Enum, auto
from typing import List

class SearchType(Enum):
    """Search strategy types"""
    BEAM_SEARCH = "beam_search"
    MONTE_CARLO = "monte_carlo"
    A_STAR = "a_star"
    GREEDY = "greedy"
    RANDOM = "random"
    
    @classmethod
    def get_default(cls) -> 'SearchType':
        """Get default search type"""
        return cls.BEAM_SEARCH
    
    @classmethod
    def get_all_types(cls) -> List['SearchType']:
        """Get all search types"""
        return list(cls)
    
    @classmethod
    def from_string(cls, value: str) -> 'SearchType':
        """Create SearchType from string"""
        try:
            return cls(value)
        except ValueError:
            return cls.get_default()

class ActionType(Enum):
    """Action types for web automation"""
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    WAIT = "wait"
    SCROLL = "scroll"
    HOVER = "hover"
    DRAG_DROP = "drag_drop"
    KEY_PRESS = "key_press"
    MOUSE_MOVE = "mouse_move"
    REFRESH = "refresh"
    BACK = "back"
    FORWARD = "forward"
    
    @classmethod
    def get_click_actions(cls) -> List['ActionType']:
        """Get all click-related actions"""
        return [cls.CLICK, cls.HOVER, cls.DRAG_DROP]
    
    @classmethod
    def get_navigation_actions(cls) -> List['ActionType']:
        """Get all navigation actions"""
        return [cls.NAVIGATE, cls.BACK, cls.FORWARD, cls.REFRESH]
    
    @classmethod
    def get_input_actions(cls) -> List['ActionType']:
        """Get all input actions"""
        return [cls.TYPE, cls.KEY_PRESS]
    
    @classmethod
    def get_wait_actions(cls) -> List['ActionType']:
        """Get all wait actions"""
        return [cls.WAIT, cls.SCROLL]
    
    @classmethod
    def is_safe_action(cls, action_type: 'ActionType') -> bool:
        """Check if action type is generally safe"""
        return action_type in [cls.WAIT, cls.SCROLL, cls.REFRESH, cls.BACK, cls.FORWARD]
    
    @classmethod
    def is_potentially_dangerous(cls, action_type: 'ActionType') -> bool:
        """Check if action type could be dangerous"""
        return action_type in [cls.CLICK, cls.TYPE, cls.DRAG_DROP]

class RewardComponent(Enum):
    """Components of reward functions"""
    TASK_PROGRESS = "task_progress"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    USER_EXPERIENCE = "user_experience"
    EXPLORATION = "exploration"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    SPEED = "speed"
    RELIABILITY = "reliability"
    
    @classmethod
    def get_core_components(cls) -> List['RewardComponent']:
        """Get core reward components"""
        return [
            cls.TASK_PROGRESS,
            cls.EFFICIENCY,
            cls.SAFETY,
            cls.USER_EXPERIENCE
        ]
    
    @classmethod
    def get_advanced_components(cls) -> List['RewardComponent']:
        """Get advanced reward components"""
        return [
            cls.EXPLORATION,
            cls.CONSISTENCY,
            cls.ACCURACY,
            cls.SPEED,
            cls.RELIABILITY
        ]
    
    @classmethod
    def get_default_weights(cls) -> dict:
        """Get default weights for reward components"""
        return {
            cls.TASK_PROGRESS: 0.4,
            cls.EFFICIENCY: 0.2,
            cls.SAFETY: 0.2,
            cls.USER_EXPERIENCE: 0.1,
            cls.EXPLORATION: 0.05,
            cls.CONSISTENCY: 0.05
        }

class ModelType(Enum):
    """Types of reward models"""
    LLM_BASED = "llm_based"
    NEURAL = "neural"
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    
    @classmethod
    def get_available_types(cls) -> List['ModelType']:
        """Get available model types"""
        return list(cls)
    
    @classmethod
    def is_hybrid(cls, model_type: 'ModelType') -> bool:
        """Check if model type is hybrid"""
        return model_type == cls.HYBRID

class TaskType(Enum):
    """Types of web automation tasks"""
    ORDER_MANAGEMENT = "order_management"
    USER_AUTHENTICATION = "user_authentication"
    DATA_ENTRY = "data_entry"
    NAVIGATION = "navigation"
    FORM_SUBMISSION = "form_submission"
    SEARCH = "search"
    CONTENT_VIEWING = "content_viewing"
    ADMIN_OPERATIONS = "admin_operations"
    E_COMMERCE = "e_commerce"
    SOCIAL_MEDIA = "social_media"
    
    @classmethod
    def get_e_commerce_tasks(cls) -> List['TaskType']:
        """Get e-commerce related tasks"""
        return [cls.ORDER_MANAGEMENT, cls.E_COMMERCE, cls.FORM_SUBMISSION]
    
    @classmethod
    def get_admin_tasks(cls) -> List['TaskType']:
        """Get admin related tasks"""
        return [cls.ADMIN_OPERATIONS, cls.USER_AUTHENTICATION, cls.DATA_ENTRY]
    
    @classmethod
    def get_navigation_tasks(cls) -> List['TaskType']:
        """Get navigation related tasks"""
        return [cls.NAVIGATION, cls.SEARCH, cls.CONTENT_VIEWING]

class SafetyLevel(Enum):
    """Safety levels for operations"""
    VERY_SAFE = "very_safe"
    SAFE = "safe"
    MODERATE = "moderate"
    RISKY = "risky"
    DANGEROUS = "dangerous"
    
    @classmethod
    def get_safe_levels(cls) -> List['SafetyLevel']:
        """Get safe safety levels"""
        return [cls.VERY_SAFE, cls.SAFE]
    
    @classmethod
    def get_risky_levels(cls) -> List['SafetyLevel']:
        """Get risky safety levels"""
        return [cls.RISKY, cls.DANGEROUS]
    
    @classmethod
    def from_numeric(cls, value: float) -> 'SafetyLevel':
        """Convert numeric safety value to SafetyLevel"""
        if value >= 0.8:
            return cls.VERY_SAFE
        elif value >= 0.6:
            return cls.SAFE
        elif value >= 0.4:
            return cls.MODERATE
        elif value >= 0.2:
            return cls.RISKY
        else:
            return cls.DANGEROUS

class PerformanceMetric(Enum):
    """Performance metrics for evaluation"""
    SUCCESS_RATE = "success_rate"
    COMPLETION_TIME = "completion_time"
    STEP_COUNT = "step_count"
    ERROR_COUNT = "error_count"
    REWARD_ACCUMULATION = "reward_accumulation"
    EFFICIENCY_SCORE = "efficiency_score"
    SAFETY_SCORE = "safety_score"
    USER_SATISFACTION = "user_satisfaction"
    
    @classmethod
    def get_primary_metrics(cls) -> List['PerformanceMetric']:
        """Get primary performance metrics"""
        return [
            cls.SUCCESS_RATE,
            cls.COMPLETION_TIME,
            cls.STEP_COUNT
        ]
    
    @classmethod
    def get_quality_metrics(cls) -> List['PerformanceMetric']:
        """Get quality-related metrics"""
        return [
            cls.ERROR_COUNT,
            cls.SAFETY_SCORE,
            cls.USER_SATISFACTION
        ]
