"""Pydantic models for Reward Guided Search module"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime

from .enums import SearchType, ActionType, RewardComponent, SafetyLevel, PerformanceMetric

class RewardSignal(BaseModel):
    """Structured reward signal"""
    value: float = Field(..., ge=0.0, le=1.0, description="Main reward value (0-1)")
    components: Dict[str, float] = Field(default_factory=dict, description="Breakdown of reward components")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the reward")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional information")
    timestamp: datetime = Field(default_factory=datetime.now, description="When reward was computed")
    
    @validator('value')
    def validate_value(cls, v):
        """Ensure value is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Reward value must be between 0.0 and 1.0')
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    def get_component_value(self, component: RewardComponent) -> float:
        """Get value for a specific reward component"""
        return self.components.get(component.value, 0.0)
    
    def set_component_value(self, component: RewardComponent, value: float):
        """Set value for a specific reward component"""
        self.components[component.value] = max(0.0, min(1.0, value))
    
    def get_safety_level(self) -> SafetyLevel:
        """Get safety level based on safety component"""
        safety_value = self.get_component_value(RewardComponent.SAFETY)
        return SafetyLevel.from_numeric(safety_value)

class SearchConfig(BaseModel):
    """Configuration for reward guided search"""
    search_type: SearchType = Field(default=SearchType.BEAM_SEARCH, description="Search strategy type")
    beam_width: int = Field(default=5, ge=1, le=20, description="Beam width for beam search")
    max_depth: int = Field(default=8, ge=1, le=20, description="Maximum search depth")
    max_iterations: int = Field(default=500, ge=10, le=10000, description="Maximum search iterations")
    exploration_constant: float = Field(default=1.414, ge=0.1, le=10.0, description="Exploration constant for MCTS")
    num_simulations: int = Field(default=100, ge=10, le=1000, description="Number of simulations for MCTS")
    use_hybrid_reward: bool = Field(default=True, description="Whether to use hybrid reward model")
    llm_model: str = Field(default="gpt-4-1106-preview", description="LLM model for reward computation")
    neural_model_path: Optional[str] = Field(default=None, description="Path to neural reward model")
    reward_weights: Optional[Dict[str, float]] = Field(default=None, description="Weights for reward components")
    cache_ttl: int = Field(default=60, ge=10, le=3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=100, ge=10, le=1000, description="Maximum cache size")
    
    @validator('reward_weights')
    def validate_reward_weights(cls, v):
        """Validate reward weights sum to reasonable values"""
        if v is not None:
            total_weight = sum(v.values())
            if total_weight < 0.1 or total_weight > 10.0:
                raise ValueError('Reward weights should sum to a reasonable value')
        return v
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True

class SearchNode(BaseModel):
    """Node in the search tree"""
    state: Dict[str, Any] = Field(..., description="State information")
    action: Optional[Dict[str, Any]] = Field(default=None, description="Action taken to reach this state")
    parent: Optional['SearchNode'] = Field(default=None, description="Parent node")
    depth: int = Field(default=0, ge=0, description="Depth in search tree")
    path_cost: float = Field(default=0.0, ge=0.0, description="Cost to reach this node")
    reward: float = Field(default=0.0, description="Accumulated reward")
    value: float = Field(default=0.0, description="Node value for search")
    visit_count: int = Field(default=0, ge=0, description="Number of visits (for MCTS)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="When node was created")
    
    @validator('depth')
    def validate_depth(cls, v):
        """Ensure depth is non-negative"""
        if v < 0:
            raise ValueError('Depth must be non-negative')
        return v
    
    @validator('path_cost')
    def validate_path_cost(cls, v):
        """Ensure path cost is non-negative"""
        if v < 0.0:
            raise ValueError('Path cost must be non-negative')
        return v
    
    def is_root(self) -> bool:
        """Check if this is the root node"""
        return self.parent is None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return self.depth >= 0  # Simplified check
    
    def get_path_to_root(self) -> List['SearchNode']:
        """Get path from this node to root"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_ancestors(self) -> List['SearchNode']:
        """Get all ancestor nodes"""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors

class SearchResult(BaseModel):
    """Result of a search operation"""
    path: List[Dict[str, Any]] = Field(..., description="Path of actions and states")
    total_reward: float = Field(..., description="Total reward along the path")
    final_state: Dict[str, Any] = Field(..., description="Final state reached")
    search_stats: Dict[str, Any] = Field(..., description="Search statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    search_time: float = Field(..., ge=0.0, description="Time taken for search in seconds")
    success: bool = Field(..., description="Whether search was successful")
    created_at: datetime = Field(default_factory=datetime.now, description="When result was created")
    
    @validator('total_reward')
    def validate_total_reward(cls, v):
        """Ensure total reward is reasonable"""
        if v < -100.0 or v > 100.0:
            raise ValueError('Total reward should be in reasonable range')
        return v
    
    @validator('search_time')
    def validate_search_time(cls, v):
        """Ensure search time is non-negative"""
        if v < 0.0:
            raise ValueError('Search time must be non-negative')
        return v
    
    def get_path_length(self) -> int:
        """Get length of the path"""
        return len(self.path)
    
    def get_average_reward_per_step(self) -> float:
        """Get average reward per step"""
        if self.get_path_length() == 0:
            return 0.0
        return self.total_reward / self.get_path_length()
    
    def get_search_efficiency(self) -> float:
        """Get search efficiency (reward per time)"""
        if self.search_time == 0.0:
            return 0.0
        return self.total_reward / self.search_time
    
    def is_optimal(self, threshold: float = 0.9) -> bool:
        """Check if result is near optimal"""
        return self.total_reward >= threshold

class PerformanceMetrics(BaseModel):
    """Performance metrics for search agent"""
    total_searches: int = Field(default=0, ge=0, description="Total number of searches performed")
    cache_hits: int = Field(default=0, ge=0, description="Number of cache hits")
    successful_searches: int = Field(default=0, ge=0, description="Number of successful searches")
    average_search_time: float = Field(default=0.0, ge=0.0, description="Average search time in seconds")
    total_search_time: float = Field(default=0.0, ge=0.0, description="Total search time in seconds")
    cache_size: int = Field(default=0, ge=0, description="Current cache size")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate of searches")
    average_reward: float = Field(default=0.0, description="Average reward per search")
    last_updated: datetime = Field(default_factory=datetime.now, description="When metrics were last updated")
    
    @validator('success_rate')
    def validate_success_rate(cls, v):
        """Ensure success rate is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Success rate must be between 0.0 and 1.0')
        return v
    
    def update_success_rate(self):
        """Update success rate based on current counts"""
        if self.total_searches > 0:
            self.success_rate = self.successful_searches / self.total_searches
        else:
            self.success_rate = 0.0
    
    def add_search_result(self, search_time: float, reward: float, success: bool):
        """Add a search result to update metrics"""
        self.total_searches += 1
        self.total_search_time += search_time
        self.average_search_time = self.total_search_time / self.total_searches
        
        if success:
            self.successful_searches += 1
        
        # Update success rate
        self.update_success_rate()
        
        # Update average reward
        if self.total_searches == 1:
            self.average_reward = reward
        else:
            self.average_reward = (self.average_reward * (self.total_searches - 1) + reward) / self.total_searches
        
        self.last_updated = datetime.now()
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        if self.total_searches == 0:
            return 0.0
        return self.cache_hits / self.total_searches
    
    def reset(self):
        """Reset all metrics"""
        self.total_searches = 0
        self.cache_hits = 0
        self.successful_searches = 0
        self.average_search_time = 0.0
        self.total_search_time = 0.0
        self.success_rate = 0.0
        self.average_reward = 0.0
        self.last_updated = datetime.now()

class ActionMetadata(BaseModel):
    """Metadata for actions"""
    element_type: Optional[str] = Field(default=None, description="Type of element targeted")
    element_id: Optional[str] = Field(default=None, description="ID of element")
    element_text: Optional[str] = Field(default=None, description="Text content of element")
    element_classes: List[str] = Field(default_factory=list, description="CSS classes of element")
    element_attributes: Dict[str, str] = Field(default_factory=dict, description="HTML attributes of element")
    page_url: Optional[str] = Field(default=None, description="URL of page when action was taken")
    timestamp: datetime = Field(default_factory=datetime.now, description="When action was taken")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in action")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    def get_element_info(self) -> Dict[str, Any]:
        """Get comprehensive element information"""
        return {
            "type": self.element_type,
            "id": self.element_id,
            "text": self.element_text,
            "classes": self.element_classes,
            "attributes": self.element_attributes
        }
    
    def is_safe_element(self) -> bool:
        """Check if element is safe to interact with"""
        dangerous_classes = ['delete', 'remove', 'danger', 'warning']
        dangerous_attributes = ['data-dangerous', 'data-confirm']
        
        # Check classes
        if any(dc in ' '.join(self.element_classes).lower() for dc in dangerous_classes):
            return False
        
        # Check attributes
        if any(da in ' '.join(self.element_attributes.values()).lower() for da in dangerous_attributes):
            return False
        
        return True

class SearchContext(BaseModel):
    """Context for search operations"""
    task_intent: str = Field(..., description="Intent of the task")
    task_type: Optional[str] = Field(default=None, description="Type of task")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    safety_constraints: List[str] = Field(default_factory=list, description="Safety constraints")
    efficiency_goals: List[str] = Field(default_factory=list, description="Efficiency goals")
    max_steps: Optional[int] = Field(default=None, ge=1, description="Maximum steps allowed")
    timeout: Optional[float] = Field(default=None, ge=1.0, description="Timeout in seconds")
    priority: str = Field(default="normal", description="Task priority")
    created_at: datetime = Field(default_factory=datetime.now, description="When context was created")
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority value"""
        valid_priorities = ['low', 'normal', 'high', 'urgent']
        if v not in valid_priorities:
            raise ValueError(f'Priority must be one of: {valid_priorities}')
        return v
    
    def is_high_priority(self) -> bool:
        """Check if task is high priority"""
        return self.priority in ['high', 'urgent']
    
    def has_safety_constraints(self) -> bool:
        """Check if task has safety constraints"""
        return len(self.safety_constraints) > 0
    
    def get_constraint_summary(self) -> str:
        """Get summary of constraints"""
        constraints = []
        if self.safety_constraints:
            constraints.append(f"Safety: {', '.join(self.safety_constraints)}")
        if self.efficiency_goals:
            constraints.append(f"Efficiency: {', '.join(self.efficiency_goals)}")
        if self.max_steps:
            constraints.append(f"Max steps: {self.max_steps}")
        if self.timeout:
            constraints.append(f"Timeout: {self.timeout}s")
        
        return "; ".join(constraints) if constraints else "No constraints"

# Update forward references
SearchNode.update_forward_refs()
