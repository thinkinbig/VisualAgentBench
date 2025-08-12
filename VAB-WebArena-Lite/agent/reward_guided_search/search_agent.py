"""Reward Guided Search Agent for VisualAgentBench"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from browser_env.actions import Action
from browser_env.utils import StateInfo, Observation
from browser_env import Trajectory

from agent.agent import Agent
from .models import SearchConfig, RewardSignal, SearchNode, SearchResult, PerformanceMetrics
from .enums import SearchType, ActionType, TaskType
from .reward_model import RewardModel, LLMBasedRewardModel, HybridRewardModel
from .search_strategy import SearchStrategy, BeamSearch, MonteCarloSearch, AStarSearch

logger = logging.getLogger(__name__)

class RewardGuidedSearchAgent(Agent):
    """Agent that uses reward-guided search to find optimal action sequences"""
    
    def __init__(self, config: SearchConfig):
        super().__init__()
        self.config = config
        
        # Initialize reward model
        if config.use_hybrid_reward:
            self.reward_model = HybridRewardModel(
                llm_model=config.llm_model,
                neural_model_path=config.neural_model_path,
                weights=config.reward_weights
            )
        else:
            self.reward_model = LLMBasedRewardModel(config.llm_model)
        
        # Initialize search strategy
        self.search_strategy = self._create_search_strategy()
        
        # Cache for search results
        self.search_cache = {}
        self.last_search_time = 0
        self.cache_ttl = 60  # Cache TTL in seconds
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "average_search_time": 0.0,
            "successful_searches": 0
        }
    
    def _create_search_strategy(self) -> SearchStrategy:
        """Create search strategy based on configuration"""
        
        if self.config.search_type == SearchType.BEAM_SEARCH:
            return BeamSearch(
                reward_model=self.reward_model,
                beam_width=self.config.beam_width,
                max_depth=self.config.max_depth,
                max_iterations=self.config.max_iterations
            )
        elif self.config.search_type == SearchType.MONTE_CARLO:
            return MonteCarloSearch(
                reward_model=self.reward_model,
                exploration_constant=self.config.exploration_constant,
                num_simulations=self.config.num_simulations,
                max_depth=self.config.max_depth,
                max_iterations=self.config.max_iterations
            )
        elif self.config.search_type == SearchType.A_STAR:
            return AStarSearch(
                reward_model=self.reward_model,
                max_depth=self.config.max_depth,
                max_iterations=self.config.max_iterations
            )
        else:
            logger.warning(f"Unknown search type: {self.config.search_type}, using beam search")
            return BeamSearch(
                reward_model=self.reward_model,
                beam_width=self.config.beam_width,
                max_depth=self.config.max_depth,
                max_iterations=self.config.max_iterations
            )
    
    def next_action(self, 
                   trajectory: Trajectory, 
                   intent: str, 
                   meta_data: Any) -> Action:
        """Generate next action using reward-guided search"""
        
        start_time = time.time()
        
        # Get current state from trajectory
        current_state = self._extract_current_state(trajectory)
        if not current_state:
            logger.error("Could not extract current state from trajectory")
            return self._create_fallback_action()
        
        # Check cache for existing search result
        cache_key = self._generate_cache_key(current_state, intent)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            self.search_stats["cache_hits"] += 1
            logger.info("Using cached search result")
            return self._extract_next_action_from_path(cached_result.path, trajectory)
        
        # Perform search
        try:
            search_result = self._perform_search(current_state, intent, meta_data)
            
            # Cache the result
            self._cache_search_result(cache_key, search_result)
            
            # Update statistics
            search_time = time.time() - start_time
            self.search_stats["total_searches"] += 1
            self.search_stats["average_search_time"] = (
                (self.search_stats["average_search_time"] * (self.search_stats["total_searches"] - 1) + search_time) /
                self.search_stats["total_searches"]
            )
            
            if search_result.path:
                self.search_stats["successful_searches"] += 1
            
            # Extract and return next action
            return self._extract_next_action_from_path(search_result.path, trajectory)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._create_fallback_action()
    
    def _perform_search(self, 
                       current_state: StateInfo, 
                       intent: str, 
                       meta_data: Any) -> Any:
        """Perform reward-guided search"""
        
        # Define goal condition based on intent
        goal_condition = self._create_goal_condition(intent, meta_data)
        
        # Create action generator
        action_generator = self._create_action_generator(current_state)
        
        # Create context for reward computation
        context = {
            "task_intent": intent,
            "meta_data": meta_data,
            "current_state": current_state
        }
        
        # Perform search
        logger.info(f"Starting {self.config.search_type} search with depth {self.config.max_depth}")
        search_result = self.search_strategy.search(
            initial_state=current_state,
            goal_condition=goal_condition,
            action_generator=action_generator,
            context=context
        )
        
        logger.info(f"Search completed: {len(search_result.path)} actions found, "
                   f"total reward: {search_result.total_reward:.3f}")
        
        return search_result
    
    def _create_goal_condition(self, intent: str, meta_data: Any) -> Callable[[StateInfo], bool]:
        """Create goal condition function based on intent"""
        
        def goal_condition(state: StateInfo) -> bool:
            """Check if goal is reached"""
            
            # Extract relevant information from state
            url = state.get('url', '')
            page_title = state.get('page_title', '')
            elements = state.get('elements', [])
            
            # Simple goal conditions based on intent
            if 'order' in intent.lower() and 'view' in intent.lower():
                # Check if we're on an order view page
                return ('order' in url.lower() or 'order' in page_title.lower()) and \
                       any('order' in str(elem).lower() for elem in elements)
            
            elif 'admin' in intent.lower():
                # Check if we're on an admin page
                return 'admin' in url.lower() or 'admin' in page_title.lower()
            
            elif 'login' in intent.lower():
                # Check if we're logged in (simplified)
                return 'logout' in str(elements).lower() or 'profile' in str(elements).lower()
            
            else:
                # Generic goal: check if page content matches intent
                return any(keyword in str(state).lower() for keyword in intent.lower().split())
        
        return goal_condition
    
    def _create_action_generator(self, current_state: StateInfo) -> Callable[[StateInfo], List[Action]]:
        """Create action generator function"""
        
        def action_generator(state: StateInfo) -> List[Action]:
            """Generate possible actions for a given state"""
            
            actions = []
            elements = state.get('elements', [])
            
            # Generate click actions for clickable elements
            for element in elements:
                if isinstance(element, dict):
                    element_type = element.get('type', '')
                    element_text = element.get('text', '')
                    element_id = element.get('id', '')
                    
                    # Click actions
                    if element_type in ['button', 'link', 'input']:
                        actions.append({
                            'action_type': 'click',
                            'target': element_id or element_text,
                            'value': '',
                            'metadata': {'element_type': element_type}
                        })
                    
                    # Type actions for input fields
                    if element_type == 'input' and 'text' in element_text.lower():
                        actions.append({
                            'action_type': 'type',
                            'target': element_id or element_text,
                            'value': 'sample_text',
                            'metadata': {'element_type': 'input'}
                        })
            
            # Add navigation actions
            actions.extend([
                {
                    'action_type': 'navigate',
                    'target': 'back',
                    'value': '',
                    'metadata': {'action': 'navigation'}
                },
                {
                    'action_type': 'navigate',
                    'target': 'refresh',
                    'value': '',
                    'metadata': {'action': 'navigation'}
                }
            ])
            
            # Add wait action
            actions.append({
                'action_type': 'wait',
                'target': 'page_load',
                'value': '2',
                'metadata': {'action': 'wait'}
            })
            
            return actions
        
        return action_generator
    
    def _extract_current_state(self, trajectory: Trajectory) -> Optional[StateInfo]:
        """Extract current state from trajectory"""
        
        if not trajectory:
            return None
        
        # Get the last state info from trajectory
        for item in reversed(trajectory):
            if hasattr(item, 'get') and callable(item.get):
                # This looks like a state
                return item
            elif hasattr(item, 'state'):
                # This has a state attribute
                return item.state
        
        return None
    
    def _generate_cache_key(self, state: StateInfo, intent: str) -> str:
        """Generate cache key for search results"""
        
        # Create a simple hash based on state URL and intent
        state_key = state.get('url', '') + state.get('page_title', '')
        return f"{hash(state_key + intent)}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached search result"""
        
        if cache_key in self.search_cache:
            cached_item = self.search_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['result']
            else:
                # Remove expired cache entry
                del self.search_cache[cache_key]
        
        return None
    
    def _cache_search_result(self, cache_key: str, result: Any):
        """Cache search result"""
        
        self.search_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Limit cache size
        if len(self.search_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.search_cache.keys(), 
                           key=lambda k: self.search_cache[k]['timestamp'])
            del self.search_cache[oldest_key]
    
    def _extract_next_action_from_path(self, path: List, trajectory: Trajectory) -> Action:
        """Extract the next action from search path"""
        
        if not path or len(path) < 2:
            return self._create_fallback_action()
        
        # Find the next action that hasn't been executed yet
        executed_actions = len(trajectory)
        
        if executed_actions < len(path) - 1:
            # Return the next action in the path
            next_action_data = path[executed_actions + 1][0]  # (action, state) tuple
            
            if next_action_data:
                return self._convert_to_action(next_action_data)
        
        # If we've executed all actions in the path, return a fallback
        return self._create_fallback_action()
    
    def _convert_to_action(self, action_data: Dict) -> Action:
        """Convert action data to Action object"""
        
        # This is a simplified conversion - you might need to adapt based on your Action class
        try:
            return Action(
                action_type=action_data.get('action_type', 'click'),
                target=action_data.get('target', ''),
                value=action_data.get('value', ''),
                metadata=action_data.get('metadata', {})
            )
        except Exception as e:
            logger.warning(f"Error converting action data: {e}")
            return self._create_fallback_action()
    
    def _create_fallback_action(self) -> Action:
        """Create a fallback action when search fails"""
        
        try:
            return Action(
                action_type='wait',
                target='fallback',
                value='1',
                metadata={'reason': 'search_failed'}
            )
        except Exception as e:
            logger.error(f"Error creating fallback action: {e}")
            # Return a minimal action
            return {'action_type': 'wait', 'target': 'fallback', 'value': '1'}
    
    def reset(self, test_config_file: str) -> None:
        """Reset the agent for a new test"""
        
        # Clear cache and statistics
        self.search_cache.clear()
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "average_search_time": 0.0,
            "successful_searches": 0
        }
        
        # Reset search strategy
        self.search_strategy.visited_states.clear()
        
        logger.info("Agent reset completed")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        
        return {
            "search_stats": self.search_stats,
            "cache_size": len(self.search_cache),
            "search_config": {
                "search_type": self.config.search_type,
                "max_depth": self.config.max_depth,
                "max_iterations": self.config.max_iterations
            }
        }
    
    def update_search_config(self, new_config: Dict[str, Any]):
        """Update search configuration dynamically"""
        
        # Update config attributes
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Recreate search strategy if search type changed
        if 'search_type' in new_config:
            self.search_strategy = self._create_search_strategy()
        
        logger.info(f"Search configuration updated: {new_config}")

class AdaptiveRewardGuidedSearchAgent(RewardGuidedSearchAgent):
    """Agent that adapts search strategy based on performance"""
    
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.performance_history = []
        self.adaptation_threshold = 0.6  # Threshold for strategy adaptation
        
    def next_action(self, 
                   trajectory: Trajectory, 
                   intent: str, 
                   meta_data: Any) -> Action:
        """Generate next action with adaptive search strategy"""
        
        # Perform search with current strategy
        action = super().next_action(trajectory, intent, meta_data)
        
        # Update performance history
        self._update_performance_history(trajectory, intent)
        
        # Adapt strategy if needed
        self._adapt_search_strategy()
        
        return action
    
    def _update_performance_history(self, trajectory: Trajectory, intent: str):
        """Update performance history for strategy adaptation"""
        
        if len(trajectory) > 0:
            # Simple performance metric: task completion progress
            current_state = self._extract_current_state(trajectory)
            if current_state:
                progress = self._estimate_task_progress(current_state, intent)
                self.performance_history.append(progress)
                
                # Keep only recent history
                if len(self.performance_history) > 10:
                    self.performance_history.pop(0)
    
    def _estimate_task_progress(self, state: StateInfo, intent: str) -> float:
        """Estimate task completion progress"""
        
        # Simple heuristic based on URL and page content
        url = state.get('url', '')
        page_title = state.get('page_title', '')
        
        if 'order' in intent.lower() and 'view' in intent.lower():
            if 'order' in url.lower() or 'order' in page_title.lower():
                return 0.8  # High progress
            elif 'admin' in url.lower():
                return 0.4  # Medium progress
            else:
                return 0.1  # Low progress
        
        return 0.5  # Default progress
    
    def _adapt_search_strategy(self):
        """Adapt search strategy based on performance"""
        
        if len(self.performance_history) < 5:
            return
        
        # Calculate average performance
        avg_performance = sum(self.performance_history) / len(self.performance_history)
        
        if avg_performance < self.adaptation_threshold:
            # Performance is poor, try different strategy
            current_type = self.config.search_type
            
            if current_type == SearchType.BEAM_SEARCH:
                new_type = SearchType.MONTE_CARLO
                logger.info("Switching from beam search to Monte Carlo search due to poor performance")
            elif current_type == SearchType.MONTE_CARLO:
                new_type = SearchType.A_STAR
                logger.info("Switching from Monte Carlo to A* search due to poor performance")
            else:
                new_type = SearchType.BEAM_SEARCH
                logger.info("Switching to beam search due to poor performance")
            
            # Update configuration
            self.update_search_config({"search_type": new_type})
            
            # Reset performance history
            self.performance_history.clear()
