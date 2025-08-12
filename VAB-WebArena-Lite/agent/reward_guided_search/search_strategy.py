"""Search strategies for reward-guided search in VisualAgentBench"""

import logging
import random
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import deque, defaultdict

import numpy as np

from browser_env.actions import Action
from browser_env.utils import StateInfo
from .reward_model import RewardModel, RewardSignal

logger = logging.getLogger(__name__)

@dataclass
class SearchNode:
    """Node in the search tree"""
    state: StateInfo
    action: Optional[Action] = None
    parent: Optional['SearchNode'] = None
    depth: int = 0
    path_cost: float = 0.0
    reward: float = 0.0
    value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.value > other.value  # Higher value first

@dataclass
class SearchResult:
    """Result of a search operation"""
    path: List[Tuple[Action, StateInfo]]
    total_reward: float
    final_state: StateInfo
    search_stats: Dict[str, Any]
    metadata: Dict[str, Any]

class SearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    def __init__(self, 
                 reward_model: RewardModel,
                 max_depth: int = 10,
                 max_iterations: int = 1000):
        self.reward_model = reward_model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.visited_states = set()
        
    @abstractmethod
    def search(self, 
               initial_state: StateInfo,
               goal_condition: Callable[[StateInfo], bool],
               action_generator: Callable[[StateInfo], List[Action]],
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Perform search from initial state to goal"""
        pass
    
    def _is_goal_reached(self, state: StateInfo, goal_condition: Callable[[StateInfo], bool]) -> bool:
        """Check if goal condition is met"""
        try:
            return goal_condition(state)
        except Exception as e:
            logger.warning(f"Error checking goal condition: {e}")
            return False
    
    def _get_state_key(self, state: StateInfo) -> str:
        """Generate a unique key for a state"""
        # Use URL as primary identifier, can be enhanced
        return state.get('url', str(hash(str(state)))
    
    def _is_visited(self, state: StateInfo) -> bool:
        """Check if state has been visited"""
        state_key = self._get_state_key(state)
        return state_key in self.visited_states
    
    def _mark_visited(self, state: StateInfo):
        """Mark state as visited"""
        state_key = self._get_state_key(state)
        self.visited_states.add(state_key)

class BeamSearch(SearchStrategy):
    """Beam search with reward-guided expansion"""
    
    def __init__(self, 
                 reward_model: RewardModel,
                 beam_width: int = 5,
                 max_depth: int = 10,
                 max_iterations: int = 1000):
        super().__init__(reward_model, max_depth, max_iterations)
        self.beam_width = beam_width
        
    def search(self, 
               initial_state: StateInfo,
               goal_condition: Callable[[StateInfo], bool],
               action_generator: Callable[[StateInfo], List[Action]],
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Perform beam search"""
        
        # Initialize beam with initial state
        beam = [SearchNode(
            state=initial_state,
            depth=0,
            value=self.reward_model.compute_state_value(initial_state, context)
        )]
        
        best_path = None
        best_reward = float('-inf')
        iterations = 0
        
        while beam and iterations < self.max_iterations:
            iterations += 1
            
            # Check if any node in current beam reaches goal
            for node in beam:
                if self._is_goal_reached(node.state, goal_condition):
                    if node.reward > best_reward:
                        best_reward = node.reward
                        best_path = self._extract_path(node)
            
            # Expand all nodes in current beam
            candidates = []
            for node in beam:
                if node.depth < self.max_depth:
                    candidates.extend(self._expand_node(
                        node, action_generator, context
                    ))
            
            # Select top beam_width candidates
            beam = self._select_beam_candidates(candidates)
            
            # Early stopping if no progress
            if not beam:
                break
        
        if best_path:
            return SearchResult(
                path=best_path,
                total_reward=best_reward,
                final_state=best_path[-1][1],
                search_stats={
                    "iterations": iterations,
                    "max_depth_reached": max_depth_reached,
                    "nodes_expanded": len(self.visited_states)
                },
                metadata={"search_type": "beam_search"}
            )
        else:
            # Return best path found even if goal not reached
            best_node = max(beam, key=lambda n: n.value) if beam else None
            if best_node:
                path = self._extract_path(best_node)
                return SearchResult(
                    path=path,
                    total_reward=best_node.reward,
                    final_state=best_node.state,
                    search_stats={
                        "iterations": iterations,
                        "max_depth_reached": best_node.depth,
                        "nodes_expanded": len(self.visited_states)
                    },
                    metadata={"search_type": "beam_search", "goal_not_reached": True}
                )
            else:
                # No path found
                return SearchResult(
                    path=[],
                    total_reward=0.0,
                    final_state=initial_state,
                    search_stats={"iterations": iterations, "nodes_expanded": 0},
                    metadata={"search_type": "beam_search", "no_path_found": True}
                )
    
    def _expand_node(self, 
                     node: SearchNode,
                     action_generator: Callable[[StateInfo], List[Action]],
                     context: Optional[Dict[str, Any]]) -> List[SearchNode]:
        """Expand a search node by generating actions"""
        
        if self._is_visited(node.state):
            return []
        
        self._mark_visited(node.state)
        candidates = []
        
        try:
            actions = action_generator(node.state)
            
            for action in actions:
                # Simulate next state (simplified - in practice you'd need environment simulation)
                next_state = self._simulate_action(node.state, action)
                
                if next_state:
                    # Compute reward for this action
                    reward_signal = self.reward_model.compute_reward(
                        node.state, action, next_state, context
                    )
                    
                    # Create child node
                    child = SearchNode(
                        state=next_state,
                        action=action,
                        parent=node,
                        depth=node.depth + 1,
                        path_cost=node.path_cost + (1.0 - reward_signal.value),  # Cost = 1 - reward
                        reward=node.reward + reward_signal.value,
                        value=self.reward_model.compute_state_value(next_state, context)
                    )
                    
                    candidates.append(child)
                    
        except Exception as e:
            logger.warning(f"Error expanding node: {e}")
        
        return candidates
    
    def _select_beam_candidates(self, candidates: List[SearchNode]) -> List[SearchNode]:
        """Select top beam_width candidates based on value"""
        
        if not candidates:
            return []
        
        # Sort by value (descending) and take top beam_width
        candidates.sort(key=lambda x: x.value, reverse=True)
        return candidates[:self.beam_width]
    
    def _extract_path(self, node: SearchNode) -> List[Tuple[Action, StateInfo]]:
        """Extract path from root to given node"""
        
        path = []
        current = node
        
        while current.parent:
            path.append((current.action, current.state))
            current = current.parent
        
        # Add initial state
        path.append((None, current.state))
        
        # Reverse to get correct order
        return list(reversed(path))
    
    def _simulate_action(self, state: StateInfo, action: Action) -> Optional[StateInfo]:
        """Simulate the result of an action (simplified)"""
        
        # This is a simplified simulation - in practice you'd need environment interaction
        # For now, return a modified copy of the state
        
        try:
            # Create a copy of the state
            next_state = state.copy() if hasattr(state, 'copy') else dict(state)
            
            # Simulate some state changes based on action type
            action_type = action.get('action_type', '')
            
            if 'click' in action_type:
                # Simulate clicking - might change page
                next_state['last_action'] = 'click'
                next_state['click_count'] = next_state.get('click_count', 0) + 1
                
            elif 'type' in action_type:
                # Simulate typing
                next_state['last_action'] = 'type'
                next_state['input_count'] = next_state.get('input_count', 0) + 1
                
            elif 'navigate' in action_type:
                # Simulate navigation
                next_state['last_action'] = 'navigate'
                next_state['navigation_count'] = next_state.get('navigation_count', 0) + 1
            
            return next_state
            
        except Exception as e:
            logger.warning(f"Error simulating action: {e}")
            return None

class MonteCarloSearch(SearchStrategy):
    """Monte Carlo Tree Search with reward guidance"""
    
    def __init__(self, 
                 reward_model: RewardModel,
                 exploration_constant: float = 1.414,
                 num_simulations: int = 100,
                 max_depth: int = 10,
                 max_iterations: int = 1000):
        super().__init__(reward_model, max_depth, max_iterations)
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        
    def search(self, 
               initial_state: StateInfo,
               goal_condition: Callable[[StateInfo], bool],
               action_generator: Callable[[StateInfo], List[Action]],
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Perform Monte Carlo Tree Search"""
        
        # Initialize root node
        root = SearchNode(
            state=initial_state,
            value=self.reward_model.compute_state_value(initial_state, context)
        )
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, action_generator, goal_condition, context)
        
        # Find best path
        best_path = self._find_best_path(root)
        
        return SearchResult(
            path=best_path,
            total_reward=root.reward,
            final_state=best_path[-1][1] if best_path else initial_state,
            search_stats={
                "simulations": self.num_simulations,
                "nodes_expanded": len(self.visited_states)
            },
            metadata={"search_type": "monte_carlo_search"}
        )
    
    def _simulate(self, 
                  node: SearchNode,
                  action_generator: Callable[[StateInfo], List[Action]],
                  goal_condition: Callable[[StateInfo], bool],
                  context: Optional[Dict[str, Any]]):
        """Run a single simulation from the given node"""
        
        current = node
        depth = 0
        
        while depth < self.max_depth:
            # Check if goal reached
            if self._is_goal_reached(current.state, goal_condition):
                break
            
            # Generate actions
            try:
                actions = action_generator(current.state)
                if not actions:
                    break
                
                # Select action using UCB1
                action = self._select_action_ucb(current, actions)
                
                # Simulate action
                next_state = self._simulate_action(current.state, action)
                if not next_state:
                    break
                
                # Create child node if it doesn't exist
                child = self._get_or_create_child(current, action, next_state, context)
                
                current = child
                depth += 1
                
            except Exception as e:
                logger.warning(f"Error in simulation: {e}")
                break
        
        # Backpropagate results
        self._backpropagate(current, context)
    
    def _select_action_ucb(self, node: SearchNode, actions: List[Action]) -> Action:
        """Select action using UCB1 formula"""
        
        if not hasattr(node, 'children'):
            node.children = {}
        
        # If some actions haven't been tried, select randomly
        untried_actions = [a for a in actions if a not in node.children]
        if untried_actions:
            return random.choice(untried_actions)
        
        # Use UCB1 to select from tried actions
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            if action in node.children:
                child = node.children[action]
                ucb_value = (child.reward / max(child.visit_count, 1) + 
                           self.exploration_constant * np.sqrt(np.log(node.visit_count) / max(child.visit_count, 1)))
                
                if ucb_value > best_value:
                    best_value = ucb_value
                    best_action = action
        
        return best_action
    
    def _get_or_create_child(self, 
                             parent: SearchNode,
                             action: Action,
                             next_state: StateInfo,
                             context: Optional[Dict[str, Any]]) -> SearchNode:
        """Get existing child or create new one"""
        
        if not hasattr(parent, 'children'):
            parent.children = {}
        
        if action not in parent.children:
            # Create new child
            reward_signal = self.reward_model.compute_reward(
                parent.state, action, next_state, context
            )
            
            child = SearchNode(
                state=next_state,
                action=action,
                parent=parent,
                depth=parent.depth + 1,
                reward=parent.reward + reward_signal.value,
                value=self.reward_model.compute_state_value(next_state, context),
                visit_count=0
            )
            
            parent.children[action] = child
        else:
            child = parent.children[action]
        
        return child
    
    def _backpropagate(self, node: SearchNode, context: Optional[Dict[str, Any]]):
        """Backpropagate simulation results up the tree"""
        
        current = node
        while current:
            if not hasattr(current, 'visit_count'):
                current.visit_count = 0
            
            current.visit_count += 1
            
            # Update reward (could be more sophisticated)
            if hasattr(current, 'children'):
                total_child_reward = sum(child.reward for child in current.children.values())
                current.reward = total_child_reward / len(current.children)
            
            current = current.parent
    
    def _find_best_path(self, root: SearchNode) -> List[Tuple[Action, StateInfo]]:
        """Find the best path from root based on visit counts"""
        
        path = []
        current = root
        
        while hasattr(current, 'children') and current.children:
            # Select child with highest visit count
            best_child = max(current.children.values(), key=lambda c: c.visit_count)
            path.append((best_child.action, best_child.state))
            current = best_child
        
        return path
    
    def _simulate_action(self, state: StateInfo, action: Action) -> Optional[StateInfo]:
        """Simulate action result (same as in BeamSearch)"""
        
        try:
            next_state = state.copy() if hasattr(state, 'copy') else dict(state)
            action_type = action.get('action_type', '')
            
            if 'click' in action_type:
                next_state['last_action'] = 'click'
                next_state['click_count'] = next_state.get('click_count', 0) + 1
            elif 'type' in action_type:
                next_state['last_action'] = 'type'
                next_state['input_count'] = next_state.get('input_count', 0) + 1
            elif 'navigate' in action_type:
                next_state['last_action'] = 'navigate'
                next_state['navigation_count'] = next_state.get('navigation_count', 0) + 1
            
            return next_state
            
        except Exception as e:
            logger.warning(f"Error simulating action: {e}")
            return None

class AStarSearch(SearchStrategy):
    """A* search with reward-based heuristic"""
    
    def __init__(self, 
                 reward_model: RewardModel,
                 max_depth: int = 10,
                 max_iterations: int = 1000):
        super().__init__(reward_model, max_depth, max_iterations)
        
    def search(self, 
               initial_state: StateInfo,
               goal_condition: Callable[[StateInfo], bool],
               action_generator: Callable[[StateInfo], List[Action]],
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Perform A* search"""
        
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        # Create initial node
        initial_node = SearchNode(
            state=initial_state,
            value=self.reward_model.compute_state_value(initial_state, context)
        )
        
        heapq.heappush(open_set, initial_node)
        
        iterations = 0
        best_path = None
        best_reward = float('-inf')
        
        while open_set and iterations < self.max_iterations:
            iterations += 1
            
            # Get node with lowest f_score
            current = heapq.heappop(open_set)
            
            # Check if goal reached
            if self._is_goal_reached(current.state, goal_condition):
                if current.reward > best_reward:
                    best_reward = current.reward
                    best_path = self._extract_path(current)
                break
            
            # Add to closed set
            closed_set.add(self._get_state_key(current.state))
            
            # Expand current node
            for action in action_generator(current.state):
                next_state = self._simulate_action(current.state, action)
                if not next_state:
                    continue
                
                # Skip if already visited
                if self._get_state_key(next_state) in closed_set:
                    continue
                
                # Compute reward
                reward_signal = self.reward_model.compute_reward(
                    current.state, action, next_state, context
                )
                
                # Create child node
                child = SearchNode(
                    state=next_state,
                    action=action,
                    parent=current,
                    depth=current.depth + 1,
                    path_cost=current.path_cost + (1.0 - reward_signal.value),
                    reward=current.reward + reward_signal.value,
                    value=self.reward_model.compute_state_value(next_state, context)
                )
                
                # Add to open set
                heapq.heappush(open_set, child)
        
        if best_path:
            return SearchResult(
                path=best_path,
                total_reward=best_reward,
                final_state=best_path[-1][1],
                search_stats={
                    "iterations": iterations,
                    "nodes_expanded": len(closed_set)
                },
                metadata={"search_type": "a_star_search"}
            )
        else:
            # Return best path found
            if open_set:
                best_node = max(open_set, key=lambda n: n.reward)
                path = self._extract_path(best_node)
                return SearchResult(
                    path=path,
                    total_reward=best_node.reward,
                    final_state=best_node.state,
                    search_stats={
                        "iterations": iterations,
                        "nodes_expanded": len(closed_set)
                    },
                    metadata={"search_type": "a_star_search", "goal_not_reached": True}
                )
            else:
                return SearchResult(
                    path=[],
                    total_reward=0.0,
                    final_state=initial_state,
                    search_stats={"iterations": iterations, "nodes_expanded": 0},
                    metadata={"search_type": "a_star_search", "no_path_found": True}
                )
    
    def _extract_path(self, node: SearchNode) -> List[Tuple[Action, StateInfo]]:
        """Extract path from root to given node"""
        
        path = []
        current = node
        
        while current.parent:
            path.append((current.action, current.state))
            current = current.parent
        
        path.append((None, current.state))
        return list(reversed(path))
    
    def _simulate_action(self, state: StateInfo, action: Action) -> Optional[StateInfo]:
        """Simulate action result (same as other strategies)"""
        
        try:
            next_state = state.copy() if hasattr(state, 'copy') else dict(state)
            action_type = action.get('action_type', '')
            
            if 'click' in action_type:
                next_state['last_action'] = 'click'
                next_state['click_count'] = next_state.get('click_count', 0) + 1
            elif 'type' in action_type:
                next_state['last_action'] = 'type'
                next_state['input_count'] = next_state.get('input_count', 0) + 1
            elif 'navigate' in action_type:
                next_state['last_action'] = 'navigate'
                next_state['navigation_count'] = next_state.get('navigation_count', 0) + 1
            
            return next_state
            
        except Exception as e:
            logger.warning(f"Error simulating action: {e}")
            return None
