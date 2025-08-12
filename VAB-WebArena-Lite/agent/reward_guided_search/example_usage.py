"""Example usage of Reward Guided Search module"""

import logging
from agent.reward_guided_search import (
    RewardGuidedSearchAgent,
    AdaptiveRewardGuidedSearchAgent,
    SearchConfig,
    create_task_specific_reward
)
from agent.reward_guided_search.enums import SearchType, TaskType
from agent.reward_guided_search.reward_functions import (
    TaskCompletionReward,
    EfficiencyReward,
    SafetyFirstReward,
    CompositeReward
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage example of RewardGuidedSearchAgent"""
    
    print("=== Basic Usage Example ===")
    
    # Create configuration
    config = SearchConfig(
        search_type=SearchType.BEAM_SEARCH,
        beam_width=5,
        max_depth=6,
        max_iterations=200,
        use_hybrid_reward=False  # Use only LLM-based reward for simplicity
    )
    
    # Create agent
    agent = RewardGuidedSearchAgent(config)
    
    print(f"Created agent with {config.search_type.value} strategy")
    print(f"Beam width: {config.beam_width}, Max depth: {config.max_depth}")
    
    return agent

def example_adaptive_agent():
    """Example of adaptive agent that changes strategy based on performance"""
    
    print("\n=== Adaptive Agent Example ===")
    
    config = SearchConfig(
        search_type=SearchType.BEAM_SEARCH,
        beam_width=3,
        max_depth=5,
        max_iterations=100
    )
    
    agent = AdaptiveRewardGuidedSearchAgent(config)
    
    print("Created adaptive agent that will switch strategies based on performance")
    print(f"Initial strategy: {config.search_type.value}")
    
    return agent

def example_custom_reward_functions():
    """Example of creating and using custom reward functions"""
    
    print("\n=== Custom Reward Functions Example ===")
    
    # Create task-specific reward function
    order_reward = create_task_specific_reward("order_management")
    print("Created order management reward function")
    
    # Create efficiency-focused reward
    efficiency_reward = EfficiencyReward(
        max_steps=8,
        time_penalty=0.2,
        weights={
            "efficiency": 0.6,
            "task_progress": 0.3,
            "safety": 0.1
        }
    )
    print("Created efficiency-focused reward function")
    
    # Create safety-first reward
    safety_reward = SafetyFirstReward(
        dangerous_keywords=["delete", "remove", "drop", "trash", "cancel"],
        weights={
            "safety": 0.7,
            "task_progress": 0.2,
            "efficiency": 0.1
        }
    )
    print("Created safety-first reward function")
    
    # Create composite reward function
    composite_reward = CompositeReward([
        (order_reward, 0.4),
        (efficiency_reward, 0.4),
        (safety_reward, 0.2)
    ])
    print("Created composite reward function combining all three")
    
    return {
        "order_reward": order_reward,
        "efficiency_reward": efficiency_reward,
        "safety_reward": safety_reward,
        "composite_reward": composite_reward
    }

def example_search_strategy_comparison():
    """Compare different search strategies"""
    
    print("\n=== Search Strategy Comparison ===")
    
    strategies = [SearchType.BEAM_SEARCH, SearchType.MONTE_CARLO, SearchType.A_STAR]
    
    agents = {}
    
    for strategy in strategies:
        config = SearchConfig(
            search_type=strategy,
            beam_width=5 if strategy == SearchType.BEAM_SEARCH else 3,
            max_depth=6,
            max_iterations=200,
            num_simulations=50 if strategy == SearchType.MONTE_CARLO else 100
        )
        
        agent = RewardGuidedSearchAgent(config)
        agents[strategy.value] = agent
        
        print(f"Created {strategy.value} agent with config: {config}")
    
    return agents

def example_hybrid_reward_model():
    """Example of using hybrid reward model (LLM + Neural)"""
    
    print("\n=== Hybrid Reward Model Example ===")
    
    config = SearchConfig(
        search_type=SearchType.BEAM_SEARCH,
        beam_width=4,
        max_depth=5,
        use_hybrid_reward=True,
        llm_model="gpt-4-1106-preview",
        neural_model_path="./models/neural_reward_model.pt",  # Path to your trained model
        reward_weights={
            "llm": 0.7,
            "neural": 0.3
        }
    )
    
    try:
        agent = RewardGuidedSearchAgent(config)
        print("Created agent with hybrid reward model")
        print(f"LLM weight: {config.reward_weights['llm']}")
        print(f"Neural weight: {config.reward_weights['neural']}")
        return agent
    except Exception as e:
        print(f"Could not create hybrid agent (neural model not available): {e}")
        # Fallback to LLM-only
        config.use_hybrid_reward = False
        agent = RewardGuidedSearchAgent(config)
        print("Created agent with LLM-only reward model")
        return agent

def example_performance_monitoring():
    """Example of monitoring agent performance"""
    
    print("\n=== Performance Monitoring Example ===")
    
    config = SearchConfig(
        search_type=SearchType.BEAM_SEARCH,
        beam_width=5,
        max_depth=6
    )
    
    agent = RewardGuidedSearchAgent(config)
    
    # Simulate some usage
    print("Simulating agent usage...")
    
    # Get initial statistics
    initial_stats = agent.get_search_statistics()
    print(f"Initial stats: {initial_stats}")
    
    # Update configuration dynamically
    print("\nUpdating search configuration...")
    agent.update_search_config({
        "max_depth": 8,
        "beam_width": 7
    })
    
    # Get updated statistics
    updated_stats = agent.get_search_statistics()
    print(f"Updated stats: {updated_stats}")
    
    return agent

def run_all_examples():
    """Run all examples"""
    
    print("🚀 Reward Guided Search Module Examples")
    print("=" * 50)
    
    # Run all examples
    agents = {}
    
    try:
        agents["basic"] = example_basic_usage()
        agents["adaptive"] = example_adaptive_agent()
        agents["reward_functions"] = example_custom_reward_functions()
        agents["strategies"] = example_search_strategy_comparison()
        agents["hybrid"] = example_hybrid_reward_model()
        agents["monitoring"] = example_performance_monitoring()
        
        print("\n✅ All examples completed successfully!")
        print(f"Created {len(agents)} different agent configurations")
        
        return agents
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return {}

if __name__ == "__main__":
    # Run all examples
    agents = run_all_examples()
    
    if agents:
        print("\n🎯 Example Summary:")
        for name, agent in agents.items():
            if hasattr(agent, 'config'):
                print(f"  {name}: {agent.config.search_type.value} strategy")
            else:
                print(f"  {name}: {type(agent).__name__}")
    
    print("\n📚 Next steps:")
    print("1. Use these agents in your VisualAgentBench environment")
    print("2. Customize reward functions for your specific tasks")
    print("3. Experiment with different search strategies")
    print("4. Train neural reward models for hybrid approaches")
