#!/usr/bin/env python3
"""
Test script for WebSpherd-inspired Reward Guided Search
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agent.reward_guided_search import RewardGuidedSearchAgent, SearchConfig
from agent.reward_guided_search.enums import SearchType
from browser_env.actions import Action
from browser_env.utils import StateInfo
from browser_env import Trajectory

def create_mock_state():
    """Create a mock state for testing"""
    return StateInfo(
        url="https://example.com",
        title="Example Page",
        elements=[
            {"id": "button1", "text": "Click Me", "type": "button"},
            {"id": "input1", "text": "", "type": "input"},
            {"id": "link1", "text": "Go to page", "type": "link"}
        ]
    )

def create_mock_trajectory():
    """Create a mock trajectory for testing"""
    trajectory = Trajectory()
    trajectory.states = [create_mock_state()]
    return trajectory

def test_webspherd_implementation():
    """Test the WebSpherd-inspired implementation"""
    
    print("🧪 Testing WebSpherd-inspired Reward Guided Search")
    print("=" * 60)
    
    # Create configuration
    config = SearchConfig(
        search_type=SearchType.BEAM_SEARCH,
        beam_width=5,
        max_depth=8,
        max_iterations=100,
        llm_model="gpt-5-nano",
        use_hybrid_reward=False
    )
    
    print(f"✅ Configuration created: {config}")
    
    try:
        # Create agent
        agent = RewardGuidedSearchAgent(config)
        print("✅ Agent created successfully")
        
        # Test action generation
        trajectory = create_mock_trajectory()
        intent = "Click the button and then type 'hello' in the input field"
        meta_data = {"task_id": "test_001"}
        
        print(f"\n🎯 Testing with intent: {intent}")
        print(f"📊 Current state: {trajectory.states[-1].url}")
        
        # Generate next action
        action = agent.next_action(trajectory, intent, meta_data)
        
        if action:
            print(f"✅ Action generated: {action}")
            print(f"   - Type: {action.action_type}")
            print(f"   - Coordinates: {action.coordinates}")
            print(f"   - Text: {action.text}")
        else:
            print("❌ No action generated")
        
        # Test statistics
        stats = agent.get_search_statistics()
        print(f"\n📊 Search Statistics:")
        print(f"   - Total searches: {stats['total_searches']}")
        print(f"   - Successful searches: {stats['successful_searches']}")
        print(f"   - Average search time: {stats['average_search_time']:.3f}s")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_action_candidate_generation():
    """Test the action candidate generation specifically"""
    
    print("\n🔍 Testing Action Candidate Generation")
    print("=" * 50)
    
    config = SearchConfig(
        search_type=SearchType.BEAM_SEARCH,
        beam_width=5,
        max_depth=8,
        llm_model="gpt-4o-mini"
    )
    
    try:
        agent = RewardGuidedSearchAgent(config)
        trajectory = create_mock_trajectory()
        intent = "Navigate to the search page"
        meta_data = {}
        
        # Test candidate generation
        candidates = agent._generate_action_candidates(
            trajectory.states[-1], intent, meta_data
        )
        
        print(f"✅ Generated {len(candidates)} action candidates")
        for i, (action, frequency) in enumerate(candidates):
            print(f"   {i+1}. Action: {action.action_type}, Frequency: {frequency}")
            
    except Exception as e:
        print(f"❌ Action candidate generation test failed: {e}")

def test_reward_scoring():
    """Test the reward scoring mechanism"""
    
    print("\n⭐ Testing Reward Scoring")
    print("=" * 40)
    
    config = SearchConfig(
        search_type=SearchType.BEAM_SEARCH,
        beam_width=5,
        max_depth=8,
        llm_model="gpt-5-nano"
    )
    
    try:
        agent = RewardGuidedSearchAgent(config)
        trajectory = create_mock_trajectory()
        intent = "Click the submit button"
        meta_data = {}
        
        # Create some mock actions
        actions = [
            (Action("click", (100, 200), "Submit"), 3),
            (Action("click", (150, 250), "Submit"), 2),
            (Action("wait", (0, 0), "1"), 1)
        ]
        
        # Test scoring
        scored = agent._score_action_candidates(
            actions, trajectory.states[-1], intent, meta_data
        )
        
        print(f"✅ Scored {len(scored)} actions")
        for action, score, frequency in scored:
            print(f"   - Action: {action.action_type}, Score: {score:.3f}, Frequency: {frequency}")
            
    except Exception as e:
        print(f"❌ Reward scoring test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting WebSpherd Implementation Tests")
    print("=" * 60)
    
    # Run tests
    test_webspherd_implementation()
    test_action_candidate_generation()
    test_reward_scoring()
    
    print("\n🎯 All tests completed!")
    print("=" * 60)
