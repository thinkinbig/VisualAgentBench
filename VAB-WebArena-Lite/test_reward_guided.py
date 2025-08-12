#!/usr/bin/env python3
"""Simple test script for the reward-guided agent."""

import json
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent_import():
    """Test if the agent can be imported correctly."""
    try:
        from agent import RewardGuidedAgent, construct_agent
        print("✓ Successfully imported RewardGuidedAgent")
        return True
    except ImportError as e:
        print(f"✗ Failed to import RewardGuidedAgent: {e}")
        return False

def test_agent_construction():
    """Test if the agent can be constructed."""
    try:
        from agent import construct_agent
        
        # Create a mock args object
        class MockArgs:
            def __init__(self):
                self.agent_type = "reward_guided"
                self.instruction_path = "configs/reward_guided_agent.yaml"
                self.action_set_tag = "webrl_id"
                self.provider = "openai"
                self.model = "gpt-4o-mini"
                self.mode = "chat"
                self.temperature = 1.0
                self.top_p = 0.9
                self.context_length = 4096
                self.max_tokens = 512
                self.stop_token = None
                self.max_obs_length = 2048
                self.max_retry = 3
                self.planner_ip = ""
        
        args = MockArgs()
        
        # Test agent construction
        agent = construct_agent(args)
        print("✓ Successfully constructed agent")
        
        # Check if it's the right type
        if hasattr(agent, '__class__') and 'RewardGuidedAgent' in str(agent.__class__):
            print("✓ Agent is of correct type")
        else:
            print(f"✗ Agent is of wrong type: {type(agent)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to construct agent: {e}")
        return False

def test_config_files():
    """Test if configuration files exist."""
    config_files = [
        "configs/reward_guided_agent.yaml",
        "test_configs/shopping_task.json"
    ]
    
    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ Config file exists: {config_file}")
        else:
            print(f"✗ Config file missing: {config_file}")
            all_exist = False
    
    return all_exist

def test_run_script():
    """Test if the run script exists and is executable."""
    run_script = "run_reward_guided.py"
    
    if os.path.exists(run_script):
        print(f"✓ Run script exists: {run_script}")
        
        # Check if it's executable
        if os.access(run_script, os.X_OK):
            print(f"✓ Run script is executable")
        else:
            print(f"⚠ Run script exists but is not executable")
        
        return True
    else:
        print(f"✗ Run script missing: {run_script}")
        return False

def main():
    """Run all tests."""
    print("Testing Reward-Guided Agent Implementation")
    print("=" * 50)
    
    tests = [
        ("Agent Import", test_agent_import),
        ("Agent Construction", test_agent_construction),
        ("Config Files", test_config_files),
        ("Run Script", test_run_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The reward-guided agent is ready to use.")
        print("\nTo run the agent:")
        print("python run_reward_guided.py --test_config_file test_configs/shopping_task.json --output_response")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
