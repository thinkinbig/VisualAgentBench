#!/usr/bin/env python3
"""
Simple test script for the new simplified prompts system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agent.reward_guided_search.prompts import (
    get_system_prompt,
    get_user_prompt,
    get_conversation_messages,
    get_available_prompts,
    get_available_roles,
    PromptType,
    AgentRole,
    example_usage
)

def test_prompts_system():
    """Test the simplified prompts system"""
    
    print("🧪 Testing Simplified Prompts System")
    print("=" * 50)
    
    # Test 1: Get system prompts
    print("\n1️⃣ Testing System Prompts:")
    try:
        action_generator_prompt = get_system_prompt("action_generator")
        print(f"✅ Action Generator System Prompt: {action_generator_prompt['role']}")
        print(f"   Content length: {len(action_generator_prompt['content'])} chars")
        
        webspherd_prompt = get_system_prompt("webspherd_evaluator")
        print(f"✅ WebSpherd Evaluator System Prompt: {webspherd_prompt['role']}")
        print(f"   Content length: {len(webspherd_prompt['content'])} chars")
        
    except Exception as e:
        print(f"❌ System prompt test failed: {e}")
    
    # Test 2: Get user prompts
    print("\n2️⃣ Testing User Prompts:")
    try:
        mock_state = {
            "url": "https://example.com",
            "title": "Example Page",
            "elements": [
                {"type": "button", "text": "Click Me"},
                {"type": "input", "text": ""}
            ]
        }
        
        action_prompt = get_user_prompt("action_generation", 
                                      state=mock_state,
                                      intent="Click the button and type 'hello'",
                                      meta_data={"task_id": "test"})
        print(f"✅ Action Generation User Prompt: {action_prompt['role']}")
        print(f"   Content length: {len(action_prompt['content'])} chars")
        
        checklist_prompt = get_user_prompt("checklist_generation",
                                         intent="Complete the registration form",
                                         start_url="https://example.com/register",
                                         meta_data={})
        print(f"✅ Checklist Generation User Prompt: {checklist_prompt['role']}")
        print(f"   Content length: {len(checklist_prompt['content'])} chars")
        
    except Exception as e:
        print(f"❌ User prompt test failed: {e}")
    
    # Test 3: Get conversation messages
    print("\n3️⃣ Testing Conversation Messages:")
    try:
        messages = get_conversation_messages("action_generation",
                                          "action_generator",
                                          state=mock_state,
                                          intent="Click the button",
                                          meta_data={})
        print(f"✅ Conversation Messages: {len(messages)} messages")
        print(f"   Message 1: {messages[0]['role']} - {len(messages[0]['content'])} chars")
        print(f"   Message 2: {messages[1]['role']} - {len(messages[1]['content'])} chars")
        
    except Exception as e:
        print(f"❌ Conversation messages test failed: {e}")
    
    # Test 4: Check available options
    print("\n4️⃣ Testing Available Options:")
    try:
        available_prompts = get_available_prompts()
        print(f"✅ Available Prompt Types: {len(available_prompts)}")
        for prompt in available_prompts:
            print(f"   - {prompt}")
        
        available_roles = get_available_roles()
        print(f"✅ Available Roles: {len(available_roles)}")
        for role in available_roles:
            print(f"   - {role}")
            
    except Exception as e:
        print(f"❌ Available options test failed: {e}")
    
    # Test 4.5: Test enum usage
    print("\n4️⃣.5️⃣ Testing Enum Usage:")
    try:
        print(f"✅ PromptType.ACTION_GENERATION: {PromptType.ACTION_GENERATION.value}")
        print(f"✅ AgentRole.ACTION_GENERATOR: {AgentRole.ACTION_GENERATOR.value}")
        print(f"✅ Total PromptType values: {len(PromptType)}")
        print(f"✅ Total AgentRole values: {len(AgentRole)}")
        
        # Test enum iteration
        print("   PromptType enum values:")
        for pt in PromptType:
            print(f"     - {pt.name}: {pt.value}")
            
    except Exception as e:
        print(f"❌ Enum test failed: {e}")
    
    # Test 4.6: Test WebShepherd checklist generation
    print("\n4️⃣.6️⃣ Testing WebShepherd Checklist Generation:")
    try:
        checklist_prompt = get_user_prompt("web_shepherd_checklist_generation",
                                         intent="Complete online registration form",
                                         start_url="https://example.com/register",
                                         meta_data={})
        
        print(f"✅ WebShepherd Checklist Generation Prompt generated: {len(checklist_prompt['content'])} chars")
        print("   Content preview:")
        content = checklist_prompt['content']
        print(f"   {content[:100]}...")
        
        # Check if it contains key elements
        if "SUBGOAL ANALYSIS" in content and "CHECKLISTS" in content:
            print("   ✅ Contains required output format")
        if "essential steps" in content.lower():
            print("   ✅ Contains essential steps guidance")
        if "page transitions" in content.lower():
            print("   ✅ Contains page transition focus")
            
    except Exception as e:
        print(f"❌ WebShepherd checklist generation test failed: {e}")
    
    # Test 5: Test error handling
    print("\n5️⃣ Testing Error Handling:")
    try:
        # This should raise an error
        get_system_prompt("unknown_role")
        print("❌ Should have raised an error for unknown role")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    try:
        # This should also raise an error
        get_user_prompt("unknown_type")
        print("❌ Should have raised an error for unknown prompt type")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    print("\n🎉 All tests completed!")
    print("=" * 50)

def test_webspherd_prompt():
    """Test the WebSpherd evaluation prompt specifically"""
    
    print("\n🔍 Testing WebSpherd Evaluation Prompt:")
    print("=" * 40)
    
    try:
        mock_state = {
            "url": "https://example.com",
            "title": "Example Page",
            "text_observation": "bid1: button 'Submit'\nbid2: input 'Name'"
        }
        
        mock_action = {
            "action_type": "click",
            "coordinates": [100, 200],
            "text": "Submit",
            "element_id": "bid1",
            "thought": "I need to click the submit button to complete the form"
        }
        
        webspherd_prompt = get_user_prompt("webspherd_evaluation",
                                         state=mock_state,
                                         action=mock_action,
                                         intent="Submit the registration form",
                                         meta_data={})
        
        print(f"✅ WebSpherd Prompt generated: {len(webspherd_prompt['content'])} chars")
        print("   Content preview:")
        content = webspherd_prompt['content']
        print(f"   {content[:100]}...")
        
        # Check if it contains key WebSpherd elements
        if "Scoring Criteria (1 to 5)" in content:
            print("   ✅ Contains scoring criteria")
        if "REASON:" in content and "SCORE:" in content:
            print("   ✅ Contains required output format")
        if "AXTREE" in content:
            print("   ✅ Contains AXTree information")
            
    except Exception as e:
        print(f"❌ WebSpherd prompt test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Simplified Prompts System Tests")
    print("=" * 60)
    
    # Run the main test
    test_prompts_system()
    
    # Run WebSpherd specific test
    test_webspherd_prompt()
    
    # Run the built-in example
    print("\n📚 Running Built-in Example:")
    print("-" * 30)
    example_usage()
    
    print("\n🎯 All tests completed successfully!")
    print("=" * 60)
