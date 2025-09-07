#!/usr/bin/env python3
"""
Test script to verify that extract_obs_nodes_info function works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))

from runtime_manager import extract_obs_nodes_info

def test_extract_obs_nodes_info():
    """Test the extract_obs_nodes_info function with mock data."""
    
    # Test case 1: Normal case with obs_nodes_info
    info_with_nodes = {
        "observation_metadata": {
            "text": {
                "obs_nodes_info": {
                    "189": {
                        "backend_id": "189",
                        "union_bound": [100, 200, 150, 220],
                        "text": "[189] link 'My Account'"
                    },
                    "190": {
                        "backend_id": "190", 
                        "union_bound": [200, 200, 250, 220],
                        "text": "[190] button 'Login'"
                    }
                }
            }
        }
    }
    
    result = extract_obs_nodes_info(info_with_nodes)
    print("Test 1 - Normal case:")
    print(f"Result: {result}")
    print(f"Expected keys: {list(result.keys())}")
    assert "189" in result
    assert "190" in result
    assert result["189"]["text"] == "[189] link 'My Account'"
    print("✓ Test 1 passed\n")
    
    # Test case 2: Empty obs_nodes_info
    info_empty = {
        "observation_metadata": {
            "text": {
                "obs_nodes_info": {}
            }
        }
    }
    
    result = extract_obs_nodes_info(info_empty)
    print("Test 2 - Empty obs_nodes_info:")
    print(f"Result: {result}")
    assert result == {}
    print("✓ Test 2 passed\n")
    
    # Test case 3: Missing observation_metadata
    info_missing_metadata = {
        "page": "some_page_info"
    }
    
    result = extract_obs_nodes_info(info_missing_metadata)
    print("Test 3 - Missing observation_metadata:")
    print(f"Result: {result}")
    assert result == {}
    print("✓ Test 3 passed\n")
    
    # Test case 4: Missing text metadata
    info_missing_text = {
        "observation_metadata": {
            "image": {
                "some_image_data": "value"
            }
        }
    }
    
    result = extract_obs_nodes_info(info_missing_text)
    print("Test 4 - Missing text metadata:")
    print(f"Result: {result}")
    assert result == {}
    print("✓ Test 4 passed\n")
    
    # Test case 5: Invalid data structure
    info_invalid = {
        "observation_metadata": {
            "text": "not_a_dict"
        }
    }
    
    result = extract_obs_nodes_info(info_invalid)
    print("Test 5 - Invalid data structure:")
    print(f"Result: {result}")
    assert result == {}
    print("✓ Test 5 passed\n")
    
    print("All tests passed! ✓")

if __name__ == "__main__":
    test_extract_obs_nodes_info()
