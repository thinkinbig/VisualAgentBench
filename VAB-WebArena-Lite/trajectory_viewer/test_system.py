#!/usr/bin/env python3
"""
Test script to verify the trajectory viewer system.
"""

import json
import time
import requests
import subprocess
import sys
from pathlib import Path

def test_backend_api():
    """Test if the backend API is running and responding."""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend API is running")
            return True
        else:
            print(f"❌ Backend API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend API is not accessible: {e}")
        return False

def test_trajectory_endpoints():
    """Test trajectory-related endpoints."""
    try:
        # Test trajectories list
        response = requests.get("http://localhost:8000/api/trajectories", timeout=10)
        if response.status_code == 200:
            trajectories = response.json()
            print(f"✅ Found {len(trajectories)} trajectories")
            
            if trajectories:
                # Test getting specific trajectory
                run_id = trajectories[0]['run_id']
                response = requests.get(f"http://localhost:8000/api/trajectories/{run_id}", timeout=10)
                if response.status_code == 200:
                    print(f"✅ Successfully loaded trajectory {run_id}")
                    return True
                else:
                    print(f"❌ Failed to load trajectory {run_id}")
                    return False
            else:
                print("⚠️  No trajectories found (this is normal if no tasks have been run)")
                return True
        else:
            print(f"❌ Failed to get trajectories list: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error testing trajectory endpoints: {e}")
        return False

def create_sample_trajectory():
    """Create a sample trajectory file for testing."""
    trajectory_dir = Path(__file__).parent.parent / "outputs" / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data = {
        "nodes": [
            {
                "node_id": "root",
                "parent_id": None,
                "run_id": "test_run_20250108_120000_1234",
                "intent": "Test trajectory for visualization",
                "url": "http://example.com",
                "screenshot_path": None
            },
            {
                "node_id": "state_1",
                "parent_id": "root",
                "step": 1,
                "url": "http://example.com/page1",
                "screenshot_path": None,
                "candidates": ["candidate_1", "candidate_2"]
            },
            {
                "node_id": "candidate_1",
                "parent_id": "state_1",
                "thought": "I need to click on the login button",
                "action": "click [123]",
                "meaning": "Click login button",
                "status": "selected"
            },
            {
                "node_id": "candidate_2",
                "parent_id": "state_1",
                "thought": "I could also try the signup link",
                "action": "click [456]",
                "meaning": "Click signup link",
                "status": "candidate"
            }
        ]
    }
    
    sample_file = trajectory_dir / "trajectory_test_20250108_120000_final.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created sample trajectory: {sample_file}")
    return sample_file

def main():
    """Main test function."""
    print("🧪 Testing Trajectory Tree Viewer System")
    print("=" * 50)
    
    # Test 1: Check if backend is running
    print("\n1. Testing backend API...")
    if not test_backend_api():
        print("\n❌ Backend is not running. Please start it with:")
        print("   ./start_backend.sh")
        return False
    
    # Test 2: Test trajectory endpoints
    print("\n2. Testing trajectory endpoints...")
    if not test_trajectory_endpoints():
        print("\n⚠️  No trajectories found. Creating sample trajectory...")
        create_sample_trajectory()
        
        # Test again after creating sample
        print("\n3. Testing with sample trajectory...")
        if not test_trajectory_endpoints():
            print("❌ Still no trajectories found")
            return False
    
    print("\n✅ All tests passed!")
    print("\n🌐 You can now access the frontend at: http://localhost:3000")
    print("📊 Backend API is available at: http://localhost:8000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
