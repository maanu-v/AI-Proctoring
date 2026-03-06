#!/usr/bin/env python3
"""
Quick test script to verify FastAPI backend is working
"""

import sys
import time
import requests
from termcolor import colored

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if API is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(colored("✓ Health check passed", "green"))
            print(f"  Service: {data['service']}")
            print(f"  Status: {data['status']}")
            print(f"  Active sessions: {data['active_sessions']}")
            return True
        else:
            print(colored("✗ Health check failed", "red"))
            print(f"  Status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(colored("✗ Cannot connect to API", "red"))
        print(f"  Make sure the server is running on {BASE_URL}")
        print(f"  Start it with: python src/web/fastapi_app.py")
        return False
    except Exception as e:
        print(colored(f"✗ Error: {e}", "red"))
        return False


def test_get_config():
    """Test getting configuration"""
    try:
        response = requests.get(f"{BASE_URL}/api/config", timeout=5)
        if response.status_code == 200:
            print(colored("✓ Get config passed", "green"))
            data = response.json()
            print(f"  Head pose yaw threshold: {data['head_pose']['yaw_threshold']}")
            print(f"  Max faces: {data['thresholds']['max_num_faces']}")
            return True
        else:
            print(colored("✗ Get config failed", "red"))
            return False
    except Exception as e:
        print(colored(f"✗ Error: {e}", "red"))
        return False


def test_session_flow():
    """Test basic session flow"""
    try:
        # Start session
        print("\nTesting session flow...")
        response = requests.post(
            f"{BASE_URL}/api/session/start",
            data={"student_id": "TEST_USER", "quiz_id": "TEST_QUIZ"}
        )
        
        if response.status_code != 200:
            print(colored("✗ Failed to start session", "red"))
            return False
        
        session_data = response.json()
        session_id = session_data["session_id"]
        print(colored("✓ Session started", "green"))
        print(f"  Session ID: {session_id}")
        
        # Get session info
        response = requests.get(f"{BASE_URL}/api/session/{session_id}")
        if response.status_code == 200:
            print(colored("✓ Get session info passed", "green"))
        else:
            print(colored("✗ Get session info failed", "red"))
            return False
        
        # Get settings
        response = requests.get(f"{BASE_URL}/api/session/{session_id}/settings")
        if response.status_code == 200:
            print(colored("✓ Get settings passed", "green"))
            settings = response.json()
            print(f"  Face detection enabled: {settings['enable_face_detection']}")
        else:
            print(colored("✗ Get settings failed", "red"))
            return False
        
        # Update settings
        response = requests.put(
            f"{BASE_URL}/api/session/{session_id}/settings",
            json={"enable_gaze": False}
        )
        if response.status_code == 200:
            print(colored("✓ Update settings passed", "green"))
        else:
            print(colored("✗ Update settings failed", "red"))
            return False
        
        # Get violations
        response = requests.get(f"{BASE_URL}/api/session/{session_id}/violations")
        if response.status_code == 200:
            print(colored("✓ Get violations passed", "green"))
            data = response.json()
            print(f"  Total violations: {data['total_violations']}")
        else:
            print(colored("✗ Get violations failed", "red"))
            return False
        
        # End session
        response = requests.post(f"{BASE_URL}/api/session/end/{session_id}")
        if response.status_code == 200:
            print(colored("✓ Session ended", "green"))
            report = response.json()
            print(f"  Total frames: {report['total_frames']}")
            print(f"  Total violations: {report['total_violations']}")
            return True
        else:
            print(colored("✗ Failed to end session", "red"))
            return False
            
    except Exception as e:
        print(colored(f"✗ Error: {e}", "red"))
        return False


def main():
    print("="*60)
    print("AI Proctoring FastAPI - Quick Test")
    print("="*60)
    print()
    
    # Test 1: Health check
    print("Test 1: Health Check")
    if not test_health_check():
        print("\n" + colored("Tests aborted - API not accessible", "red"))
        sys.exit(1)
    
    print()
    
    # Test 2: Get config
    print("Test 2: Get Configuration")
    test_get_config()
    
    print()
    
    # Test 3: Session flow
    print("Test 3: Session Flow")
    test_session_flow()
    
    print()
    print("="*60)
    print(colored("All tests completed!", "green"))
    print("="*60)
    print()
    print("Next steps:")
    print("  1. Run the full example: python examples/client_example.py")
    print("  2. Open interactive docs: http://localhost:8000/docs")
    print("  3. Read the documentation: docs/API_DOCUMENTATION.md")
    print()


if __name__ == "__main__":
    try:
        import termcolor
    except ImportError:
        print("Installing termcolor for better output...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "termcolor"])
        import termcolor
    
    main()
