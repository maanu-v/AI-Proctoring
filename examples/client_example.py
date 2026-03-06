"""
Example Client for AI Proctoring FastAPI Backend
Demonstrates how to interact with the proctoring API (modular version)
"""

import requests
import cv2
import base64
import time
import json
from typing import Optional


class ProctorClient:
    """Client for interacting with the AI Proctoring API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
    
    def health_check(self):
        """Check if the API is running"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def start_session(self, student_id: str, quiz_id: str, profile_image_path: Optional[str] = None):
        """
        Start a new proctoring session
        
        Args:
            student_id: Unique student identifier
            quiz_id: Unique quiz identifier
            profile_image_path: Optional path to profile image for identity verification
        
        Returns:
            Session information
        """
        files = {}
        data = {
            "student_id": student_id,
            "quiz_id": quiz_id
        }
        
        if profile_image_path:
            with open(profile_image_path, "rb") as f:
                files["profile_image"] = f
                response = requests.post(
                    f"{self.base_url}/api/session/start",
                    data=data,
                    files=files
                )
        else:
            response = requests.post(
                f"{self.base_url}/api/session/start",
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            self.session_id = result["session_id"]
            print(f"✓ Session started: {self.session_id}")
            return result
        else:
            print(f"✗ Error starting session: {response.status_code}")
            print(response.text)
            return None
    
    def end_session(self, session_id: Optional[str] = None):
        """End a proctoring session and get final report"""
        session_id = session_id or self.session_id
        if not session_id:
            print("No active session")
            return None
        
        response = requests.post(f"{self.base_url}/api/session/end/{session_id}")
        
        if response.status_code == 200:
            print(f"✓ Session ended: {session_id}")
            return response.json()
        else:
            print(f"✗ Error ending session: {response.status_code}")
            return None
    
    def analyze_frame(self, frame, session_id: Optional[str] = None):
        """
        Analyze a single frame
        
        Args:
            frame: OpenCV image (numpy array)
            session_id: Optional session ID (uses current session if not provided)
        
        Returns:
            Analysis results
        """
        session_id = session_id or self.session_id
        if not session_id:
            print("No active session")
            return None
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = requests.post(
            f"{self.base_url}/api/analyze/frame/{session_id}",
            json={"frame_base64": frame_base64}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ Error analyzing frame: {response.status_code}")
            print(response.text)
            return None
    
    def get_session_info(self, session_id: Optional[str] = None):
        """Get information about a session"""
        session_id = session_id or self.session_id
        if not session_id:
            print("No active session")
            return None
        
        response = requests.get(f"{self.base_url}/api/session/{session_id}")
        return response.json() if response.status_code == 200 else None
    
    def get_violations(self, session_id: Optional[str] = None):
        """Get all violations for a session"""
        session_id = session_id or self.session_id
        if not session_id:
            print("No active session")
            return None
        
        response = requests.get(f"{self.base_url}/api/session/{session_id}/violations")
        return response.json() if response.status_code == 200 else None
    
    def get_settings(self, session_id: Optional[str] = None):
        """Get current settings for a session"""
        session_id = session_id or self.session_id
        if not session_id:
            print("No active session")
            return None
        
        response = requests.get(f"{self.base_url}/api/session/{session_id}/settings")
        return response.json() if response.status_code == 200 else None
    
    def update_settings(self, settings: dict, session_id: Optional[str] = None):
        """
        Update settings for a session
        
        Args:
            settings: Dictionary of settings to update
            session_id: Optional session ID
        
        Example:
            client.update_settings({
                "enable_head_pose": True,
                "enable_gaze": False
            })
        """
        session_id = session_id or self.session_id
        if not session_id:
            print("No active session")
            return None
        
        response = requests.put(
            f"{self.base_url}/api/session/{session_id}/settings",
            json=settings
        )
        
        if response.status_code == 200:
            print("✓ Settings updated")
            return response.json()
        else:
            print(f"✗ Error updating settings: {response.status_code}")
            return None
    
    def get_config(self):
        """Get global configuration"""
        response = requests.get(f"{self.base_url}/api/config")
        return response.json() if response.status_code == 200 else None
    
    def update_config(self, config: dict):
        """Update global configuration"""
        response = requests.put(f"{self.base_url}/api/config", json=config)
        
        if response.status_code == 200:
            print("✓ Configuration updated")
            return response.json()
        else:
            print(f"✗ Error updating config: {response.status_code}")
            return None


# ============================================================================
# Example Usage
# ============================================================================

def example_webcam_proctoring():
    """Example: Real-time proctoring with webcam"""
    
    print("=== AI Proctoring Client Demo ===\n")
    
    # Initialize client
    client = ProctorClient()
    
    # Health check
    print("1. Checking API health...")
    health = client.health_check()
    print(f"   API Status: {health['status']}")
    print(f"   Active Sessions: {health['active_sessions']}\n")
    
    # Start session
    print("2. Starting proctoring session...")
    session = client.start_session(
        student_id="S12345",
        quiz_id="QUIZ_2024_01",
        # profile_image_path="path/to/profile.jpg"  # Optional
    )
    print(f"   Session ID: {session['session_id']}\n")
    
    # Open webcam
    print("3. Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   ✗ Could not open webcam")
        return
    
    print("   ✓ Webcam opened")
    print("\n4. Starting frame analysis...")
    print("   Press 'q' to quit, 's' to show settings, 'v' to show violations\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every 10th frame (for demo, adjust as needed)
            if frame_count % 10 == 0:
                result = client.analyze_frame(frame)
                
                if result:
                    # Display analysis summary
                    print(f"   Frame {result['frame_count']}:")
                    print(f"   - Faces: {result['face_count']}")
                    
                    if result['warnings']:
                        print(f"   ⚠ Warnings: {', '.join(result['warnings'])}")
                    
                    if result['violations']:
                        print(f"   ⚠ NEW VIOLATIONS: {len(result['violations'])}")
                        for v in result['violations']:
                            print(f"     - {v['type']}: {v['message']}")
                    
                    # Overlay info on frame
                    cv2.putText(frame, f"Frame: {result['frame_count']}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Faces: {result['face_count']}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if result['warnings']:
                        cv2.putText(frame, "WARNING!", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("AI Proctoring - Student View", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                settings = client.get_settings()
                print("\n   Current Settings:")
                for k, v in settings.items():
                    print(f"     {k}: {v}")
                print()
            elif key == ord('v'):
                violations = client.get_violations()
                print(f"\n   Total Violations: {violations['total_violations']}")
                for v in violations['violations'][-5:]:  # Last 5
                    print(f"     - {v.get('type', 'unknown')}: {v['message']}")
                print()
    
    except KeyboardInterrupt:
        print("\n   Interrupted by user")
    
    finally:
        # Clean up
        print("\n5. Ending session...")
        report = client.end_session()
        
        if report:
            print("\n=== Final Report ===")
            print(f"Student ID: {report['student_id']}")
            print(f"Quiz ID: {report['quiz_id']}")
            print(f"Duration: {report['started_at']} to {report['ended_at']}")
            print(f"Total Frames: {report['total_frames']}")
            print(f"Total Violations: {report['total_violations']}")
            
            if report['violations']:
                print("\nViolations:")
                for v in report['violations']:
                    print(f"  - [{v.get('type', 'unknown')}] {v['message']}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Session closed")


def example_frame_by_frame():
    """Example: Sending individual frames"""
    
    client = ProctorClient()
    
    # Start session
    session = client.start_session("S67890", "QUIZ_TEST")
    
    # Capture and send a few frames
    cap = cv2.VideoCapture(0)
    
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"\nAnalyzing frame {i+1}...")
            result = client.analyze_frame(frame)
            
            if result:
                print(f"  Faces detected: {result['face_count']}")
                print(f"  Warnings: {len(result['warnings'])}")
                print(f"  Violations: {len(result['violations'])}")
        
        time.sleep(1)
    
    cap.release()
    
    # Get final report
    report = client.end_session()
    print(f"\nFinal Report: {report['total_violations']} violations detected")


def example_settings_management():
    """Example: Managing settings during a session"""
    
    client = ProctorClient()
    
    # Start session
    client.start_session("S11111", "QUIZ_SETTINGS_TEST")
    
    # Get current settings
    print("\nInitial Settings:")
    settings = client.get_settings()
    print(json.dumps(settings, indent=2))
    
    # Disable some features
    print("\nDisabling gaze and blink detection...")
    client.update_settings({
        "enable_gaze": False,
        "enable_blink": False
    })
    
    # Verify changes
    print("\nUpdated Settings:")
    settings = client.get_settings()
    print(json.dumps(settings, indent=2))
    
    # Clean up
    client.end_session()


def example_config_management():
    """Example: Managing global configuration"""
    
    client = ProctorClient()
    
    # Get current config
    print("\nCurrent Global Configuration:")
    config = client.get_config()
    print(json.dumps(config, indent=2))
    
    # Update thresholds
    print("\nUpdating violation thresholds...")
    client.update_config({
        "thresholds": {
            "no_face_persistence_time": 5.0,
            "head_pose_persistence_time": 4.0
        }
    })
    
    # Verify changes
    print("\nUpdated Configuration:")
    config = client.get_config()
    print(f"No face persistence: {config['thresholds']['no_face_persistence_time']}s")
    print(f"Head pose persistence: {config['thresholds']['head_pose_persistence_time']}s")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI Proctoring System - Client Examples")
    print("="*60)
    print("\nSelect an example to run:")
    print("1. Webcam Proctoring (Real-time)")
    print("2. Frame-by-Frame Analysis")
    print("3. Settings Management")
    print("4. Configuration Management")
    print("\nOr run the default webcam example by pressing Enter")
    print("="*60 + "\n")
    
    choice = input("Enter choice (1-4) or press Enter: ").strip()
    
    if choice == "1" or choice == "":
        example_webcam_proctoring()
    elif choice == "2":
        example_frame_by_frame()
    elif choice == "3":
        example_settings_management()
    elif choice == "4":
        example_config_management()
    else:
        print("Invalid choice. Running default example...")
        example_webcam_proctoring()
