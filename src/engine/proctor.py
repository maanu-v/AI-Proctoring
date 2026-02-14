
import time

class ViolationTracker:
    def __init__(self):
        self.violations = []
        self.no_face_start_time = None
        self.face_count_start_time = None

    def reset(self):
        self.violations = []
        self.no_face_start_time = None
        self.face_count_start_time = None

    def check_face_count(self, face_count, max_faces, persistence_time=0):
        if face_count > max_faces:
             if self.face_count_start_time is None:
                 self.face_count_start_time = time.time()
                 if persistence_time > 0:
                     return False, False
             
             elapsed = time.time() - self.face_count_start_time
             if elapsed < persistence_time:
                 return False, False
             
             # Active!
             # Limit log frequency or just log
             # For simplicity, we log if it's a new violation or just return status
             # But the app logic expects (active, triggered)
             # Let's say we log it.
             msg = f"Multiple faces detected: {face_count}"
             # Avoid spamming logs?
             # The original app likely handled debouncing or the user doesn't care about spam yet.
             # I'll add a simple check to not spam identical messages within 1 sec if needed, 
             # but strictly following "log violation" logic.
             
             # Actually, if we look at app.py:
             # hp_active, hp_triggered = ...
             # if hp_triggered ... st.toast
             
             # So this method should return (is_active, is_newly_triggered)
             
             is_active = True
             is_triggered = False
             
             # simple dedup: if last violation was recent and same, don't trigger
             if not self.violations or (self.violations[-1]['message'] != msg) or (time.time() - self.violations[-1]['timestamp'] > 2.0):
                  self.log_violation(msg)
                  is_triggered = True
             
             return is_active, is_triggered
        else:
             self.face_count_start_time = None
             return False, False

    def check_no_face(self, face_count, enabled, persistence_time):
        if not enabled:
            self.no_face_start_time = None
            return False, False
            
        if face_count == 0:
            if self.no_face_start_time is None:
                self.no_face_start_time = time.time()
                # Not active yet, accumulating time
                return False, False
            
            elapsed = time.time() - self.no_face_start_time
            if elapsed >= persistence_time:
                # Active!
                msg = f"No face detected for {persistence_time} seconds!"
                
                is_active = True
                is_triggered = False
                
                # Debounce log
                if not self.violations or (self.violations[-1]['message'] != msg) or (time.time() - self.violations[-1]['timestamp'] > 5.0):
                    self.log_violation(msg)
                    is_triggered = True
                    
                return is_active, is_triggered
            else:
                 # Still waiting
                 return False, False
        else:
            # Face found, reset timer
            self.no_face_start_time = None
            return False, False

    def log_violation(self, message):
        self.violations.append({
            "timestamp": time.time(),
            "message": message
        })

    def get_logs(self):
        return self.violations

    def get_violation_count(self):
        return len(self.violations)
