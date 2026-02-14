
import time

class ViolationTracker:
    def __init__(self):
        self.violations = []

    def reset(self):
        self.violations = []

    def check_face_count(self, face_count, max_faces):
        if face_count > max_faces:
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
