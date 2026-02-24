
import time

class ViolationTracker:
    def __init__(self):
        self.violations = []
        self.no_face_start_time = None
        self.face_count_start_time = None
        self.head_pose_start_time = None
        self.last_pose_direction = None

    def reset(self):
        self.violations = []
        self.no_face_start_time = None
        self.face_count_start_time = None
        self.head_pose_start_time = None
        self.last_pose_direction = None

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
             is_active = True
             is_triggered = False
             
             # Log violation with type 'multiple_faces'
             if self.log_violation(msg, violation_type='multiple_faces'):
                  is_triggered = True
             
             return is_active, is_triggered
        else:
             self.face_count_start_time = None
             return False, False

    def check_no_face(self, face_count, enabled, persistence_time):
        if not enabled:
            self.no_face_start_time = None
            return False, False, ""
            
        if face_count == 0:
            if self.no_face_start_time is None:
                self.no_face_start_time = time.time()
                # Not active yet, accumulating time
                return False, False, ""
            
            elapsed = time.time() - self.no_face_start_time
            if elapsed >= persistence_time:
                # Active!
                msg = f"No face detected for {int(elapsed)} seconds."
                
                is_active = True
                is_triggered = False
                
                # Log violation with type 'no_face'
                # The message changes (time increments), so we want to UPDATE the last log if it was also 'no_face'
                if self.log_violation(msg, violation_type='no_face'):
                    is_triggered = True
                    
                return is_active, is_triggered, msg
            else:
                 # Still waiting
                 return False, False, ""
        else:
            # Face found, reset timer
            self.no_face_start_time = None
            return False, False, ""

    def check_head_pose(self, direction, persistence_time):
        """
        Check if the user is looking away for longer than persistence_time.
        Returns: (is_active, is_triggered, duration_message)
        """
        if direction == "Forward":
            self.head_pose_start_time = None
            self.last_pose_direction = None
            return False, False, ""

        # Looking away
        if self.head_pose_start_time is None:
            # New potential violation
            self.head_pose_start_time = time.time()
            self.last_pose_direction = direction
            return False, False, ""
        
        # Check if direction changed (e.g. from Left to Right)
        # If changed, reset timer? Or keep accumulating if still not forward?
        # Requirement: "looking somewhere other than forward"
        # So arguably, switching from Left to Right is still "not Forward".
        # But if we want to log "Looking Right for 5s", tracking specific direction is better.
        # Let's track specific direction for better logs.
        if direction != self.last_pose_direction:
             # Reset timer for new direction
             self.head_pose_start_time = time.time()
             self.last_pose_direction = direction
             return False, False, ""

        elapsed = time.time() - self.head_pose_start_time
        if elapsed >= persistence_time:
            # Violation!
            msg = f"User looking {direction} for {int(elapsed)} seconds."
            
            is_active = True
            is_triggered = False
            
            # Log violation with type 'head_pose'
            # We want to consolidate continuous looking away in the same direction
            
            # Use a specific type for direction? e.g. 'head_pose_Right'
            # Or just 'head_pose' and rely on message update?
            # If user switches from Right to Left rapidly, we probably want separate logs.
            # So let's include direction in type or just use generic 'head_pose' but rely on logic.
            
            # If I use 'head_pose', and message changes from "Looking Right..." to "Looking Left...", 
            # ideally that should be a new entry? 
            # My logic in log_violation will update if type is same.
            # So if I use type='head_pose', "Looking Right" might be replaced by "Looking Left".
            # That's probably okay or maybe desired.
            # But let's use direction-specific type to be safe? 
            # Actually, "looking somewhere other than forward" is the violation.
            # If I stick to `violation_type='head_pose'`, it will just update the text.
            # "User looking Right for 5s" -> "User looking Left for 6s" (if they switch instantly).
            # That might be confusing.
            # Let's make type specific to direction: f'head_pose_{direction}'
            
            if self.log_violation(msg, violation_type=f'head_pose_{direction}'):
                is_triggered = True
            
            return is_active, is_triggered, msg
        
        return False, False, ""

    def log_violation(self, message, violation_type=None):
        """
        Log a violation. If it's a continuation of the previous violation (same type, recent),
        update the existing entry instead of creating a new one.
        Returns True if a new log entry was added or an existing one was updated (triggering a toast).
        """
        current_time = time.time()
        
        if self.violations:
            last_violation = self.violations[-1]
            last_type = last_violation.get('type')
            last_time = last_violation.get('timestamp')
            
            # Check if this is a continuation
            # Same type and within 2 seconds (allow for some processing jitter)
            if violation_type is not None and last_type == violation_type and (current_time - last_time < 2.0):
                # Update existing violation
                # Check if message is different only? 
                # Actually, we always want to update timestamp to keep it "alive"
                # And update message to show new duration
                
                # Only return True (trigger toast) if message CHANGED substantially?
                # The user wants toast for "3 sec", "4 sec".
                # But maybe we don't want to spam toast every second?
                # The prompt says: "if no face is detected for 6 sec ... instead of adding 4 entries ... make it as a single entry"
                # "also note that we have to add it with timestamps"
                
                # For LOGS: single entry updating.
                # For TOAST: The app code triggers toast if `is_triggered` is True.
                # If `log_violation` returns True every time we update, we spam toasts.
                # Maybe we only return True on NEW entry?
                # The user said: "a violation log and toast shld be added ... for 4 sec also it shuld be added kinda as like we did in head_psoe"
                # This implies they WANT the toast notification potentially.
                # But for logs, they want single entry.
                
                # Let's update the log entry in place.
                msg_changed = (last_violation['message'] != message)
                last_violation['message'] = message
                last_violation['timestamp'] = current_time
                # We do NOT append.
                
                # Do we return True?
                # If we return True, `app.py` shows a toast.
                # If we return False, `app.py` does nothing.
                # If we want toast updates, we should return True.
                # But typical toast behavior is to show briefly. If we spam it, it might stack up.
                # Streamlit `st.toast` stacks. 
                # If we want to avoid stack spam, maybe we throttle toasts in app.py or here?
                # Let's return True effectively "triggering" the event, but let app handle it.
                # Wait, if I return True, `hp_triggered` becomes True, and `app.py` runs `st.toast(hp_msg)`.
                # If this happens every frame (buffered by sleep(0.03) but logic is every second?), we get many toasts.
                # The user query was specifically about LOGS: "make it as a single entry like no face detected for 6 seconds".

                # Update: User specifically requested only toast updates per second change
                return msg_changed
        
        # New violation
        self.violations.append({
            "timestamp": current_time,
            "message": message,
            "type": violation_type
        })
        return True

    def get_logs(self):
        return self.violations

    def check_object_violation(self, object_data, persistence_time=2.0):
        """
        Check for object detection violations:
        - cell phone detected
        - multiple people detected (body)
        
        Returns: (is_active, is_triggered, message)
        """
        # 1. Phone Detection
        if object_data.get('phone_detected', False):
            # Check persistence
             # Use a unique key for phone timer?
             # Or reuse a generic object timer? Let's use specific.
             if not hasattr(self, 'phone_start_time'):
                 self.phone_start_time = None
                 
             if self.phone_start_time is None:
                 self.phone_start_time = time.time()
                 return False, False, ""
             
             elapsed = time.time() - self.phone_start_time
             if elapsed >= persistence_time:
                 msg = "Mobile Phone Detected!"
                 is_triggered = False
                 if self.log_violation(msg, violation_type='object_phone'):
                     is_triggered = True
                 return True, is_triggered, msg
        else:
            self.phone_start_time = None
            
        # 2. Person Count (Body) > 1
        # The user said "apart from the user if someone else is wandering".
        # So count > 1.
        if object_data.get('person_count', 0) > 1:
             if not hasattr(self, 'person_body_start_time'):
                 self.person_body_start_time = None
                 
             if self.person_body_start_time is None:
                 self.person_body_start_time = time.time()
                 return False, False, ""
             
             elapsed = time.time() - self.person_body_start_time
             if elapsed >= persistence_time:
                 msg = f"Multiple People Detected (Body: {object_data['person_count']})"
                 is_triggered = False
                 # We use a different type than face count to distinguish
                 if self.log_violation(msg, violation_type='object_multiple_people'):
                     is_triggered = True
                 return True, is_triggered, msg
        else:
            self.person_body_start_time = None
            
        return False, False, ""

    def check_identity(self, is_verified, persistence_time=2.0):
        """
        Check for identity mismatch.
        
        Args:
            is_verified (bool): True if identity matches or no reference set (safe). False if mismatch.
            persistence_time (float): Time to wait before triggering violation.
            
        Returns: (is_active, is_triggered, message)
        """
        if is_verified:
            # Identity confirmed or not checking
            self.identity_start_time = None
            return False, False, ""
            
        # Mismatch detected!
        if not hasattr(self, 'identity_start_time') or self.identity_start_time is None:
            self.identity_start_time = time.time()
            return False, False, ""
            
        elapsed = time.time() - self.identity_start_time
        if elapsed >= persistence_time:
            msg = "Identity Verification Failed! Unknown Person Detected."
            is_triggered = False
            
            if self.log_violation(msg, violation_type='identity_mismatch'):
                is_triggered = True
                
            return True, is_triggered, msg
            
        return False, False, ""

    def get_violation_count(self):
        return len(self.violations)

    def check_gaze_violation(self, direction, persistence_time=3.0):
        """
        Check for gaze violations (looking away from screen).
        
        Args:
            direction (str): Current gaze direction.
            persistence_time (float): Time to wait before triggering violation.
            
        Returns: (is_active, is_triggered, message)
        """
        if direction == "Center":
            self.gaze_start_time = None
            self.last_gaze_direction = None
            return False, False, ""
            
        # Looking Away
        if not hasattr(self, 'gaze_start_time') or self.gaze_start_time is None:
            self.gaze_start_time = time.time()
            self.last_gaze_direction = direction
            return False, False, ""
            
        # If direction changed (e.g. Left -> Right), reset?
        # Similar to head pose, we probably want to track specific direction
        if direction != self.last_gaze_direction:
            self.gaze_start_time = time.time()
            self.last_gaze_direction = direction
            return False, False, ""
            
        elapsed = time.time() - self.gaze_start_time
        if elapsed >= persistence_time:
            msg = f"Gaze Violation: Looking {direction} for {int(elapsed)} seconds."
            is_triggered = False
            
            if self.log_violation(msg, violation_type=f'gaze_{direction}'):
                is_triggered = True
                
            return True, is_triggered, msg
            
        return False, False, ""
