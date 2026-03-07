# Session Resume Feature

## Overview

The Session Resume feature automatically handles students leaving and returning to exams, tracking this activity as a potential violation indicator.

## Problem Solved

Previously, if a student closed their browser or lost connection and tried to restart the session, they would get a **400 Bad Request** error saying "Session already exists". This forced manual intervention to end and restart sessions.

Now, the system:
- ✅ Automatically resumes existing sessions
- ✅ Tracks how long student was inactive
- ✅ Logs as violation if suspicious (> 10 seconds)
- ✅ Counts total number of resume attempts
- ✅ Provides detailed resume history in final report

## Use Cases

### Legitimate Scenarios
- Accidental browser close
- Network disconnection
- Page refresh
- Brief connection issues

### Suspicious Scenarios (Flagged as Violations)
- Leaving to look up answers
- Consulting with others
- Accessing unauthorized materials
- Repeatedly leaving and returning (pattern detection)

## How It Works

### 1. Student Starts Exam
```javascript
POST /api/session/start
{
  student_id: "STU123",
  quiz_id: "QUIZ001"
}

Response:
{
  "session_id": "STU123_QUIZ001_1710675045",
  "message": "Session started successfully"
}
```

### 2. Student Leaves (closes browser)
- System tracks last activity timestamp
- Session remains active
- **Session is NOT ended**

### 3. Student Returns (same student_id + quiz_id)
```javascript
POST /api/session/start
{
  student_id: "STU123",  // Same student
  quiz_id: "QUIZ001"     // Same quiz
}

Response:
{
  "session_id": "STU123_QUIZ001_1710675045",
  "message": "Session resumed successfully (Resume #1)"
}
```

### 4. System Actions on Resume
```javascript
{
  timestamp: "2026-03-07T10:35:20.123456",
  inactivity_seconds: 45,
  resume_number: 1
}
```

**If inactivity > 10 seconds:**
- Violation logged: `session_resume`
- Message: "Session resumed after 45s of inactivity (Resume #1)"
- Added to violations list
- Included in final report

## API Response Details

### Session Info (GET /api/session/{session_id})
```json
{
  "student_id": "STU123",
  "quiz_id": "QUIZ001",
  "created_at": "2026-03-07T10:30:00.000000",
  "last_activity": "2026-03-07T10:35:20.123456",
  "frame_count": 1500,
  "violation_count": 3,
  "resume_count": 1,  // NEW
  "settings": { ... }
}
```

### Final Report (POST /api/session/{session_id}/end)
```json
{
  "session_id": "STU123_QUIZ001_1710675045",
  "student_id": "STU123",
  "duration_seconds": 3600,
  "total_violations": 3,
  "resume_count": 2,  // NEW
  "resume_events": [  // NEW - Detailed history
    {
      "timestamp": "2026-03-07T10:35:20.123456",
      "inactivity_seconds": 45,
      "resume_number": 1
    },
    {
      "timestamp": "2026-03-07T10:42:15.789012",
      "inactivity_seconds": 180,
      "resume_number": 2
    }
  ],
  "violations": [
    {
      "type": "session_resume",
      "message": "Session resumed after 45s of inactivity (Resume #1)",
      "timestamp": "2026-03-07T10:35:20.123456"
    },
    {
      "type": "session_resume",
      "message": "Session resumed after 180s of inactivity (Resume #2)",
      "timestamp": "2026-03-07T10:42:15.789012"
    }
  ]
}
```

## Integration Examples

### Basic Integration

```javascript
async function startOrResumeExam(studentId, quizId) {
    const formData = new FormData();
    formData.append('student_id', studentId);
    formData.append('quiz_id', quizId);
    
    const response = await fetch('http://localhost:8000/api/session/start', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    // Check if resumed
    if (data.message.includes('resumed')) {
        console.warn('Session was resumed');
        showWarningToStudent('Your session was paused. Please stay on this page.');
    }
    
    return data.session_id;
}
```

### Advanced Pattern Detection

```javascript
async function monitorSessionResumes(sessionId) {
    const response = await fetch(`http://localhost:8000/api/session/${sessionId}`);
    const session = await response.json();
    
    // Alert if suspicious pattern
    if (session.resume_count > 3) {
        alertProctor({
            severity: 'high',
            message: `Student ${session.student_id} has left exam ${session.resume_count} times`,
            action: 'review_required'
        });
    }
    
    // Check recent inactivity
    const lastActivity = new Date(session.last_activity);
    const now = new Date();
    const idleSeconds = (now - lastActivity) / 1000;
    
    if (idleSeconds > 30) {
        showWarning('Are you still there? Please interact with the exam.');
    }
}
```

### Automatic Monitoring

```javascript
class ExamSession {
    constructor(studentId, quizId) {
        this.studentId = studentId;
        this.quizId = quizId;
        this.sessionId = null;
        this.monitorInterval = null;
    }
    
    async start() {
        // Start or resume session
        const formData = new FormData();
        formData.append('student_id', this.studentId);
        formData.append('quiz_id', this.quizId);
        
        const response = await fetch('http://localhost:8000/api/session/start', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        this.sessionId = data.session_id;
        
        // Handle resume
        if (data.message.includes('Resume')) {
            const resumeNum = parseInt(data.message.match(/#(\d+)/)[1]);
            this.handleResume(resumeNum);
        }
        
        // Start monitoring
        this.startMonitoring();
    }
    
    handleResume(resumeCount) {
        // Show warning to student
        showNotification(
            'Your exam session was resumed. ' +
            'Leaving the exam multiple times may be flagged as suspicious.',
            'warning'
        );
        
        // Alert proctor if excessive
        if (resumeCount >= 3) {
            fetch('/your-api/proctor/alert', {
                method: 'POST',
                body: JSON.stringify({
                    student_id: this.studentId,
                    type: 'excessive_resumes',
                    count: resumeCount
                })
            });
        }
    }
    
    startMonitoring() {
        // Check every 30 seconds
        this.monitorInterval = setInterval(async () => {
            const response = await fetch(
                `http://localhost:8000/api/session/${this.sessionId}`
            );
            const data = await response.json();
            
            console.log(`Resume count: ${data.resume_count}`);
            console.log(`Violations: ${data.violation_count}`);
        }, 30000);
    }
    
    stop() {
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
        }
    }
}

// Usage
const exam = new ExamSession('STU123', 'QUIZ001');
await exam.start();
```

## Configuration

### Adjust Inactivity Threshold

Currently hardcoded to 10 seconds in `src/web/core/session_manager.py`:

```python
# Log as violation if inactivity was significant (> 10 seconds)
if inactivity_duration > 10:
    msg = f"Session resumed after {int(inactivity_duration)}s of inactivity (Resume #{self.resume_count})"
    self.violation_tracker.log_violation(msg, violation_type='session_resume')
```

To customize, modify the threshold value.

## Best Practices

### 1. Inform Students
```javascript
// Before starting exam
function showExamRules() {
    alert(
        'Exam Rules:\n' +
        '- Stay on this page throughout the exam\n' +
        '- Do not close or refresh the browser\n' +
        '- Leaving the exam will be tracked as a violation\n' +
        '- Multiple exits may result in exam termination'
    );
}
```

### 2. Set Limits
```javascript
async function enforceResumeLimit(sessionId, maxResumes = 3) {
    const response = await fetch(`http://localhost:8000/api/session/${sessionId}`);
    const session = await response.json();
    
    if (session.resume_count >= maxResumes) {
        // Auto-submit exam
        await submitExam(sessionId);
        
        // Show message
        alert(
            'You have left the exam too many times. ' +
            'Your exam has been automatically submitted.'
        );
        
        return false;
    }
    
    return true;
}
```

### 3. Grace Period for Brief Disconnections
```javascript
// Don't count very brief disconnections (< 5 seconds)
// This is handled server-side, but you can add client-side logic too

function handleBeforeUnload() {
    // Warn user before leaving
    return 'Leaving this page will pause your exam. Are you sure?';
}

window.addEventListener('beforeunload', handleBeforeUnload);
```

### 4. Real-time Alerts
```javascript
// WebSocket-based real-time monitoring
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.violations) {
        data.violations.forEach(v => {
            if (v.type === 'session_resume') {
                // Notify proctor in real-time
                notifyProctor({
                    student: data.student_id,
                    event: 'resumed_session',
                    details: v.message
                });
            }
        });
    }
};
```

## Limitations

1. **Session ID Based on Student+Quiz:** Same student can't have multiple concurrent sessions for same quiz
2. **No Cross-Browser Detection:** If student switches browsers, creates new session
3. **Manual Threshold:** Inactivity threshold is hardcoded (can be made configurable)
4. **No Auto-End:** Sessions don't automatically end after prolonged inactivity

## Future Enhancements

- [ ] Configurable inactivity thresholds per exam
- [ ] Auto-end sessions after X minutes of inactivity
- [ ] Browser fingerprinting to detect device switches
- [ ] Graduated penalties (warning → point deduction → auto-submit)
- [ ] Real-time proctor dashboard showing resume patterns
- [ ] Analytics: average resumes per exam, correlation with scores

---

**Last Updated:** March 7, 2026
**Feature Version:** 1.0
