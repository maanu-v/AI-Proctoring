# 🎨 Test UI Guide

## Quick Start

1. **Start the FastAPI server:**
   ```bash
   ./start_api.sh
   ```

2. **Open the test UI:**
   ```bash
   # Open in your default browser
   open test_ui.html
   # Or on Linux
   xdg-open test_ui.html
   ```

3. **That's it!** The UI will connect to `http://localhost:8000/api`

## Features

### 📋 Session Management
- Create new quiz sessions with Student ID and Quiz ID
- Optional profile image upload for identity verification
- View session details in real-time
- End sessions when done

### 📹 Video Feed
- Start/stop webcam
- Real-time video preview
- FPS counter
- Frame counter
- Live analysis at ~10 FPS

### 📊 Analysis Results
- **Real-time stats cards:**
  - Face count
  - Head pose status (✓ or ⚠️)
  - Gaze status (✓ or ⚠️)
  - Object count

- **Detailed results panel:**
  - Latest 5 analysis results
  - Face detection status
  - Head pose angles (yaw, pitch)
  - Gaze ratios (horizontal, vertical)
  - Blink count and EAR
  - Detected objects with confidence
  - Identity verification status

- **Violations panel:**
  - All active violations
  - Timestamps (first and last occurrence)
  - Violation counts
  - Clear violations button

### ⚙️ Session Settings (Per-Session)
Toggle features on/off for the current session:
- ✓ No Face Detection
- ✓ Multiple Face Detection
- ✓ Head Pose Detection
- ✓ Gaze Detection
- ✓ Object Detection
- ✓ Identity Verification

Changes apply immediately to the current session.

### 🔧 Global Configuration
Adjust thresholds that affect all sessions:

**Head Pose Thresholds:**
- Max Yaw (degrees) - Default: 30°
- Max Pitch (degrees) - Default: 20°

**Gaze Thresholds:**
- Max Horizontal Deviation - Default: 0.3
- Max Vertical Deviation - Default: 0.2

**Blink Detection:**
- EAR Threshold - Default: 0.2
- Consecutive Frames - Default: 2

**Violation Persistence:**
- No Face Duration (seconds) - Default: 3s
- Multiple Faces Duration (seconds) - Default: 2s

Click "Update Configuration" to apply changes globally.

## Usage Workflow

### Basic Testing Flow

```
1. Create Session
   ↓
2. Start Camera
   ↓
3. Start Analysis
   ↓
4. Monitor Results
   ↓
5. End Session
```

### Testing Specific Features

#### Test Face Detection
1. Create session
2. Start camera and analysis
3. Move out of frame → Should trigger "No Face" violation
4. Have someone join you → Should trigger "Multiple Faces" violation

#### Test Head Pose
1. Start analysis
2. Turn head left/right (yaw) → Watch head pose status change
3. Look up/down (pitch) → Watch head pose angles
4. Adjust thresholds in config if too sensitive

#### Test Gaze Detection
1. Start analysis
2. Look at camera (screen) → Should show ✓
3. Look away → Should show ⚠️
4. Adjust gaze thresholds if needed

#### Test Object Detection
1. Start analysis
2. Hold up phone → Should detect "cell phone"
3. Phone appears in "Objects" stat and results

#### Test Identity Verification
1. Upload profile image when creating session
2. Start analysis
3. Your face → Should verify as "Match"
4. Different person → Should show "Mismatch"

#### Test Settings Toggle
1. During active session
2. Toggle any setting off (e.g., Head Pose Detection)
3. That analysis won't run anymore
4. Toggle back on to re-enable

#### Test Configuration Changes
1. Set very low thresholds (e.g., Yaw = 5°)
2. Click "Update Configuration"
3. Head pose becomes very sensitive
4. Reset to comfortable values

## UI Components

### Status Indicators
- 🟦 **Blue (Info):** No active session
- 🟩 **Green (Success):** Session active, things OK
- 🟧 **Orange (Warning):** Warnings/anomalies detected
- 🟥 **Red (Error):** Errors/violations

### Badges
- **Green Badge:** Feature OK/verified
- **Orange Badge:** Warning state
- **Red Badge:** Violation/mismatch

### Stats Cards
- Show real-time counts and status
- Update with every frame analysis
- Color-coded for quick scanning

## Troubleshooting

### Camera Not Working
```
Error: "Permission denied" or "Device not found"
```
**Solution:**
- Check browser permissions (allow camera access)
- Ensure no other app is using the camera
- Try different browser (Chrome/Firefox recommended)

### API Connection Error
```
Error: "Failed to fetch" or "Network error"
```
**Solution:**
- Ensure FastAPI server is running (`./start_api.sh`)
- Check console for CORS errors
- Verify API is at `http://localhost:8000`

### Analysis Not Working
```
Session creates but analysis fails
```
**Solution:**
- Check FastAPI logs for errors
- Ensure all AI models are downloaded
- Check that `yolov8n.pt` exists
- Verify MediaPipe model is present

### Slow Performance
```
Low FPS or laggy UI
```
**Solution:**
- Analysis runs at ~10 FPS by default
- Reduce camera resolution if needed
- Check CPU usage (AI models are intensive)
- Close other heavy applications

## Browser Compatibility

### Recommended Browsers
- ✅ Chrome/Chromium (Best)
- ✅ Firefox (Good)
- ✅ Edge (Good)
- ⚠️ Safari (Limited)

### Required Browser Features
- WebRTC (camera access)
- Canvas API
- Fetch API
- ES6 JavaScript

## Advanced Usage

### Change Analysis Rate
Edit line 691 in `test_ui.html`:
```javascript
analysisInterval = setInterval(analyzeFrame, 100); // 100ms = 10 FPS
```
- `100` = 10 FPS (default)
- `50` = 20 FPS (more frequent)
- `200` = 5 FPS (less frequent, lower CPU)

### Change API Endpoint
Edit line 618 in `test_ui.html`:
```javascript
const API_BASE = 'http://localhost:8000/api';
```
Change to your server URL if deployed elsewhere.

### Customize Styling
All CSS is in the `<style>` section at the top. Modify colors, sizes, layouts as needed.

## Keyboard Shortcuts
None currently implemented, but you can add them!

## Mobile Testing
- UI is responsive and works on mobile
- Camera access works on mobile browsers
- Touch-friendly controls
- May have lower performance on mobile

## Security Notes
- This is a **test/development UI** - not production-ready
- No authentication implemented
- No HTTPS (use for local testing only)
- Camera feed stays local (only processed frames sent to API)

## Next Steps

### For Production
Consider:
- Add authentication/authorization
- Use HTTPS
- Add input validation
- Implement proper error handling
- Add loading states
- Add retry logic
- Add session persistence
- Add analytics/logging

### Integration
This UI demonstrates the API. To integrate:
- Use the same API calls in your production frontend
- Framework of choice (React, Vue, Angular, etc.)
- Copy the JavaScript functions as reference
- Adapt to your state management

## Support

If you encounter issues:
1. Check browser console for errors
2. Check FastAPI logs (`./start_api.sh` output)
3. Verify all dependencies installed
4. Review API documentation (`/docs`)

## API Documentation
For complete API reference: http://localhost:8000/docs

---

**Happy Testing! 🚀**
