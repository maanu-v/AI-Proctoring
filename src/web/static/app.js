// ===== Application State =====
const state = {
    features: [],
    sessionStartTime: Date.now(),
    isConnected: false,
    videoSocket: null,
    controlSocket: null
};

// ===== DOM Elements =====
const elements = {
    videoFeed: document.getElementById('videoFeed'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    cameraStatus: document.getElementById('cameraStatus'),
    resolution: document.getElementById('resolution'),
    fps: document.getElementById('fps'),
    featuresList: document.getElementById('featuresList'),
    alertsContainer: document.getElementById('alertsContainer'),
    sessionTime: document.getElementById('sessionTime'),
    facesCount: document.getElementById('facesCount'),
    gazeStatus: document.getElementById('gazeStatus'),
    violationsCount: document.getElementById('violationsCount'),
    confidence: document.getElementById('confidence'),
    sidebarToggle: document.getElementById('sidebarToggle'),
    sidebar: document.getElementById('sidebar'),
    fullscreenBtn: document.getElementById('fullscreenBtn'),
    snapshotBtn: document.getElementById('snapshotBtn'),
    toastContainer: document.getElementById('toastContainer')
};

// ===== Utility Functions =====
function formatTime(milliseconds) {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 250ms ease-in-out';
        setTimeout(() => toast.remove(), 250);
    }, duration);
}

function addAlert(message, type = 'warning') {
    // Remove empty state if present
    const emptyState = elements.alertsContainer.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }
    
    const alert = document.createElement('div');
    alert.className = `alert-item ${type}`;
    alert.textContent = message;
    
    elements.alertsContainer.insertBefore(alert, elements.alertsContainer.firstChild);
    
    // Keep only last 5 alerts
    const alerts = elements.alertsContainer.querySelectorAll('.alert-item');
    if (alerts.length > 5) {
        alerts[alerts.length - 1].remove();
    }
}

// ===== Video Streaming =====
function initVideoStream() {
    // Use HTTP streaming (img tag with multipart stream)
    const videoUrl = `${window.location.origin}/api/video/feed`;
    elements.videoFeed.src = videoUrl;
    
    elements.videoFeed.onerror = () => {
        console.error('Video feed error');
        showToast('Failed to load video feed', 'error');
        updateConnectionStatus(false);
    };
    
    elements.videoFeed.onload = () => {
        console.log('Video feed loaded');
        updateConnectionStatus(true);
    };
}

// ===== WebSocket for Control =====
function initControlWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/control`;
    
    state.controlSocket = new WebSocket(wsUrl);
    
    state.controlSocket.onopen = () => {
        console.log('Control WebSocket connected');
        showToast('Connected to server', 'success');
        
        // Request initial status
        sendControlMessage({ action: 'get_status' });
    };
    
    state.controlSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleControlMessage(data);
    };
    
    state.controlSocket.onerror = (error) => {
        console.error('Control WebSocket error:', error);
        showToast('Connection error', 'error');
    };
    
    state.controlSocket.onclose = () => {
        console.log('Control WebSocket disconnected');
        showToast('Disconnected from server', 'warning');
        updateConnectionStatus(false);
        
        // Attempt to reconnect after 3 seconds
        setTimeout(initControlWebSocket, 3000);
    };
}

function sendControlMessage(message) {
    if (state.controlSocket && state.controlSocket.readyState === WebSocket.OPEN) {
        state.controlSocket.send(JSON.stringify(message));
    }
}

function handleControlMessage(data) {
    console.log('Control message received:', data);
    
    switch (data.type) {
        case 'status':
            updateCameraProperties(data.properties);
            break;
        case 'ack':
            showToast(`${data.feature} ${data.enabled ? 'enabled' : 'disabled'}`, 'success');
            break;
        case 'error':
            showToast(data.message, 'error');
            break;
    }
}

// ===== UI Updates =====
function updateConnectionStatus(connected) {
    state.isConnected = connected;
    
    if (connected) {
        elements.statusDot.className = 'status-dot status-active';
        elements.statusText.textContent = 'Active';
        elements.cameraStatus.textContent = 'Connected';
    } else {
        elements.statusDot.className = 'status-dot status-inactive';
        elements.statusText.textContent = 'Inactive';
        elements.cameraStatus.textContent = 'Disconnected';
    }
}

function updateCameraProperties(properties) {
    if (properties.width && properties.height) {
        elements.resolution.textContent = `${properties.width}x${properties.height}`;
    }
    if (properties.fps) {
        elements.fps.textContent = properties.fps;
    }
}

function updateSessionTime() {
    const elapsed = Date.now() - state.sessionStartTime;
    elements.sessionTime.textContent = formatTime(elapsed);
}

// ===== Features Management =====
async function loadFeatures() {
    try {
        const response = await fetch('/api/features');
        const data = await response.json();
        state.features = data.features;
        renderFeatures();
    } catch (error) {
        console.error('Failed to load features:', error);
        showToast('Failed to load features', 'error');
    }
}

function renderFeatures() {
    elements.featuresList.innerHTML = '';
    
    state.features.forEach(feature => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        featureItem.innerHTML = `
            <div class="feature-header">
                <span class="feature-name">${feature.name}</span>
                <label class="toggle-switch">
                    <input type="checkbox" 
                           data-feature-id="${feature.id}" 
                           ${feature.enabled ? 'checked' : ''}>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <p class="feature-description">${feature.description}</p>
        `;
        
        const toggle = featureItem.querySelector('input[type="checkbox"]');
        toggle.addEventListener('change', (e) => {
            handleFeatureToggle(feature.id, e.target.checked);
        });
        
        elements.featuresList.appendChild(featureItem);
    });
}

function handleFeatureToggle(featureId, enabled) {
    console.log(`Toggle feature: ${featureId} = ${enabled}`);
    
    // Update local state
    const feature = state.features.find(f => f.id === featureId);
    if (feature) {
        feature.enabled = enabled;
    }
    
    // Send to server
    sendControlMessage({
        action: 'toggle_feature',
        feature: featureId,
        enabled: enabled
    });
    
    // Simulate alert for demo
    if (enabled) {
        addAlert(`${feature.name} activated`, 'success');
    }
}

// ===== Event Handlers =====
function handleSidebarToggle() {
    elements.sidebar.classList.toggle('open');
}

function handleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

function handleSnapshot() {
    // Create a canvas to capture the current frame
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = elements.videoFeed.naturalWidth || elements.videoFeed.width;
    canvas.height = elements.videoFeed.naturalHeight || elements.videoFeed.height;
    
    ctx.drawImage(elements.videoFeed, 0, 0);
    
    // Convert to blob and download
    canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `snapshot-${Date.now()}.jpg`;
        a.click();
        URL.revokeObjectURL(url);
        
        showToast('Snapshot saved', 'success');
    }, 'image/jpeg', 0.95);
}

// ===== Event Listeners =====
elements.sidebarToggle.addEventListener('click', handleSidebarToggle);
elements.fullscreenBtn.addEventListener('click', handleFullscreen);
elements.snapshotBtn.addEventListener('click', handleSnapshot);

// ===== Health Check =====
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateConnectionStatus(data.video_stream_active);
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateConnectionStatus(false);
    }
}

// ===== Initialization =====
async function init() {
    console.log('Initializing AI Proctor application...');
    
    // Load features
    await loadFeatures();
    
    // Initialize video stream
    initVideoStream();
    
    // Initialize control WebSocket
    initControlWebSocket();
    
    // Start session timer
    setInterval(updateSessionTime, 1000);
    
    // Periodic health check
    setInterval(checkHealth, 10000);
    
    // Initial health check
    checkHealth();
    
    console.log('Application initialized');
}

// ===== Start Application =====
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// ===== Demo Data Updates (Remove when real AI is integrated) =====
setInterval(() => {
    if (state.isConnected) {
        // Simulate random updates for demo
        elements.facesCount.textContent = Math.random() > 0.5 ? '1' : '0';
        elements.gazeStatus.textContent = Math.random() > 0.7 ? 'Looking Away' : 'Normal';
        elements.confidence.textContent = `${Math.floor(90 + Math.random() * 10)}%`;
    }
}, 2000);
