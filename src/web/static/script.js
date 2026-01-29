document.addEventListener('DOMContentLoaded', () => {
    console.log("AI Proctor UI Intialized");

    const videoFeed = document.querySelector('.video-feed');
    const toggleViewBtn = document.getElementById('toggleView');
    
    // Placeholder for future interactivity
    const toggleBtn = document.getElementById('toggleCamera');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            console.log("Camera toggle clicked (Not implemented yet)");
        });
    }

    if (toggleViewBtn && videoFeed) {
        toggleViewBtn.addEventListener('click', () => {
            const currentMode = toggleViewBtn.dataset.mode;
            
            if (currentMode === 'raw') {
                // Switch to Analyzed
                toggleViewBtn.dataset.mode = 'analyzed';
                toggleViewBtn.textContent = 'Switch to Raw View';
                videoFeed.src = "/video_feed?mode=analyzed";
            } else {
                // Switch to Raw
                toggleViewBtn.dataset.mode = 'raw';
                toggleViewBtn.textContent = 'Switch to Analyzed View';
                videoFeed.src = "/video_feed?mode=raw";
            }
        });
    }
});
