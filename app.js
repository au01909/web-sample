const videoElement = document.getElementById('video-stream');
const logList = document.getElementById('log-list');
const startButton = document.getElementById('start-button');

// Function to start video stream
startButton.addEventListener('click', async () => {
    if (navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;

            // Call the backend for predictions every few seconds
            setInterval(async () => {
                const frame = await captureFrame(videoElement);
                const action = await getPrediction(frame);
                logAction(action);
            }, 2000);

        } catch (err) {
            console.error("Error accessing video stream: ", err);
        }
    }
});

// Capture frame from video element
async function captureFrame(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
}

// Send frame to backend for prediction
async function getPrediction(frame) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: JSON.stringify({ image: frame }),
        headers: { 'Content-Type': 'application/json' }
    });
    const result = await response.json();
    return result.action;
}

// Log detected action
function logAction(action) {
    const listItem = document.createElement('li');
    listItem.textContent = `${new Date().toLocaleTimeString()} - ${action}`;
    logList.appendChild(listItem);
}
