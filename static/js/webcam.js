// webcam.js — Reusable webcam utilities
let stream = null;

async function startWebcam(videoElementId) {
    const video = document.getElementById(videoElementId);
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

function captureFrame(videoElementId, canvasElementId) {
    const video  = document.getElementById(videoElementId);
    const canvas = document.getElementById(canvasElementId);
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg");  // Returns base64 string
}

async function sendToAPI(endpoint, payload) {
    const response = await fetch(endpoint, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload)
    });
    return await response.json();
}