from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import time
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

cap = cv2.VideoCapture(0)

# Load face detection model (OpenCV DNN SSD)
prototxt = "models/deploy.prototxt"
weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        # Prepare input blob and perform face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        h, w = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        encoded = encode_frame(frame)
        socketio.emit('video_frame', {'frame': encoded})
        socketio.sleep(0.03)  # important: socketio.sleep, not time.sleep

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.start_background_task(target=generate_frames)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5008, debug=True, use_reloader=False)
