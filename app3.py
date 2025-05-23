from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import time
import numpy as np
from deepface import DeepFace
import threading
from queue import Queue
import torch
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# Initialize video capture with debug information
logger.info("Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open webcam")
    exit()

logger.info("Camera opened successfully")

# Get initial camera properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
logger.info(f"Initial camera properties:")
logger.info(f"Resolution: {width}x{height}")
logger.info(f"FPS: {fps}")

# Load face detection model (OpenCV DNN SSD)
logger.info("Loading face detection model...")
prototxt = "models/deploy.prototxt"
weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)

# Configure face detection model
face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
logger.info("Using CPU for face detection (OpenCV DNN)")

# Global variables for face analysis
face_queue = Queue(maxsize=5)
analysis_results = {}
analysis_lock = threading.Lock()
last_analysis_time = {}
PROCESSING_INTERVAL = 0.1  # Process every 100ms
FACE_ANALYSIS_INTERVAL = 3.0  # Analyze each face every 3 seconds

# Configure DeepFace to use MPS if available
if torch.backends.mps.is_available():
    logger.info("MPS (Metal) is available - Using for DeepFace analysis")
    device = 'mps'
else:
    logger.info("MPS not available - Using CPU for DeepFace analysis")
    device = 'cpu'

def encode_frame(frame):
    try:
        # Convert frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding frame: {str(e)}")
        return None

def draw_text_with_background(img, text, position, font_scale=0.7, thickness=2):
    try:
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate background rectangle position
        x, y = position
        bg_rect_pt1 = (x, y - text_height - 10)
        bg_rect_pt2 = (x + text_width + 10, y + 10)
        
        # Draw background rectangle
        cv2.rectangle(img, bg_rect_pt1, bg_rect_pt2, (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, text, (x + 5, y - 5), font, font_scale, (0, 255, 0), thickness)
    except Exception as e:
        logger.error(f"Error drawing text: {str(e)}")

def analyze_faces_worker():
    logger.info(f"Starting face analysis worker using {device}")
    
    while True:
        try:
            if not face_queue.empty():
                face_data = face_queue.get()
                frame_id, face_img, face_id = face_data
                
                # Skip if this face was analyzed recently
                current_time = time.time()
                if face_id in last_analysis_time and current_time - last_analysis_time[face_id] < FACE_ANALYSIS_INTERVAL:
                    continue
                
                try:
                    # Analyze face using DeepFace with optimized settings
                    result = DeepFace.analyze(
                        face_img,
                        actions=['age', 'gender'],
                        enforce_detection=False,
                        silent=True,
                        detector_backend='opencv',  # Using OpenCV for detection since we already have the face
                        align=True
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    with analysis_lock:
                        analysis_results[face_id] = (result['age'], result['gender'])
                        last_analysis_time[face_id] = current_time
                        logger.info(f"Face analyzed - Age: {result['age']}, Gender: {result['gender']}")
                except Exception as e:
                    logger.error(f"Error in face analysis: {str(e)}")
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
        time.sleep(0.05)

def get_face_id(x1, y1, x2, y2):
    return f"{x1}_{y1}_{x2}_{y2}"

def generate_frames():
    logger.info("Starting frame generation...")
    frame_count = 0
    last_frame_time = time.time()
    analysis_thread = threading.Thread(target=analyze_faces_worker, daemon=True)
    analysis_thread.start()
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_frame_time < PROCESSING_INTERVAL:
                time.sleep(0.01)
                continue
                
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame")
                time.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info(f"Processing frame {frame_count}")

            if frame_count % 3 != 0:  # Process every third frame
                continue

            # Prepare input blob and perform face detection
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            h, w = frame.shape[:2]
            faces_detected = 0
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    faces_detected += 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_id = get_face_id(x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    with analysis_lock:
                        if face_id in analysis_results:
                            age, gender = analysis_results[face_id]
                            text = f"Age: {age} | Gender: {gender}"
                            draw_text_with_background(frame, text, (x1, y1 - 10))
                    
                    if face_queue.qsize() < 3:
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            face_queue.put((frame_count, face_img, face_id))

            if faces_detected > 0:
                logger.info(f"Detected {faces_detected} faces in frame")

            # Add frame counter for debugging
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            encoded = encode_frame(frame)
            if encoded:
                socketio.emit('video_frame', {'frame': encoded})
                if frame_count % 30 == 0:
                    logger.info(f"Frame {frame_count} sent successfully")
            else:
                logger.error("Failed to encode frame")
            last_frame_time = current_time
            time.sleep(0.03)
        except Exception as e:
            logger.error(f"Error in frame generation: {str(e)}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index2.html')

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    socketio.start_background_task(target=generate_frames)

if __name__ == '__main__':
    logger.info("Starting server...")
    socketio.run(app, host='0.0.0.0', port=5009, debug=True, use_reloader=False) 