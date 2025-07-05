from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort

app = Flask(__name__)

# Global variables
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue()
camera_thread = None
processing_thread = None
stop_event = threading.Event()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# Load the genderage model
GENDERAGE_MODEL_PATH = 'models/genderage.onnx'
genderage_session = ort.InferenceSession(GENDERAGE_MODEL_PATH)

def process_face(face_img):
    """Process a single face image for age and gender"""
    try:
        # Preprocess: resize to 96x96, BGR->RGB, normalize to [0,1], shape (1,3,96,96)
        face = cv2.resize(face_img, (96, 96))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        
        # Run inference
        input_name = genderage_session.get_inputs()[0].name
        output = genderage_session.run(None, {input_name: face})[0][0]
        
        # The output format is [male_prob, female_prob, age]
        male_prob = output[0]
        female_prob = output[1]
        age = int(output[2] * 100)  # Scale age to 0-100 range
        gender = "Male" if male_prob > female_prob else "Female"
        
        return age, gender
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None, None

def camera_worker():
    """Background thread for camera capture"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Put frame in queue, remove old frame if queue is full
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
        
        time.sleep(0.01)  # Small delay to prevent CPU overload
    
    cap.release()

def processing_worker():
    """Background thread for face processing"""
    last_analysis_time = {}
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = face_detection.process(frame_rgb)
            
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    # Get face region
                    face_roi = frame[y:y+height, x:x+width]
                    
                    # Process face every 3 seconds
                    current_time = time.time()
                    face_id = f"{x}_{y}_{width}_{height}"
                    
                    if face_id not in last_analysis_time or current_time - last_analysis_time[face_id] > 3:
                        # Process face in a separate thread
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(process_face, face_roi)
                            age, gender = future.result()
                            
                            if age is not None and gender is not None:
                                # Draw results
                                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                
                                # Add text with black background
                                text = f"Age: {age} | Gender: {gender}"
                                
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.7
                                thickness = 2
                                
                                # Get text size
                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                
                                # Draw black background
                                cv2.rectangle(frame, (x, y-30), (x+text_width+10, y), (0, 0, 0), -1)
                                
                                # Draw text
                                cv2.putText(frame, text, (x+5, y-10), font, font_scale, (0, 255, 0), thickness)
                                
                                last_analysis_time[face_id] = current_time
            
            # Put processed frame in result queue
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            result_queue.put(frame)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in processing worker: {e}")
            continue

def generate_frames():
    """Generator function for video streaming"""
    while not stop_event.is_set():
        try:
            frame = result_queue.get(timeout=1)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error generating frame: {e}")
            continue

@app.route('/')
def index():
    return render_template('index7.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_background_threads():
    """Start background threads for camera and processing"""
    global camera_thread, processing_thread
    
    camera_thread = threading.Thread(target=camera_worker)
    processing_thread = threading.Thread(target=processing_worker)
    
    camera_thread.daemon = True
    processing_thread.daemon = True
    
    camera_thread.start()
    processing_thread.start()

def cleanup():
    """Cleanup function to stop threads and release resources"""
    stop_event.set()
    if camera_thread:
        camera_thread.join()
    if processing_thread:
        processing_thread.join()

if __name__ == '__main__':
    try:
        start_background_threads()
        app.run(host='0.0.0.0', port=5013, debug=False)
    finally:
        cleanup() 