from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import time

app = Flask(__name__)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

def analyze_face(face_img):
    try:
        # Analyze using DeepFace's pre-trained models
        result = DeepFace.analyze(
            face_img,
            actions=['age', 'gender'],
            enforce_detection=False,
            silent=True
        )
        return result[0]['age'], result[0]['gender']
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None, None

def generate_frames():
    last_analysis_time = {}
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MediaPipe
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
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Get face region
                face_roi = frame[y:y+height, x:x+width]
                
                # Analyze face every 3 seconds
                current_time = time.time()
                face_id = f"{x}_{y}_{width}_{height}"
                
                if face_id not in last_analysis_time or current_time - last_analysis_time[face_id] > 3:
                    try:
                        # Predict gender and age using DeepFace
                        age, gender = analyze_face(face_roi)
                        
                        if age is not None and gender is not None:
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
                    except Exception as e:
                        print(f"Error analyzing face: {e}")
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index6.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5012, debug=True) 