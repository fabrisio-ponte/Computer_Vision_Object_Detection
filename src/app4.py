from flask import Flask, Response, render_template
import cv2
import numpy as np
from deepface import DeepFace
import time

app = Flask(__name__)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    last_analysis_time = {}
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Analyze face every 3 seconds
            current_time = time.time()
            face_id = f"{x}_{y}_{w}_{h}"
            
            if face_id not in last_analysis_time or current_time - last_analysis_time[face_id] > 3:
                try:
                    # Analyze face
                    result = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)
                    if isinstance(result, list):
                        result = result[0]
                    
                    # Add text with black background
                    text = f"Age: {result['age']} | Gender: {result['gender']}"
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
    return render_template('index4.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True) 