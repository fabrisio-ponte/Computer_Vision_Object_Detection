from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import threading
import time
import numpy as np

import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

cap = cv2.VideoCapture(0)

#Face detection model load

prototxt = "models/deploy.prototxt"
weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

### Models

# Load FairFace model (race)
fairface_model = torch.hub.load('s-nlp/face-emotion-recognition', 'fairface_race', pretrained=True)
fairface_model.eval()

# Load emotion model (example)
emotion_model = torch.hub.load('s-nlp/face-emotion-recognition', 'emotion_vgg', pretrained=True)
emotion_model.eval()

# Age and gender model files (OpenCV Caffe)
age_proto = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_proto = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"

age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
RACE_LIST = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_attributes(face_img):
    # face_img: cropped face in BGR

    # Age and gender (OpenCV)
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    # Race (FairFace)
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        race_preds = fairface_model(input_tensor)
    race = RACE_LIST[race_preds.argmax()]

    # Emotion
    with torch.no_grad():
        emotion_preds = emotion_model(input_tensor)
    emotion_idx = emotion_preds.argmax()
    emotion = EMOTION_LIST[emotion_idx]

    return gender, age, race, emotion

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
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
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                gender, age, race, emotion = predict_attributes(face)

                label = f"{gender}, {age}, {race}, {emotion}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        encoded = encode_frame(frame)
        socketio.emit('video_frame', {'frame': encoded})

        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    thread = threading.Thread(target=generate_frames)
    thread.daemon = True
    thread.start()
    socketio.run(app, host='0.0.0.0', port=5007)
