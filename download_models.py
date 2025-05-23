import os
import gdown
import onnx
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import cv2

def download_and_prepare_models():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Download genderage model
    genderage_model_url = "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/genderage.onnx"
    genderage_model_path = 'models/genderage.onnx'
    
    if not os.path.exists(genderage_model_path):
        print("Downloading genderage model...")
        os.system(f'curl -L {genderage_model_url} -o {genderage_model_path}')
    
    # Download ArcFace model
    arcface_model_url = "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx"
    arcface_model_path = 'models/w600k_r50.onnx'
    
    if not os.path.exists(arcface_model_path):
        print("Downloading ArcFace model...")
        os.system(f'curl -L {arcface_model_url} -o {arcface_model_path}')
    
    # Optimize models for inference
    print("Optimizing models for inference...")
    
    # Optimize genderage model
    if os.path.exists(genderage_model_path):
        model = onnx.load(genderage_model_path)
        optimized_model_path = genderage_model_path.replace('.onnx', '_optimized.onnx')
        onnx.save(model, optimized_model_path)
        print(f"Genderage model optimized and saved to {optimized_model_path}")
    
    # Optimize ArcFace model
    if os.path.exists(arcface_model_path):
        model = onnx.load(arcface_model_path)
        optimized_model_path = arcface_model_path.replace('.onnx', '_optimized.onnx')
        onnx.save(model, optimized_model_path)
        print(f"ArcFace model optimized and saved to {optimized_model_path}")

# Load the model
GENDERAGE_MODEL_PATH = 'models/genderage.onnx'
genderage_session = ort.InferenceSession(GENDERAGE_MODEL_PATH)

def predict_age_gender(face_img):
    # Preprocess: resize to 64x64, BGR->RGB, normalize to [0,1], shape (1,3,64,64)
    face = cv2.resize(face_img, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))  # HWC to CHW
    face = np.expand_dims(face, axis=0)
    # Run inference
    input_name = genderage_session.get_inputs()[0].name
    outputs = genderage_session.run(None, {input_name: face})
    gender_logits, age = outputs
    gender = "Male" if gender_logits[0][0] > gender_logits[0][1] else "Female"
    age = int(age[0][0])
    return age, gender

def process_face(face_img):
    """Process a single face image for age and gender using genderage.onnx"""
    try:
        # Preprocess: resize to 64x64, BGR->RGB, normalize to [0,1], shape (1,3,64,64)
        face = cv2.resize(face_img, (64, 64))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))  # HWC to CHW
        face = np.expand_dims(face, axis=0)
        # Run inference
        input_name = genderage_session.get_inputs()[0].name
        outputs = genderage_session.run(None, {input_name: face})
        gender_logits, age = outputs
        gender = "Male" if gender_logits[0][0] > gender_logits[0][1] else "Female"
        age = int(age[0][0])
        return age, gender
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None, None

if __name__ == "__main__":
    download_and_prepare_models() 