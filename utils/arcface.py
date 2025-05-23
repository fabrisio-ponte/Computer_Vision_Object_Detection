import cv2
import numpy as np
import onnxruntime

def norm_crop_image(img, landmark=None, image_size=112):
    """Normalize and crop face image using landmarks"""
    if landmark is None:
        return cv2.resize(img, (image_size, image_size))
    
    # Get the center point between eyes
    eye_center = np.mean(landmark[0:2], axis=0)
    
    # Calculate the angle for rotation
    eye_angle = np.degrees(np.arctan2(landmark[1][1] - landmark[0][1], landmark[1][0] - landmark[0][0]))
    
    # Get the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(tuple(eye_center), eye_angle, 1)
    
    # Apply rotation
    rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    
    # Crop the face
    face_size = int(max(np.linalg.norm(landmark[0] - landmark[1]) * 2, 
                       np.linalg.norm(landmark[2] - landmark[3]) * 2))
    face_size = int(face_size * 1.5)  # Add some margin
    
    center = tuple(eye_center.astype(int))
    x1 = max(0, center[0] - face_size // 2)
    y1 = max(0, center[1] - face_size // 2)
    x2 = min(img.shape[1], center[0] + face_size // 2)
    y2 = min(img.shape[0], center[1] + face_size // 2)
    
    cropped = rotated[y1:y2, x1:x2]
    
    # Resize to target size
    return cv2.resize(cropped, (image_size, image_size))

class ArcFace:
    def __init__(self, model_path: str = None, session=None) -> None:
        self.session = session
        self.input_mean = 127.5
        self.input_std = 127.5
        self.taskname = "recognition"

        if session is None:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape

        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape

        outputs = self.session.get_outputs()
        output_names = []
        for output in outputs:
            output_names.append(output.name)

        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def get_feat(self, images: np.ndarray) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]

        input_size = self.input_size
        blob = cv2.dnn.blobFromImages(
            images,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})[0]
        return outputs

    def __call__(self, image, kps):
        aligned_image = norm_crop_image(image, landmark=kps)
        embedding = self.get_feat(aligned_image).flatten()
        return embedding 