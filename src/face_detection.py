import dlib
import cv2
import numpy as np
from pathlib import Path

class FaceDetector:
    def __init__(self, shape_predictor_path: Path, face_rec_model_path: Path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(shape_predictor_path))
        self.face_rec = dlib.face_recognition_model_v1(str(face_rec_model_path))
    
    def detect_faces(self, image):
        return self.detector(image)
    
    def get_face_encoding(self, image, face):
        shape = self.predictor(image, face)
        return np.array(self.face_rec.compute_face_descriptor(image, shape))
