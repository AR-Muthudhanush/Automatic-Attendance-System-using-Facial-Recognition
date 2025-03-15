import cv2
import numpy as np
from pathlib import Path
import pickle
import logging
from .face_detection import FaceDetector

class FaceRecognizer:
    def __init__(self, shape_predictor_path: Path, face_rec_model_path: Path, model_path: Path = None):
        self.face_detector = FaceDetector(shape_predictor_path, face_rec_model_path)
        self.known_face_encodings = []
        self.known_face_names = []
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train(self, data_dir: Path):
        logging.info(f"Training on data from: {data_dir}")
        for student_dir in data_dir.glob("*"):
            if not student_dir.is_dir():
                continue
                
            student_name = student_dir.name
            encodings = []
            
            for img_path in student_dir.glob("*.jpg"):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                    
                faces = self.face_detector.detect_faces(image)
                
                if len(faces) == 1:
                    face_encoding = self.face_detector.get_face_encoding(image, faces[0])
                    encodings.append(face_encoding)
            
            if encodings:
                self.known_face_encodings.append(np.mean(encodings, axis=0))
                self.known_face_names.append(student_name)
                logging.info(f"Trained on {len(encodings)} images for {student_name}")
    
    def save_model(self, model_path: Path):
        with open(model_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        logging.info(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: Path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
        logging.info(f"Model loaded from: {model_path}")
    
    def recognize_faces(self, image, tolerance=0.6):
        faces = self.face_detector.detect_faces(image)
        results = []
        
        for face in faces:
            encoding = self.face_detector.get_face_encoding(image, face)
            distances = [np.linalg.norm(encoding - known_enc) for known_enc in self.known_face_encodings]
            
            if distances and min(distances) < tolerance:
                best_match_idx = np.argmin(distances)
                name = self.known_face_names[best_match_idx]
                results.append((face, name))
        
        return results