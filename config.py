from pathlib import Path

class Config:
    # Base directories
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # Student photos and dataset directories
    CLASSROOM_DIR = DATA_DIR / "classroom"
    STUDENTS_DIR = DATA_DIR / "students"
    AUGMENTED_DIR = DATA_DIR / "augmented"
    ATTENDANCE_DIR = DATA_DIR / "attendance"
    
    # Model directories
    MODELS_DIR = PROJECT_ROOT / "models"
    TRAINED_MODEL_PATH = MODELS_DIR / "face_recognition_model.pkl"
    
    # Dlib model paths
    SHAPE_PREDICTOR_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
    FACE_RECOGNITION_MODEL_PATH = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"
    
    # Recognition parameters
    FACE_RECOGNITION_TOLERANCE = 0.6
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary directories."""
        dirs = [
            cls.DATA_DIR,
            cls.LOG_DIR,
            cls.STUDENTS_DIR,
            cls.AUGMENTED_DIR,
            cls.ATTENDANCE_DIR,
            cls.MODELS_DIR,
            cls.CLASSROOM_DIR
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)