import cv2
import logging
from pathlib import Path
from datetime import datetime

from config import Config
from src.augmentation import DataAugmentor
from src.face_recognition import FaceRecognizer
from src.attendance_manager import AttendanceManager
from src.utils import draw_results, setup_logging
from src.evaluation import ModelEvaluator

def train_model():
    """Train the face recognition model on the augmented dataset."""
    logging.info("Starting model training...")
    
    # First, augment the dataset
    augmentor = DataAugmentor(
        input_dir=Config.STUDENTS_DIR,
        output_dir=Config.AUGMENTED_DIR
    )
    augmentor.augment_dataset()
    logging.info("Data augmentation completed")
    
    # Train the model
    recognizer = FaceRecognizer(
        Config.SHAPE_PREDICTOR_PATH,
        Config.FACE_RECOGNITION_MODEL_PATH
    )
    recognizer.train(Config.AUGMENTED_DIR)
    recognizer.save_model(Config.TRAINED_MODEL_PATH)
    logging.info("Model training completed and saved")
    
    return recognizer

def process_classroom_image(image_path: str, recognizer: FaceRecognizer):
    """Process a classroom image and mark attendance."""
    logging.info(f"Processing image: {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Recognize faces
    recognition_results = recognizer.recognize_faces(
        image, 
        tolerance=Config.FACE_RECOGNITION_TOLERANCE
    )
    
    # Draw boxes and labels
    output_image = draw_results(image.copy(), recognition_results)
    
    # Mark attendance
    attendance_manager = AttendanceManager(Config.ATTENDANCE_DIR)
    recognized_names = [name for _, name in recognition_results]
    attendance_manager.mark_attendance(recognized_names)
    
    # Save output image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(Config.DATA_DIR / f"output_{timestamp}.jpg")
    cv2.imwrite(output_path, output_image)
    
    return output_path, recognized_names

def main():
    """Main execution function."""
    Config.setup_directories()
    setup_logging(Config.LOG_DIR)
    
    # Initialize recognizer
    if not Config.TRAINED_MODEL_PATH.exists():
        recognizer = train_model()
    else:
        recognizer = FaceRecognizer(
            Config.SHAPE_PREDICTOR_PATH,
            Config.FACE_RECOGNITION_MODEL_PATH,
            Config.TRAINED_MODEL_PATH
        )
        logging.info("Loaded existing model")

    # Process and evaluate single classroom image
    classroom_image_path = "D:\Attendance System\data\classroom\IMG_2860.JPG"
    
    # Process image for attendance
    output_path, recognized_names = process_classroom_image(
        classroom_image_path,
        recognizer
    )
    
    # Evaluate single image
    evaluator = ModelEvaluator(
        recognizer=recognizer,
        confidence_threshold=Config.FACE_RECOGNITION_TOLERANCE
    )
    
    try:
        evaluation_results = evaluator.evaluate_single_image(classroom_image_path)
        
        # Log evaluation results
        logging.info("\nEvaluation Results for Single Image:")
        logging.info(f"Image: {evaluation_results['image_name']}")
        logging.info(f"Recognition Accuracy: {evaluation_results.get('recognition_accuracy', 0):.2%}")
        logging.info(f"Correct Recognitions: {evaluation_results['correct_recognitions']}")
        logging.info(f"False Positives: {evaluation_results['false_positives']}")
        logging.info(f"False Negatives: {evaluation_results['false_negatives']}")
        logging.info(f"Correctly Recognized Students: {evaluation_results['correctly_recognized_students']}")
        logging.info(f"Falsely Recognized Students: {evaluation_results['falsely_recognized_students']}")
        logging.info(f"Missed Students: {evaluation_results['missed_students']}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)