import logging
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from config import Config
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self, recognizer, confidence_threshold=Config.FACE_RECOGNITION_TOLERANCE):
        """
        Initialize the model evaluator.
        
        Args:
            recognizer: FaceRecognizer instance
            confidence_threshold: Threshold for face recognition confidence
        """
        self.recognizer = recognizer
        self.confidence_threshold = confidence_threshold
        self.evaluation_results_dir = Config.DATA_DIR / "evaluation_results"
        self.evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    def _get_ground_truth(self, image_path):
        """
        Get ground truth data from corresponding JSON file.
        """
        json_path = Path(image_path).with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get('present_students', [])
        else:
            logging.warning(f"No ground truth file found for {image_path}")
            return []

    def calculate_distances(self, face_encoding, known_encodings):
        """Calculate distances between a face encoding and all known encodings."""
        return [np.linalg.norm(face_encoding - known_enc) for known_enc in known_encodings]

    def evaluate_single_image(self, image_path):
        """
        Evaluate model performance on a single classroom image.
        """
        logging.info(f"Evaluating image: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Get ground truth data
        true_names = self._get_ground_truth(image_path)
        logging.info(f"Ground truth students: {true_names}")
        
        # Get predictions using the recognizer
        recognition_results = self.recognizer.recognize_faces(image, self.confidence_threshold)
        predicted_names = [name for _, name in recognition_results]
        logging.info(f"Predicted students: {predicted_names}")
        
        # Calculate metrics
        correct_recognitions = set(true_names) & set(predicted_names)
        false_positives = set(predicted_names) - set(true_names)
        false_negatives = set(true_names) - set(predicted_names)
        
        metrics = {
            'image_name': Path(image_path).name,
            'total_actual_students': len(true_names),
            'total_predictions': len(predicted_names),
            'correct_recognitions': len(correct_recognitions),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'correctly_recognized_students': list(correct_recognitions),
            'falsely_recognized_students': list(false_positives),
            'missed_students': list(false_negatives)
        }
        
        if len(true_names) > 0:
            metrics['recognition_accuracy'] = len(correct_recognitions) / len(true_names)
        else:
            metrics['recognition_accuracy'] = 0.0
            
        return metrics

    def calculate_recognition_scores(self, image_path):
        """
        Calculate similarity scores for faces in an image against the database.
        """
        image = cv2.imread(str(image_path))
        similarity_scores = []
        true_labels = []
        
        # Get ground truth for current image
        true_names = self._get_ground_truth(image_path)
        
        # Detect faces using the face detector from the recognizer
        faces = self.recognizer.face_detector.detect_faces(image)
        
        for face in faces:
            # Get encoding for detected face
            face_encoding = self.recognizer.face_detector.get_face_encoding(image, face)
            if face_encoding is None:
                continue
                
            # Calculate distances to all known faces
            distances = self.calculate_distances(face_encoding, self.recognizer.known_face_encodings)
            
            if not distances:
                continue
                
            # Convert distances to similarity scores (1 - normalized distance)
            min_dist = min(distances)
            best_match_idx = np.argmin(distances)
            best_match_name = self.recognizer.known_face_names[best_match_idx]
            
            similarity_score = 1 - min(min_dist, 1.0)  # Normalize and convert to similarity
            similarity_scores.append(similarity_score)
            
            # Determine if this is a true match
            true_labels.append(1 if best_match_name in true_names else 0)
        
        return np.array(similarity_scores), np.array(true_labels)

    def evaluate_dataset(self, classroom_dir=None):
        """
        Evaluate the model on multiple classroom images.
        """
        if classroom_dir is None:
            classroom_dir = Config.CLASSROOM_DIR
        
        classroom_dir = Path(classroom_dir)
        logging.info(f"Starting evaluation on images in {classroom_dir}")
        
        all_metrics = []
        all_similarity_scores = []
        all_true_labels = []
        
        # Process each image that has a corresponding JSON file
        for json_file in classroom_dir.glob("*.json"):
            image_path = json_file.with_suffix('.jpg')
            if not image_path.exists():
                logging.warning(f"No corresponding image found for {json_file}")
                continue
            
            try:
                # Evaluate recognition performance
                metrics = self.evaluate_single_image(image_path)
                all_metrics.append(metrics)
                
                # Calculate similarity scores for ROC curve
                scores, labels = self.calculate_recognition_scores(image_path)
                if len(scores) > 0:
                    all_similarity_scores.extend(scores)
                    all_true_labels.extend(labels)
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
        
        # Generate final report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate ROC curve if we have similarity scores
        roc_results = None
        if len(all_similarity_scores) > 0:
            roc_results = self.plot_roc_curve(
                np.array(all_similarity_scores), 
                np.array(all_true_labels),
                timestamp
            )
        
        # Compile final results
        results = {
            'timestamp': timestamp,
            'individual_image_results': all_metrics,
            'aggregate_metrics': self._calculate_aggregate_metrics(all_metrics),
            'roc_results': roc_results,
            'confidence_threshold': self.confidence_threshold
        }
        
        # Save results
        results_path = self.evaluation_results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info(f"Evaluation results saved to {results_path}")
        return results

    def _calculate_aggregate_metrics(self, all_metrics):
        """Calculate aggregate metrics across all evaluated images."""
        if not all_metrics:
            return {}
            
        total_metrics = {
            'total_images_evaluated': len(all_metrics),
            'total_actual_students': sum(m['total_actual_students'] for m in all_metrics),
            'total_predictions': sum(m['total_predictions'] for m in all_metrics),
            'total_correct_recognitions': sum(m['correct_recognitions'] for m in all_metrics),
            'total_false_positives': sum(m['false_positives'] for m in all_metrics),
            'total_false_negatives': sum(m['false_negatives'] for m in all_metrics)
        }
        
        if total_metrics['total_actual_students'] > 0:
            total_metrics['overall_recognition_accuracy'] = (
                total_metrics['total_correct_recognitions'] / 
                total_metrics['total_actual_students']
            )
            
        return total_metrics

    def plot_roc_curve(self, similarity_scores, true_labels, timestamp):
        """Generate and save ROC curve plot with additional metrics."""
        fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
                label=f'Optimal threshold = {optimal_threshold:.2f}')
        plt.legend(loc="lower right")
        
        # Save the plot
        plot_path = self.evaluation_results_dir / f"roc_curve_{timestamp}.png"
        plt.savefig(str(plot_path))
        plt.close()
        
        return {
            'roc_auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'plot_path': str(plot_path)
        }