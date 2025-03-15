# Automatic Attendance System Using Facial Recognition

## Project Overview

The **Automatic Attendance System** leverages facial recognition technology to automate and streamline the traditional attendance-taking process. By utilizing deep learning models, it captures classroom images, detects faces, identifies individuals, and records attendance automatically. The system enhances accuracy, efficiency, and security in attendance management, making it suitable for educational and professional environments.

## Photos & Architecture

## Architectural Diagram  
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1sED-EYoFRcPqCPrM9BUckR_MiIPfmOoz" width="600">
</p>  

## System Implementation Photos  
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1rWcB2ZlAwp8yx7BKagfXca-1gVGh5YSP" width="400">
  <img src="https://drive.google.com/uc?export=view&id=1TXinEjZtTU2Wr0X_s7uiqwhJC8q3S0JN" width="400">
</p>

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1YLQMmaRm9ffuucAhkLPdHuMQie-6J6Rf" width="400">
  <img src="https://drive.google.com/uc?export=view&id=1-qzNyCdcR-qHLJig02Zu2W7Vjv5BRJUg" width="400">
</p>


### Detailed Project Description
For an in-depth understanding of the system's working, refer to the detailed documentation: [Project Working PDF](https://drive.google.com/file/d/19GCaB-zp9WXLOfB8iwRN5YHk7aZimB0c/view?usp=sharing)

## Features

- **Automated Attendance Recording**: Eliminates manual roll calls and sign-in sheets.
- **Facial Recognition**: Uses deep learning for accurate face detection and recognition.
- **Data Augmentation**: Enhances training data for robust model performance.
- **CSV-Based Attendance Storage**: Ensures structured and accessible records.
- **Model Evaluation**: Includes ROC curve analysis for performance assessment.
- **Modular System**: Easy to expand and maintain.

## Technology Stack

### Programming Language:

- Python

### Libraries Used:

- **OpenCV (cv2)**: Image processing and visualization.
- **Dlib**: Face detection and recognition.
- **NumPy**: Numerical computations for face encoding.
- **Pandas**: Data storage and manipulation.
- **Albumentations**: Data augmentation for training.
- **Scikit-learn**: Model evaluation and ROC analysis.
- **Matplotlib**: Visualization of evaluation metrics.
- **Pathlib & Logging**: File handling and error logging.
- **JSON**: Ground truth label storage.

### Models Used:

- **Face Detection**: `dlib.get_frontal_face_detector()` (Histogram of Oriented Gradients based model).
- **Face Recognition**: `dlib.face_recognition_model_v1` (ResNet-based deep learning model generating 128-dimensional face embeddings).

## System Workflow

1. **Data Collection**: Student photos are collected and stored.
2. **Data Augmentation**: Images are transformed to enhance diversity.
3. **Model Training**: Facial recognition models are trained on augmented data.
4. **Classroom Image Acquisition**: Real-time images are captured.
5. **Face Detection**: Identifies faces using Dlib’s face detector.
6. **Face Encoding**: Converts faces into numerical embeddings.
7. **Face Recognition**: Matches detected faces with stored student profiles.
8. **Attendance Marking**: Saves recognized student details into a CSV file.
9. **Model Evaluation**: Assesses recognition accuracy using ROC analysis.

## System Components

### Core Modules:

- **DataAugmentor**: Performs image transformations.
- **FaceRecognizer**: Handles detection, encoding, and recognition.
- **AttendanceManager**: Stores and maintains attendance records.
- **ModelEvaluator**: Evaluates system performance using metrics.

## Project Directory Structure
```
ATTENDANCE SYSTEM
│── data
│   │── attendance
│   │   ├── attendance_2024-12-24.csv
│   │── augmented
│   │── classroom
│   │── evaluation_results
│   │── students
│   ├── output_20241224_185026.jpg
│   ├── output_20241224_185546.jpg
│   ├── output_20241224_194451.jpg
│   ├── output_20241224_194601.jpg
│   ├── output_20241224_195828.jpg
│   ├── output_20241224_195847.jpg
│── logs
│── models
│   ├── dlib_face_recognition_resnet_model.dat
│   ├── face_recognition_model.pkl
│   ├── shape_predictor_68_face_landmarks.dat
│── src
│   ├── attendance_manager.py
│   ├── augmentation.py
│   ├── evaluation.py
│   ├── face_detection.py
│   ├── face_recognition.py
│   ├── utils.py
│   ├── config.py
│   ├── main.py
```

## Future Enhancements

- **Real-time video processing** for dynamic attendance.
- **User-friendly GUI** for easy interaction.
- **Cloud-based storage** for remote attendance tracking.

## Conclusion

This system offers an efficient, accurate, and scalable solution for automated attendance management. By leveraging facial recognition, it minimizes errors, saves time, and enhances security. With further improvements, it can be adapted to broader applications, including corporate and public sector environments.

