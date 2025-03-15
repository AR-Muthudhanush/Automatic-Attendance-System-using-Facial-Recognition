import cv2
import numpy as np
from pathlib import Path
import albumentations as A

class DataAugmentor:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5)
        ])
    
    def augment_dataset(self):
        for student_dir in self.input_dir.glob("*"):
            if not student_dir.is_dir():
                continue
                
            output_student_dir = self.output_dir / student_dir.name
            output_student_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in student_dir.glob("*.jpg"):
                self._augment_image(img_path, output_student_dir)
    
    def _augment_image(self, img_path: Path, output_dir: Path):
        image = cv2.imread(str(img_path))
        if image is None:
            return
            
        # Save original image
        cv2.imwrite(str(output_dir / f"{img_path.stem}_original.jpg"), image)
        
        # Create augmented versions
        for i in range(5):
            augmented = self.transform(image=image)['image']
            output_path = output_dir / f"{img_path.stem}_aug_{i}.jpg"
            cv2.imwrite(str(output_path), augmented)