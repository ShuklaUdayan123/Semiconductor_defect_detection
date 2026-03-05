import os
import random
from ultralytics import YOLO

def test_model():
    print("Loading your custom-trained wafer brain...")
    # Updated path to match your exact file explorer structure
    model_path = 'runs/detect/runs/wafer_defects/yolov8_run1/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}.")
        return
        
    model = YOLO(model_path)

    print("Picking a random unseen wafer from the validation set...")
    val_dir = 'data/yolo_dataset/images/val'
    val_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    
    # Grab one random image
    test_img = os.path.join(val_dir, random.choice(val_images))
    print(f"Running inference on: {test_img}")

    # The magic command: predict() 
    # conf=0.25 means the AI will only draw a box if it is at least 25% confident
    results = model.predict(source=test_img, save=True, conf=0.25)
    
    print("\nInference complete! Look in the newly generated 'predict' folder to see the drawn bounding box.")

if __name__ == '__main__':
    test_model()    