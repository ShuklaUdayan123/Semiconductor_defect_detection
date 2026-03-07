from ultralytics import YOLO

def test_all_validation_images():
    print("Loading your custom-trained wafer brain...")
    # Pointing to your exact model path from earlier
    model_path = 'runs/detect/runs/wafer_defects/yolov8_run1/weights/best.pt'
    model = YOLO(model_path)

    print("Running inference on ALL validation images...")
    # Instead of one image, we hand it the entire validation folder
    val_dir = 'data/yolo_dataset/images/val'

    # The AI will automatically loop through all 5,000+ images!
    results = model.predict(source=val_dir, save=True, conf=0.25)
    
    print("\nMassive inference complete! Look in the newest 'predict' folder to see thousands of drawn bounding boxes.")

if __name__ == '__main__':
    test_all_validation_images()