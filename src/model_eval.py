from ultralytics import YOLO

def evaluate_model():
    print("Loading your custom-trained wafer brain for official grading...")
    # Pointing to your exact model path
    model_path = 'runs/detect/runs/wafer_defects/yolov8_run1/weights/best.pt'
    model = YOLO(model_path)

    print("Running official validation...")
    # model.val() automatically uses dataset.yaml to find the images AND the answer keys!
    metrics = model.val()
    
    print("\nGrading complete! Check the new 'runs/detect/val' folder for your fresh graphs.")

if __name__ == '__main__':
    evaluate_model()