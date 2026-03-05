from ultralytics import YOLO

def train_wafer_model():
    print("Loading pre-trained YOLOv8 Nano model...")
    # We use the 'nano' version (yolov8n) because it is lightweight and fast to train
    model = YOLO('yolov8n.pt')

    print("Starting AI training sequence...")
    # This single command handles the entire neural network training process
    results = model.train(
        data='dataset.yaml',      # Pointing to the cheat sheet we just made
        epochs=10,                # Number of times it will study all 20,415 images
        imgsz=128,                # Resizing the small wafer maps to a standard AI size
        batch=32,                 # How many images it memorizes at the exact same time
        project='runs/wafer_defects', # The master folder where it saves its brain later
        name='yolov8_run1'        # The specific name of this training session
    )
    
    print("Training completely finished! Check the 'runs/' folder for the results.")

if __name__ == '__main__':
    train_wafer_model()