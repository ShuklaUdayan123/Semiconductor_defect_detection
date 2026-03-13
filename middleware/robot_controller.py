import os
import json
import sqlite3
import random
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

def setup_database():
    os.makedirs('middleware', exist_ok=True)
    db_path = os.path.join('middleware', 'wafer_control.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wafer_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wafer_id TEXT,
            scan_time TEXT,
            status TEXT,
            defect_type TEXT,
            action TEXT,
            confidence REAL,
            roi_coordinates TEXT
        )
    ''')
    conn.commit()
    return conn

def run_single_scan(conn, model):
    cursor = conn.cursor()

    # Grab a random wafer from the validation set
    val_dir = 'data/yolo_dataset/images/val'
    val_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    
    if not val_images:
        print("Error: No images found in validation directory.")
        return

    test_img = os.path.join(val_dir, random.choice(val_images))
    wafer_id = os.path.basename(test_img).split('.')[0]

    # Run the prediction
    results = model.predict(source=test_img, conf=0.25, verbose=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        box = boxes[0]
        class_id = int(box.cls[0].item())
        defect_type = model.names[class_id]
        confidence = round(box.conf[0].item(), 2)
        
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()] 
        coords = [x1, y1, x2, y2]
        
        os.makedirs("Output_ROI", exist_ok=True)
        img_array = cv2.imread(test_img)
        cropped_roi = img_array[y1:y2, x1:x2]
        
        if cropped_roi.size > 0:
            roi_filename = f"Output_ROI/{wafer_id}_ROI_{defect_type}.jpg"
            cv2.imwrite(roi_filename, cropped_roi)

        status = "FAIL"
        action = "ROUTE_TO_SCRAP" if defect_type in ["Center", "Near-full"] else "MOVE_TO_MICRO_STAGE"
    else:
        status = "PASS"
        defect_type = "None"
        action = "ROUTE_TO_ASSEMBLY"
        confidence = 1.0
        coords = []

    # Log to Database
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO wafer_logs (wafer_id, scan_time, status, defect_type, action, confidence, roi_coordinates)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (wafer_id, scan_time, status, defect_type, action, confidence, str(coords)))
    
    conn.commit()
    print(f"Processed {wafer_id}: {status} ({defect_type}) -> {action}")

if __name__ == '__main__':
    print("Initializing Robotic Control Middleware...")
    
    # 1. Setup Resources Once
    db_connection = setup_database()
    model_path = 'runs/detect/runs/wafer_defects/yolov8_run1/weights/best.pt'
    wafer_model = YOLO(model_path)

    try:
        # 2. Loop 10 Times
        for i in range(1, 11):
            print(f"\n--- Scanning Wafer {i}/10 ---")
            run_single_scan(db_connection, wafer_model)
            
            # Optional: Small delay to simulate assembly line movement
            time.sleep(0.5) 

    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        
    finally:
        # 3. Clean up
        db_connection.close()
        print("\nSimulation complete. Database connection closed.")