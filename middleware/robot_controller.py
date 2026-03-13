import os
import json
import sqlite3
import random
import cv2
from datetime import datetime
from ultralytics import YOLO

def setup_database():
    # Connect to a local SQLite database inside the middleware folder
    db_path = os.path.join('middleware', 'wafer_control.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the SQL table for our robotic logs
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

def run_robotic_simulation():
    print("Initializing Robotic Control Middleware...")
    conn = setup_database()
    cursor = conn.cursor()

    # Load your custom-trained brain
    model_path = 'runs/detect/runs/wafer_defects/yolov8_run1/weights/best.pt'
    model = YOLO(model_path)

    # Grab a random wafer from the validation set
    val_dir = 'data/yolo_dataset/images/val'
    val_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    test_img = os.path.join(val_dir, random.choice(val_images))
    wafer_id = os.path.basename(test_img).split('.')[0]

    print(f"\nScanning {wafer_id} on the assembly line...")
    # Run the prediction
    results = model.predict(source=test_img, conf=0.25, verbose=False)
    boxes = results[0].boxes

    # --- THE LOGIC GATES ---
    if len(boxes) > 0:
        box = boxes[0]
        class_id = int(box.cls[0].item())
        defect_type = model.names[class_id]
        confidence = round(box.conf[0].item(), 2)
        
        # PHASE 2: EXTRACT THE ROI
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()] 
        coords = [x1, y1, x2, y2]
        
        os.makedirs("Output_ROI", exist_ok=True)
        img_array = cv2.imread(test_img)
        cropped_roi = img_array[y1:y2, x1:x2]
        
        if cropped_roi.size > 0:
            roi_filename = f"Output_ROI/{wafer_id}_ROI_{defect_type}.jpg"
            cv2.imwrite(roi_filename, cropped_roi)
            print(f"Successfully cropped {defect_type} ROI and saved to {roi_filename}")

        # PHASE 3: SIMULATE ROBOTIC COMMAND
        status = "FAIL"
        action = "ROUTE_TO_SCRAP" if defect_type in ["Center", "Near-full"] else "MOVE_TO_MICRO_STAGE"
        
    else:
        status = "PASS"
        defect_type = "None"
        action = "ROUTE_TO_ASSEMBLY"
        confidence = 1.0
        coords = []

    # Package the JSON Payload
    payload = {
        "wafer_id": wafer_id,
        "status": status,
        "defect_type": defect_type,
        "action": action,
        "confidence": confidence,
        "coordinates": coords
    }

    print("\nTRANSMITTING JSON PAYLOAD TO ROBOTIC ARM:")
    print(json.dumps(payload, indent=4))

    # Log it into the SQL Database
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO wafer_logs (wafer_id, scan_time, status, defect_type, action, confidence, roi_coordinates)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (wafer_id, scan_time, status, defect_type, action, confidence, str(coords)))
    
    conn.commit()
    conn.close()
    print("\nData successfully logged to local SQL database (middleware/wafer_control.db).")

if __name__ == '__main__':
    run_robotic_simulation()