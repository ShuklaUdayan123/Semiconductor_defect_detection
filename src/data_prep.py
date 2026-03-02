import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2  # OpenCV for generating the images

# --- 1. THE FIX for old Pandas architecture ---
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

# --- 2. CONFIGURATION ---
# The path where Kaggle downloaded your dataset
RAW_DATA_PATH = os.path.expanduser('~/.cache/kagglehub/datasets/qingyi/wm811k-wafer-map/versions/1/LSWMD.pkl')

# Our output folders
IMAGE_DIR = 'data/processed/images'
LABEL_DIR = 'data/processed/labels'

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# YOLO needs integers (0-7), not text words for classes
CLASS_MAP = {
    'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
    'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7
}

def process_data():
    print("Loading dataset (this might take a minute)...")
    with open(RAW_DATA_PATH, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
        
    print("Cleaning data...")
    df['failure_class'] = df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else 'None')
    defective_wafers = df[(df['failure_class'] != 'None') & (df['failure_class'] != 'none')]
    
    print(f"Found {len(defective_wafers)} defective wafers.")
    print("Starting conversion on a test batch of 500 wafers...")
    
#processing the whole set now
    count = 0
    for index, row in defective_wafers.iterrows():
        wafer_map = row['waferMap']
        defect_class_text = row['failure_class']
        
        # Skip if the class isn't in our dictionary (safety check)
        if defect_class_text not in CLASS_MAP:
            continue
            
        class_id = CLASS_MAP[defect_class_text]
        
        # --- 3. BOUNDING BOX MATH ---
        # Find all Y (row) and X (col) coordinates where value is 2 (the defect)
        defect_y, defect_x = np.where(wafer_map == 2)
        
        # If there are no 2s for some reason, skip this wafer
        if len(defect_x) == 0:
            continue
            
        # Find the extreme edges
        xmin, xmax = np.min(defect_x), np.max(defect_x)
        ymin, ymax = np.min(defect_y), np.max(defect_y)
        
        # Total array dimensions
        img_height, img_width = wafer_map.shape
        
        # Calculate YOLO format (normalized 0.0 to 1.0)
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        w_yolo = (xmax - xmin) / img_width
        h_yolo = (ymax - ymin) / img_height
        
        # --- 4. GENERATE IMAGE ---
        # Map 0 -> Black (0), 1 -> Gray (127), 2 -> White (255)
        # We use a grayscale format because it gives the AI perfect contrast to see the defects
        img_array = np.zeros((img_height, img_width), dtype=np.uint8)
        img_array[wafer_map == 1] = 127
        img_array[wafer_map == 2] = 255
        
        # Save the image using OpenCV
        img_filename = os.path.join(IMAGE_DIR, f"wafer_{index}.jpg")
        cv2.imwrite(img_filename, img_array)
        
        # --- 5. GENERATE LABEL FILE ---
        label_filename = os.path.join(LABEL_DIR, f"wafer_{index}.txt")
        with open(label_filename, 'w') as f:
            # YOLO strictly requires: class_id x_center y_center width height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_yolo:.6f} {h_yolo:.6f}\n")
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} wafers...")
            
    print("Pipeline complete! Check your data/processed/ folders.")

if __name__ == "__main__":
    process_data()