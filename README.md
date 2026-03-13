# Semiconductor Wafer Defect Detection: End-to-End YOLOv8 Pipeline

## Project Overview
This project is a complete, end-to-end Applied Computer Vision pipeline designed for the semiconductor manufacturing industry. It takes raw, legacy mathematical array data representing 25,000+ defective semiconductor wafers, engineers them into an AI-ready computer vision dataset, and trains a custom YOLOv8 object detection model to automatically identify and classify 8 distinct types of manufacturing defects.

**Final Model Performance:** `0.962 mAP@50` (96.2% overall accuracy on unseen validation data).

## Business Value
In semiconductor fabrication, identifying microscopic defects early in the manufacturing process saves millions in scrapped materials. This project automates quality control by transitioning from manual coordinate analysis to real-time, AI-driven visual defect detection.

## The Technical Pipeline

### Phase 1: Data Engineering (`src/data_prep.py`)
* **The Challenge:** The original dataset consisted of raw `.txt` files containing numeric 2D arrays (0=background, 1=good chip, 2=defect). YOLOv8 cannot read text arrays; it requires physical images and normalized bounding box coordinates.
* **The Solution:** Built a custom Python pipeline using `NumPy` and `OpenCV` to parse over 25,000 text files. 
* **The Math:** Programmatically identified the spatial extremes (`xmin`, `ymin`, `xmax`, `ymax`) of the `2` values, normalized them to YOLO's strict `0.0 - 1.0` format, and dynamically rendered high-contrast `.jpg` images alongside corresponding `.txt` label files.

### Phase 2: Dataset Architecture (`src/split_data.py`)
* Used `scikit-learn` to execute a mathematically rigorous 80/20 train/validation split.
* Programmatically generated the strict directory architecture required by YOLO, migrating over 50,000 individual files into structured `train` and `val` directories.

### Phase 3: Model Training (`src/model_train.py`)
* Initialized a pre-trained **YOLOv8 Nano** (`yolov8n.pt`) model for lightweight, high-speed inference.
* Trained on 20,415 wafer images for 10 epochs.
* Mapped 8 specific manufacturing defect classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full).

### Phase 4: Batch Inference & Evaluation (`src/batch_inference.py` & `src/model_eval.py`)
* Deployed the custom-trained `best.pt` weights to run batch inference on 5,104 unseen validation images.
* Model successfully drew accurate bounding boxes and assigned confidence scores entirely autonomously.

## Performance Metrics
The model achieved phenomenal results on the blind 5,104-image validation set:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **mAP50 (All Classes)** | **96.2%** | Overall model accuracy at a 50% confidence threshold. |
| **Recall** | **93.1%** | The model successfully located 93.1% of all physical defects. |
| **Edge-Ring (mAP50)** | **99.4%** | Near-flawless detection of Edge-Ring anomalies. |
| **Near-full (mAP50)** | **94.4%** | Highly accurate even on severely underrepresented edge-case classes. |

*(Note: Visual evidence of bounding box predictions, Precision-Recall curves, and Confusion Matrices can be found in the `runs/` directory).*

## Tech Stack
* **Languages:** Python
* **Computer Vision:** Ultralytics (YOLOv8), OpenCV (`cv2`)
* **Data Engineering:** Pandas, NumPy, Scikit-learn
* **Version Control:** Git

---
*Designed and engineered by Udayan Shashank Shukla.