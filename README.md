# Semiconductor Wafer Defect Detection: End-to-End AI Pipeline

## Project Overview
This project is a complete, end-to-end Applied AI pipeline designed for the semiconductor manufacturing industry. It takes raw mathematical array data representing defective semiconductor wafers, engineers them into an AI-ready computer vision dataset, trains a custom YOLOv8 object detection model, and feeds the results into a predictive material waste model and real-time dashboard.

**Final YOLOv8 Model Performance:** `0.962 mAP@50` (96.2% overall accuracy on unseen validation data).
**Predictive Waste Model Performance:** `R² = 0.9637` (Highly accurate material waste prediction).

## Business Value
In semiconductor fabrication, identifying microscopic defects early in the manufacturing process saves millions in scrapped materials. This project automates quality control by transitioning from manual coordinate analysis to real-time, AI-driven visual defect detection, while simultaneously forecasting future material waste to optimize supply chain planning.

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
* Deployed the custom-trained `best.pt` weights to run batch inference on unseen validation images.
* Model successfully drew accurate bounding boxes and assigned confidence scores entirely autonomously.

### Phase 5: Production Middleware, Predictive Modeling & Dashboard
* **Robotic Scanner Simulation (`middleware/robot_controller.py`):** Operates on a massive hybrid dataset of **823,953 wafers** (Mixed-type + WM-811K datasets) with a realistic 95.5% pass rate. It automatically routes passed wafers and runs YOLOv8 inference on defective ones, logging everything into a centralized SQLite database (`wafer_control.db`).
* **Material Waste Predictor (`middleware/material_predictor.py`):** A Random Forest Regressor trained on the historical scan database. It accurately predicts the average percentage of material wasted within defective wafers, allowing fabs to estimate future material needs.
* **Real-time Dashboard (`middleware/dashboard.py`):** A **Plotly Dash** web application that visualizes historical defect rates, defect distributions, routing actions, and integrates interactive material forecasting inputs.

## Upcoming Feature: LLM Troubleshooting Assistant (Planned)
**Goal:** Integrate an intelligent Large Language Model (LLM) bot to assist fab engineers directly on the factory floor.
* **Functionality:** When the dashboard flags a sudden spike in a specific defect type (e.g., "Edge-Ring" defects), the engineer can consult the LLM bot.
* **Use Case:** The bot will analyze the defect trends, cross-reference historical manufacturing guidelines, and suggest potential root causes (such as misaligned etching tools or incorrect gas pressure), drastically reducing troubleshooting and downtime.
*(Note: This feature is currently in the design phase and not yet implemented).*

## Performance Metrics
The YOLOv8 model achieved phenomenal results on the blind validation set:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **mAP50 (All Classes)** | **96.2%** | Overall model accuracy at a 50% confidence threshold. |
| **Recall** | **93.1%** | The model successfully located 93.1% of all physical defects. |
| **Edge-Ring (mAP50)** | **99.4%** | Near-flawless detection of Edge-Ring anomalies. |

The Random Forest Material Waste Predictor achieved:
| Metric | Score | Note |
| :--- | :--- | :--- |
| **R² Score** | **0.9637** | Excellent correlation on predictive targets. |
| **MAE** | **0.09%** | Average prediction error is less than one-tenth of a percent. |

## Tech Stack
* **Languages:** Python
* **Computer Vision:** Ultralytics (YOLOv8), OpenCV (`cv2`)
* **Machine Learning & Data:** Pandas, NumPy, Scikit-learn, SQLite
* **Web UI & Visualization:** Plotly, Dash

---
*Designed and engineered by Udayan Shashank Shukla.*