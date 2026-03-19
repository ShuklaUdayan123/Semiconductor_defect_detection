"""
Robotic Control Middleware — Full Production Scan
Phase 1: Loads Mixed-type Wafer Defect Dataset, runs YOLOv8 on defective wafers.
Phase 2: Loads ALL passed wafers from WM-811K dataset (direct insert, no YOLO needed).
"""

import os
import sys
import pickle
import sqlite3
import random
import cv2
import time
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO

# Fix for old Pandas architecture in WM-811K pickle
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
import pandas as pd

# --- CONFIGURATION ---
NPZ_PATH = os.path.expanduser(
    '~/.cache/kagglehub/datasets/co1d7era/mixedtype-wafer-defect-datasets/versions/4/Wafer_Map_Datasets.npz'
)
WM811K_PATH = os.path.expanduser(
    '~/.cache/kagglehub/datasets/qingyi/wm811k-wafer-map/versions/1/LSWMD.pkl'
)
MODEL_PATH = 'runs/detect/runs/wafer_defects/yolov8_run1/weights/best.pt'
DB_PATH = os.path.join('middleware', 'wafer_control.db')

# Defect names matching the 8-dim one-hot encoding order in the dataset
DEFECT_NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']


def setup_database():
    """Creates a fresh wafer_logs table with ground_truth column."""
    os.makedirs('middleware', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS wafer_logs')
    cursor.execute('''
        CREATE TABLE wafer_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wafer_id TEXT,
            batch_id TEXT,
            scan_time TEXT,
            status TEXT,
            ground_truth TEXT,
            defect_type TEXT,
            action TEXT,
            confidence REAL,
            roi_coordinates TEXT,
            defect_area_px INTEGER,
            material_wasted_pct REAL
        )
    ''')
    conn.commit()
    return conn


def decode_label(one_hot):
    """Convert 8-dim one-hot label to human-readable defect string."""
    active = np.where(one_hot == 1)[0]
    if len(active) == 0:
        return 'Normal'
    return '+'.join([DEFECT_NAMES[i] for i in active])


def wafer_to_image(wafer_map):
    """Convert a 52x52 wafer map array to a 3-channel BGR image for YOLOv8."""
    img = np.zeros(wafer_map.shape, dtype=np.uint8)
    img[wafer_map == 1] = 127   # Normal die → gray
    img[wafer_map == 2] = 255   # Broken die → white
    img[wafer_map == 3] = 255   # Treat 3 as defect too (rare edge artifact)
    # YOLOv8 expects 3-channel (BGR) images
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr


def compute_defect_area(coords):
    """Calculate bounding box area in pixels from [x1, y1, x2, y2]."""
    if not coords or len(coords) != 4:
        return 0
    x1, y1, x2, y2 = coords
    return max(0, (x2 - x1) * (y2 - y1))


def run_production_scan(conn, model, wafer_maps, labels, batch_id, start_time):
    """Process all wafers: YOLO inference on defective, direct insert for normals."""
    cursor = conn.cursor()
    total = len(wafer_maps)

    for i in range(total):
        wafer_id = f"wafer_{i}"
        ground_truth = decode_label(labels[i])

        # Simulate realistic timestamps spread across 30 days
        scan_time = start_time + timedelta(
            days=i // 1267,             # ~1,267 wafers per day over 30 days
            seconds=random.randint(0, 68)
        )
        scan_time_str = scan_time.strftime("%Y-%m-%d %H:%M:%S")

        if ground_truth == 'Normal':
            # PASS wafer — no YOLO needed
            cursor.execute('''
                INSERT INTO wafer_logs 
                (wafer_id, batch_id, scan_time, status, ground_truth, defect_type, action, confidence, roi_coordinates, defect_area_px, material_wasted_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (wafer_id, batch_id, scan_time_str, "PASS", ground_truth, "None", "ROUTE_TO_ASSEMBLY", 1.0, "[]", 0, 0.0))
        else:
            # Defective wafer — convert to image and run YOLO
            img = wafer_to_image(wafer_maps[i])
            wafer_area_px = img.shape[0] * img.shape[1]  # 52*52 = 2704

            results = model.predict(source=img, conf=0.25, verbose=False)
            boxes = results[0].boxes

            if len(boxes) > 0:
                box = boxes[0]
                class_id = int(box.cls[0].item())
                defect_type = model.names[class_id]
                confidence = round(box.conf[0].item(), 2)

                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                coords = [x1, y1, x2, y2]

                status = "FAIL"
                action = "ROUTE_TO_SCRAP" if defect_type in ["Center", "Near-full"] else "MOVE_TO_MICRO_STAGE"
            else:
                # YOLO didn't detect anything (could be mixed pattern it can't see)
                status = "FAIL"
                defect_type = "Undetected"
                action = "MOVE_TO_MICRO_STAGE"
                confidence = 0.0
                coords = []

            defect_area = compute_defect_area(coords)
            material_wasted_pct = round((defect_area / wafer_area_px) * 100, 2) if defect_area > 0 else 0.0

            cursor.execute('''
                INSERT INTO wafer_logs 
                (wafer_id, batch_id, scan_time, status, ground_truth, defect_type, action, confidence, roi_coordinates, defect_area_px, material_wasted_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (wafer_id, batch_id, scan_time_str, status, ground_truth, defect_type, action, confidence, str(coords), defect_area, material_wasted_pct))

        # Commit in batches
        if (i + 1) % 500 == 0:
            conn.commit()
            print(f"  Processed {i + 1}/{total} wafers...")

    conn.commit()


def insert_wm811k_passed(conn, batch_id, start_time):
    """Load ALL passed wafers from WM-811K and insert directly into DB."""
    print("Loading WM-811K dataset...")
    with open(WM811K_PATH, 'rb') as f:
        wm_df = pickle.load(f, encoding='latin1')

    wm_df['failure_class'] = wm_df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else 'None')
    passed = wm_df[(wm_df['failure_class'] == 'None') | (wm_df['failure_class'] == 'none')]
    passed = passed[passed['waferMap'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)]

    total = len(passed)
    print(f"  Found {total:,} passed wafers in WM-811K")
    print(f"  Inserting all into database...")

    cursor = conn.cursor()
    rows = []
    for i, (index, row) in enumerate(passed.iterrows()):
        scan_time = start_time + timedelta(
            days=i // 26200,   # spread across 30 days
            seconds=random.randint(0, 3)
        )
        rows.append((
            f"wm811k_{index}", batch_id, scan_time.strftime("%Y-%m-%d %H:%M:%S"),
            "PASS", "Normal", "None", "ROUTE_TO_ASSEMBLY", 1.0, "[]", 0, 0.0
        ))

        if len(rows) >= 10000:
            cursor.executemany('''
                INSERT INTO wafer_logs
                (wafer_id, batch_id, scan_time, status, ground_truth, defect_type, action, confidence, roi_coordinates, defect_area_px, material_wasted_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)
            conn.commit()
            rows = []
            print(f"  Inserted {i + 1:,}/{total:,} passed wafers...")

    if rows:
        cursor.executemany('''
            INSERT INTO wafer_logs
            (wafer_id, batch_id, scan_time, status, ground_truth, defect_type, action, confidence, roi_coordinates, defect_area_px, material_wasted_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        conn.commit()

    print(f"  Done! Inserted {total:,} passed wafers.")
    return total


if __name__ == '__main__':
    print("=" * 60)
    print("  ROBOTIC CONTROL MIDDLEWARE — HYBRID PRODUCTION SCAN")
    print("=" * 60)

    # 1. Load Mixed-type dataset
    print("\nPhase 1: Loading Mixed-type Wafer Defect Dataset...")
    data = np.load(NPZ_PATH)
    X = data['arr_0']
    Y = data['arr_1']

    normals_mixed = sum(1 for y in Y if np.sum(y) == 0)
    defective = len(X) - normals_mixed

    print(f"  Mixed-type: {len(X):,} total ({defective:,} defective + {normals_mixed:,} normal)")
    print(f"  WM-811K:    ~786K passed wafers")

    # 2. Setup
    db_connection = setup_database()
    wafer_model = YOLO(MODEL_PATH)

    batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = datetime.now() - timedelta(days=30)

    print(f"\n  Batch ID: {batch_id}")
    print(f"  Scan window: {start_time.strftime('%Y-%m-%d')} → {datetime.now().strftime('%Y-%m-%d')}")

    try:
        # PHASE 1: YOLOv8 on Mixed-type defective wafers
        print(f"\n{'=' * 60}")
        print(f"  PHASE 1: YOLOv8 Inference ({len(X):,} Mixed-type wafers)")
        print(f"{'=' * 60}\n")

        t0 = time.time()
        run_production_scan(db_connection, wafer_model, X, Y, batch_id, start_time)
        t1 = time.time()
        print(f"\n  Phase 1 complete: {t1 - t0:.1f}s")

        # PHASE 2: All passed wafers from WM-811K
        print(f"\n{'=' * 60}")
        print(f"  PHASE 2: WM-811K Passed Wafers (all ~786K)")
        print(f"{'=' * 60}\n")

        passed_count = insert_wm811k_passed(db_connection, batch_id, start_time)
        t2 = time.time()

        total = len(X) + passed_count
        print(f"\n{'=' * 60}")
        print(f"  SCAN COMPLETE")
        print(f"  Defective (Mixed-type): {defective:,}")
        print(f"  Normal (Mixed-type):    {normals_mixed:,}")
        print(f"  Passed (WM-811K):       {passed_count:,}")
        print(f"  Total records:          {total:,}")
        print(f"  Pass rate:              {(normals_mixed + passed_count) / total * 100:.1f}%")
        print(f"  Time elapsed:           {t2 - t0:.1f}s")
        print(f"  Database:               {DB_PATH}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\nError during scan: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db_connection.close()
        print("Database connection closed.")