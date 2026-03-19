"""
Material Predictor — Phase 5
Trains a Random Forest model on historical wafer scan data to predict
material waste percentage for future production batches.
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'wafer_control.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'material_model.pkl')

# Defect types for feature engineering
DEFECT_TYPES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'None', 'Undetected']


def load_data():
    """Load wafer logs from the SQLite database into a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM wafer_logs", conn)
    conn.close()
    df['scan_time'] = pd.to_datetime(df['scan_time'])
    return df


def engineer_features(df):
    """
    Build daily-aggregated features from raw scan logs.
    Each row = one day of production with aggregated metrics.
    """
    df['scan_date'] = df['scan_time'].dt.date
    df['is_fail'] = (df['status'] == 'FAIL').astype(int)
    df['is_scrap'] = (df['action'] == 'ROUTE_TO_SCRAP').astype(int)

    # One-hot encode defect types per scan
    for defect in DEFECT_TYPES:
        col_name = f'is_{defect.lower().replace("-", "_")}'
        df[col_name] = (df['defect_type'] == defect).astype(int)

    # --- Aggregate by day ---
    daily = df.groupby('scan_date').agg(
        total_scans=('id', 'count'),
        fail_count=('is_fail', 'sum'),
        scrap_count=('is_scrap', 'sum'),
        avg_confidence=('confidence', 'mean'),
        avg_defect_area=('defect_area_px', 'mean'),
        max_defect_area=('defect_area_px', 'max'),
        total_waste_pct=('material_wasted_pct', 'sum'),
        avg_waste_pct=('material_wasted_pct', 'mean'),
        # Defect type counts per day
        center_count=('is_center', 'sum'),
        donut_count=('is_donut', 'sum'),
        edge_loc_count=('is_edge_loc', 'sum'),
        edge_ring_count=('is_edge_ring', 'sum'),
        loc_count=('is_loc', 'sum'),
        random_count=('is_random', 'sum'),
        scratch_count=('is_scratch', 'sum'),
        near_full_count=('is_near_full', 'sum'),
        pass_count=('is_none', 'sum'),
    ).reset_index()

    # --- Compute waste among defective wafers only ---
    defective_daily = df[df['status'] == 'FAIL'].groupby('scan_date').agg(
        avg_waste_defective=('material_wasted_pct', 'mean'),
        avg_defect_area_fail=('defect_area_px', 'mean'),
        avg_confidence_fail=('confidence', 'mean'),
    ).reset_index()

    daily = daily.merge(defective_daily, on='scan_date', how='left')
    daily['avg_waste_defective'] = daily['avg_waste_defective'].fillna(0)
    daily['avg_defect_area_fail'] = daily['avg_defect_area_fail'].fillna(0)
    daily['avg_confidence_fail'] = daily['avg_confidence_fail'].fillna(0)

    # Derived ratios
    daily['fail_rate'] = daily['fail_count'] / daily['total_scans']
    daily['scrap_rate'] = daily['scrap_count'] / daily['total_scans']

    # Time features
    daily['scan_date'] = pd.to_datetime(daily['scan_date'])
    daily['day_of_week'] = daily['scan_date'].dt.dayofweek
    daily['day_index'] = (daily['scan_date'] - daily['scan_date'].min()).dt.days

    return daily


def train_model(daily):
    """Train a Random Forest to predict avg material waste among defective wafers."""
    feature_cols = [
        'total_scans', 'fail_count', 'scrap_count', 'avg_confidence',
        'avg_defect_area', 'max_defect_area', 'fail_rate', 'scrap_rate',
        'avg_defect_area_fail', 'avg_confidence_fail',
        'center_count', 'donut_count', 'edge_loc_count', 'edge_ring_count',
        'loc_count', 'random_count', 'scratch_count', 'near_full_count',
        'pass_count', 'day_of_week', 'day_index'
    ]

    target = 'avg_waste_defective'

    X = daily[feature_cols]
    y = daily[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  MODEL EVALUATION (target: avg waste % per wafer)")
    print(f"  Mean Absolute Error:  {mae:.2f}%")
    print(f"  R² Score:             {r2:.4f}")
    print(f"{'=' * 50}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top 5 Feature Importances:")
    for feat, imp in importance.head(5).items():
        print(f"    {feat:25s} {imp:.4f}")

    return model, feature_cols, {'mae': mae, 'r2': r2, 'importances': importance.to_dict()}


def predict_material_needs(model, feature_cols, total_scans, fail_rate, defect_distribution):
    """
    Predict material waste for a hypothetical future production day.
    """
    fail_count = int(total_scans * fail_rate)
    pass_count = total_scans - fail_count

    features = {
        'total_scans': total_scans,
        'fail_count': fail_count,
        'scrap_count': int(fail_count * defect_distribution.get('Center', 0) +
                          fail_count * defect_distribution.get('Near-full', 0)),
        'avg_confidence': 0.95,
        'avg_defect_area': 1500,
        'max_defect_area': 2704,
        'fail_rate': fail_rate,
        'scrap_rate': defect_distribution.get('Center', 0) + defect_distribution.get('Near-full', 0),
        'avg_defect_area_fail': 1500,
        'avg_confidence_fail': 0.85,
        'center_count': int(fail_count * defect_distribution.get('Center', 0)),
        'donut_count': int(fail_count * defect_distribution.get('Donut', 0)),
        'edge_loc_count': int(fail_count * defect_distribution.get('Edge-Loc', 0)),
        'edge_ring_count': int(fail_count * defect_distribution.get('Edge-Ring', 0)),
        'loc_count': int(fail_count * defect_distribution.get('Loc', 0)),
        'random_count': int(fail_count * defect_distribution.get('Random', 0)),
        'scratch_count': int(fail_count * defect_distribution.get('Scratch', 0)),
        'near_full_count': int(fail_count * defect_distribution.get('Near-full', 0)),
        'pass_count': pass_count,
        'day_of_week': 2,
        'day_index': 30,
    }

    X = pd.DataFrame([features])[feature_cols]
    avg_waste_per_wafer = model.predict(X)[0]
    total_waste_wafers = (avg_waste_per_wafer / 100.0) * fail_count
    return {
        'avg_waste_per_wafer': round(avg_waste_per_wafer, 2),
        'total_daily_waste': round(total_waste_wafers, 1),
        'total_scans': total_scans,
        'fail_rate': fail_rate,
    }


if __name__ == '__main__':
    print("=" * 50)
    print("  MATERIAL WASTE PREDICTOR — Training")
    print("=" * 50)

    # 1. Load and engineer features
    print("\nLoading scan data...")
    raw_df = load_data()
    print(f"  Total records: {len(raw_df)}")
    print(f"  PASS: {len(raw_df[raw_df['status'] == 'PASS'])}")
    print(f"  FAIL: {len(raw_df[raw_df['status'] == 'FAIL'])}")

    print("Engineering daily features...")
    daily_df = engineer_features(raw_df)
    print(f"  Training days: {len(daily_df)}")

    # 2. Train
    print("\nTraining Random Forest model...")
    trained_model, feat_cols, metrics = train_model(daily_df)

    # 3. Save
    model_package = {
        'model': trained_model,
        'feature_cols': feat_cols,
        'metrics': metrics,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to: {MODEL_PATH}")

    # 4. Demo prediction
    print(f"\n{'=' * 50}")
    print("  DEMO PREDICTION")
    print(f"{'=' * 50}")

    demo_distribution = {
        'Center': 0.15, 'Edge-Ring': 0.37, 'Edge-Loc': 0.06,
        'Donut': 0.23, 'Random': 0.03, 'Scratch': 0.03,
        'Loc': 0.10, 'Near-full': 0.01
    }

    pred = predict_material_needs(trained_model, feat_cols,
                                   total_scans=1300, fail_rate=0.97,
                                   defect_distribution=demo_distribution)
    print(f"  Scenario: 1,300 wafers/day, 97% defect rate")
    print(f"  Predicted avg waste per wafer: {pred['avg_waste_per_wafer']:.2f}%")
    print(f"  Predicted total daily waste:   {pred['total_daily_waste']:.1f} equivalent wafers")
    print(f"{'=' * 50}")
