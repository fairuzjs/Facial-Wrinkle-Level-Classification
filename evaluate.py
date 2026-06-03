"""
evaluate.py (v2)
-----------------------------------------------------------------------------
Evaluasi akurasi sistem deteksi kerutan pada dataset Images/.

Cara pakai:
    python evaluate.py                  # Proses semua gambar
    python evaluate.py --max 100        # Proses hanya 100 gambar pertama
    python evaluate.py --output out.csv # Simpan ke file CSV custom

Output:
    - Ringkasan metrik di konsol (Accuracy, F1, MAE, Confusion Matrix)
    - File CSV: evaluation_results.csv (per-image results)
-----------------------------------------------------------------------------
"""

import cv2
import os
import re
import csv
import sys
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wrinkle_engine import WrinkleEngine, AgeEstimator

# -----------------------------------------------------------------------------
# Konstanta
# -----------------------------------------------------------------------------

IMAGES_DIR      = 'Images'
DEFAULT_OUTPUT  = 'evaluation_results.csv'
FACE_CASCADE    = 'haarcascade_frontalface_default.xml'

CATEGORIES = ['Muda', 'Paruh Baya', 'Tua']


def age_to_category(age: int) -> str:
    if age < 30:
        return 'Muda'
    elif age < 55:
        return 'Paruh Baya'
    else:
        return 'Tua'


CAFFE_MIDPOINTS = {
    '(0-2)':    1,
    '(4-6)':    5,
    '(8-12)':  10,
    '(15-20)': 17,
    '(25-32)': 28,
    '(38-43)': 40,
    '(48-53)': 50,
    '(60-100)':70,
}


# -----------------------------------------------------------------------------
# Parsing nama file dataset
# -----------------------------------------------------------------------------

FILENAME_RE = re.compile(r'^.+_(\d{4})-(\d{2})-(\d{2})_(\d{4})\.jpg$')


def parse_filename(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    birth_year = int(m.group(1))
    photo_year = int(m.group(4))
    age = photo_year - birth_year
    return age if 0 <= age <= 120 else None


# -----------------------------------------------------------------------------
# Deteksi wajah
# -----------------------------------------------------------------------------

def detect_face(image: np.ndarray, cascade) -> np.ndarray:
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return image[y:y+h, x:x+w]


# -----------------------------------------------------------------------------
# Fungsi Evaluasi Utama
# -----------------------------------------------------------------------------

def evaluate(max_images: int, output_csv: str):
    print("=" * 65)
    print("  EVALUASI SISTEM DETEKSI KERUTAN WAJAH v2")
    print("=" * 65)

    cascade = cv2.CascadeClassifier(FACE_CASCADE)
    if cascade.empty():
        print("[ERROR] Gagal memuat Haar Cascade.")
        return

    engine        = WrinkleEngine(calibration_path='wrinkle_calibration.json')
    age_estimator = AgeEstimator()
    print()

    all_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
    all_files.sort()
    if max_images > 0:
        all_files = all_files[:max_images]

    print(f"Memproses {len(all_files)} gambar dari '{IMAGES_DIR}/'...")
    print("-" * 65)

    rows            = []
    face_not_found  = 0
    parse_failed    = 0

    true_ages         = []
    caffe_pred_ages   = []
    wrinkle_cats_true = []
    wrinkle_cats_pred = []
    engine_age_preds  = []
    classifier_methods = []

    for i, filename in enumerate(all_files):
        img_path = os.path.join(IMAGES_DIR, filename)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(all_files)}] {filename}")

        true_age = parse_filename(filename)
        if true_age is None:
            parse_failed += 1
            rows.append({'filename': filename, 'true_age': '', 'status': 'parse_failed'})
            continue

        image = cv2.imread(img_path)
        if image is None:
            rows.append({'filename': filename, 'true_age': true_age, 'status': 'read_failed'})
            continue

        face = detect_face(image, cascade)
        if face is None:
            face_not_found += 1
            rows.append({'filename': filename, 'true_age': true_age, 'status': 'no_face'})
            continue

        wrinkle_result = engine.analyze(face)
        age_result     = age_estimator.predict(face)
        gt_category    = age_to_category(true_age)
        caffe_age      = CAFFE_MIDPOINTS.get(age_result.age_label, 0)

        # Confidence ML (jika tersedia)
        ml_conf_str = ''
        if wrinkle_result.ml_proba is not None:
            idx     = CATEGORIES.index(wrinkle_result.age_category)
            ml_conf = wrinkle_result.ml_proba[idx] * 100
            ml_conf_str = f"{ml_conf:.1f}"

        row = {
            'filename':           filename,
            'true_age':           true_age,
            'gt_category':        gt_category,
            'wrinkle_pct':        round(wrinkle_result.avg_wrinkle_pct, 2),
            'wrinkle_mata_kiri':  round(wrinkle_result.per_region_pct[0], 2),
            'wrinkle_mata_kanan': round(wrinkle_result.per_region_pct[1], 2),
            'wrinkle_dahi':       round(wrinkle_result.per_region_pct[2], 2),
            'wrinkle_pipi_kiri':  round(wrinkle_result.per_region_pct[3], 2),
            'wrinkle_pipi_kanan': round(wrinkle_result.per_region_pct[4], 2),
            'pred_category':      wrinkle_result.age_category,
            'classifier_method':  wrinkle_result.classifier_used,
            'ml_confidence_pct':  ml_conf_str,
            'caffe_age_label':    age_result.age_label,
            'caffe_confidence':   round(age_result.confidence * 100, 1),
            'caffe_mid_age':      caffe_age,
            'engine_est_age':     (round(wrinkle_result.estimated_age, 1)
                                   if wrinkle_result.estimated_age else ''),
            'used_mediapipe':     wrinkle_result.used_mediapipe,
            'status':             'ok',
        }
        rows.append(row)

        true_ages.append(true_age)
        caffe_pred_ages.append(caffe_age)
        wrinkle_cats_true.append(gt_category)
        wrinkle_cats_pred.append(wrinkle_result.age_category)
        classifier_methods.append(wrinkle_result.classifier_used)
        if wrinkle_result.estimated_age is not None:
            engine_age_preds.append((true_age, wrinkle_result.estimated_age))

    # -- Simpan CSV -----------------------------------------------------------
    fieldnames = [
        'filename', 'true_age', 'gt_category',
        'wrinkle_pct', 'wrinkle_mata_kiri', 'wrinkle_mata_kanan',
        'wrinkle_dahi', 'wrinkle_pipi_kiri', 'wrinkle_pipi_kanan',
        'pred_category', 'classifier_method', 'ml_confidence_pct',
        'caffe_age_label', 'caffe_confidence',
        'caffe_mid_age', 'engine_est_age', 'used_mediapipe', 'status'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    print()
    print("=" * 65)
    print("  HASIL EVALUASI")
    print("=" * 65)

    total_valid = len(true_ages)
    print(f"\nTotal gambar diproses    : {len(all_files)}")
    print(f"  Berhasil dianalisis    : {total_valid}")
    print(f"  Gagal parse nama file  : {parse_failed}")
    print(f"  Wajah tidak terdeteksi : {face_not_found}")

    if total_valid == 0:
        print("\n[!] Tidak ada data valid untuk dihitung metriknya.")
        engine.close()
        return

    # Metode classifier yang digunakan
    n_ml  = classifier_methods.count('ml')
    n_thr = classifier_methods.count('threshold')
    print(f"\n  Classifier digunakan   : ML={n_ml} | Threshold-fallback={n_thr}")

    # -- Akurasi Kategori Kerutan --------------------------------------------
    correct_cat = sum(p == t for p, t in zip(wrinkle_cats_pred, wrinkle_cats_true))
    acc_cat     = correct_cat / total_valid * 100

    print(f"\n{'─'*65}")
    print(f"  AKURASI KATEGORI KERUTAN")
    print(f"{'─'*65}")
    print(f"  Benar: {correct_cat}/{total_valid}  →  Accuracy = {acc_cat:.1f}%")

    # F1 per kategori
    from collections import Counter
    print(f"\n  Per-Kategori (Precision / Recall / F1):")
    for cat in CATEGORIES:
        tp = sum(1 for t, p in zip(wrinkle_cats_true, wrinkle_cats_pred) if t == cat and p == cat)
        fp = sum(1 for t, p in zip(wrinkle_cats_true, wrinkle_cats_pred) if t != cat and p == cat)
        fn = sum(1 for t, p in zip(wrinkle_cats_true, wrinkle_cats_pred) if t == cat and p != cat)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        n_gt = tp + fn
        print(f"  {cat:12s}: N={n_gt:4d}  Prec={prec:.2f}  Rec={rec:.2f}  F1={f1:.2f}")

    # Confusion Matrix
    print(f"\n  Confusion Matrix (baris=Ground Truth, kolom=Prediksi):")
    print(f"  {'':12s}  " + "  ".join(f"{c:>10s}" for c in CATEGORIES))
    for gt_c in CATEGORIES:
        row_vals = [
            sum(1 for t, p in zip(wrinkle_cats_true, wrinkle_cats_pred)
                if t == gt_c and p == pred_c)
            for pred_c in CATEGORIES
        ]
        print(f"  {gt_c:12s}  " + "  ".join(f"{v:>10d}" for v in row_vals))

    # -- MAE CaffeNet --------------------------------------------------------
    mae_caffe = np.mean(np.abs(np.array(true_ages) - np.array(caffe_pred_ages)))
    print(f"\n{'─'*65}")
    print(f"  MAE MODEL CAFFE NET")
    print(f"{'─'*65}")
    print(f"  Mean Absolute Error (usia): {mae_caffe:.1f} tahun")

    # -- MAE Engine (Ridge) --------------------------------------------------
    if engine_age_preds:
        ta        = np.array([x[0] for x in engine_age_preds])
        pa        = np.array([x[1] for x in engine_age_preds])
        mae_engine = np.mean(np.abs(ta - pa))
        print(f"\n{'─'*65}")
        print(f"  MAE ENGINE (Ridge Regression)")
        print(f"{'─'*65}")
        print(f"  Mean Absolute Error (usia): {mae_engine:.1f} tahun")

    print(f"\nHasil lengkap disimpan ke: '{output_csv}'")
    print("=" * 65)
    engine.close()


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluasi sistem deteksi kerutan wajah v2')
    parser.add_argument('--max',    type=int, default=0,
                        help='Jumlah maksimum gambar (0 = semua)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Path file CSV output')
    args = parser.parse_args()

    evaluate(max_images=args.max, output_csv=args.output)
