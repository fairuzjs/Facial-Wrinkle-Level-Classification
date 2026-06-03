# -*- coding: utf-8 -*-
"""
calibrate.py (v2)
-----------------------------------------------------------------------------
Kalibrasi & pelatihan model klasifikasi deteksi kerutan menggunakan dataset
Images/.

Pipeline v2 (akurasi ditingkatkan):
  1. Parse semua gambar di Images/ → ekstrak usia aktual dari nama file
  2. Jalankan WrinkleEngine pada setiap gambar → ekstrak 55 fitur
       - 5 fitur: Canny edge % per area ROI
       - 50 fitur: LBP histogram per area ROI (5 area × 10 bin)
  3. Bagi data: 80% train / 20% test (stratified)
  4. Latih Random Forest classifier (Muda / Paruh Baya / Tua)
       - GridSearch untuk hyperparameter optimal
  5. Latih Ridge Regression untuk estimasi usia numerik (backward compat.)
  6. Simpan model ke wrinkle_classifier.pkl + wrinkle_calibration.json
  7. Laporkan akurasi test-set, F1-score, dan confusion matrix

Cara pakai:
    python calibrate.py                 # Proses semua gambar
    python calibrate.py --max 200       # Proses hanya 200 gambar (lebih cepat)
    python calibrate.py --output custom.json

Perkiraan waktu: ~5-12 menit untuk ~1.047 gambar (tergantung hardware).
-----------------------------------------------------------------------------
"""

import cv2
import os
import re
import sys
import json
import argparse
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import (cross_val_score, StratifiedShuffleSplit,
                                     GridSearchCV, train_test_split)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False
    print("[calibrate] PERINGATAN: joblib tidak tersedia — model tidak bisa disimpan.")
    print("  Instal dengan: pip install joblib")

sys.path.insert(0, os.path.dirname(__file__))
from wrinkle_engine import WrinkleEngine

# -----------------------------------------------------------------------------
# Konstanta
# -----------------------------------------------------------------------------

IMAGES_DIR       = 'Images'
FACE_CASCADE     = 'haarcascade_frontalface_default.xml'
DEFAULT_OUTPUT   = 'wrinkle_calibration.json'
CLASSIFIER_OUT   = 'wrinkle_classifier.pkl'

CATEGORIES       = ['Muda', 'Paruh Baya', 'Tua']   # Urutan kelas
CAT_ENCODE       = {'Muda': 0, 'Paruh Baya': 1, 'Tua': 2}

FILENAME_RE = re.compile(r'^.+_(\d{4})-\d{2}-\d{2}_(\d{4})\.jpg$')


def parse_age(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    age = int(m.group(2)) - int(m.group(1))
    return age if 5 <= age <= 100 else None


def age_to_category(age: int) -> str:
    if age < 30:
        return 'Muda'
    elif age < 55:
        return 'Paruh Baya'
    else:
        return 'Tua'


def detect_face(image: np.ndarray, cascade) -> np.ndarray:
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return image[y:y+h, x:x+w]


# -----------------------------------------------------------------------------
# Pengumpulan Fitur dari Dataset
# -----------------------------------------------------------------------------

def collect_features(max_images: int) -> tuple:
    """
    Proses dataset, kembalikan (X, y_age, y_cat):
      X      : array (N, 55) — fitur gabungan Canny% + LBP per area
      y_age  : array (N,)    — usia aktual (float)
      y_cat  : array (N,)    — kategori integer (0=Muda, 1=PB, 2=Tua)
    """
    cascade = cv2.CascadeClassifier(FACE_CASCADE)
    if cascade.empty():
        print("[ERROR] Gagal memuat Haar Cascade.")
        return np.array([]), np.array([]), np.array([])

    # WrinkleEngine tanpa kalibrasi (tidak perlu, hanya ekstrak fitur)
    engine = WrinkleEngine(calibration_path=None, classifier_path=None)

    all_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')])
    if max_images > 0:
        all_files = all_files[:max_images]

    features_list = []
    ages_list     = []
    cats_list     = []

    skipped_parse = 0
    skipped_face  = 0
    skipped_read  = 0
    skipped_feat  = 0

    print(f"Mengumpulkan fitur dari {len(all_files)} gambar...")
    print("(Ini mungkin memakan waktu beberapa menit)")
    print("-" * 60)

    for i, filename in enumerate(all_files):
        if (i + 1) % 50 == 0 or i == 0:
            found = len(features_list)
            print(f"  [{i+1:4d}/{len(all_files)}]  Valid: {found}  |  {filename[:45]}")

        age = parse_age(filename)
        if age is None:
            skipped_parse += 1
            continue

        img_path = os.path.join(IMAGES_DIR, filename)
        image    = cv2.imread(img_path)
        if image is None:
            skipped_read += 1
            continue

        face = detect_face(image, cascade)
        if face is None:
            skipped_face += 1
            continue

        # Ekstraksi 55 fitur
        feat_vec, region_pcts = engine.extract_features_for_training(face)
        if feat_vec is None:
            skipped_feat += 1
            continue

        features_list.append(feat_vec)
        ages_list.append(float(age))
        cats_list.append(CAT_ENCODE[age_to_category(age)])

    engine.close()

    print()
    print(f"  Berhasil          : {len(features_list)} sampel")
    print(f"  Gagal parse nama  : {skipped_parse}")
    print(f"  Gagal baca gambar : {skipped_read}")
    print(f"  Tanpa wajah       : {skipped_face}")
    print(f"  Fitur gagal       : {skipped_feat}")

    return (np.array(features_list, dtype=np.float32),
            np.array(ages_list,     dtype=np.float32),
            np.array(cats_list,     dtype=np.int32))


# -----------------------------------------------------------------------------
# Pelatihan Classifier (Random Forest utama)
# -----------------------------------------------------------------------------

def train_classifier(X: np.ndarray, y_cat: np.ndarray) -> dict:
    """
    Latih classifier kategori usia (Muda/Paruh Baya/Tua) dengan:
      - Train/test split 80/20 stratified
      - Random Forest (utama) dengan hyperparameter tuning
      - Laporan akurasi, F1-score, confusion matrix pada test-set

    Return dict untuk disimpan ke pkl.
    """
    print("\n" + "=" * 60)
    print("  PELATIHAN CLASSIFIER KATEGORI USIA")
    print("=" * 60)

    # Distribusi kelas
    for i, cat in enumerate(CATEGORIES):
        n = int(np.sum(y_cat == i))
        print(f"  Kelas {cat:12s}: {n:4d} sampel ({n/len(y_cat)*100:.1f}%)")

    # Stratified train/test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.20, random_state=42, stratify=y_cat
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # Standarisasi fitur (diperlukan untuk beberapa algoritma)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    # ---- Random Forest -------------------------------------------------------
    # Gunakan class_weight='balanced' untuk kompensasi imbalance
    print("\n-- Random Forest (Hyperparameter Search) --------------------")
    rf_params = {
        'n_estimators':      [200, 300, 400],
        'max_depth':         [None, 15, 25],
        'min_samples_split': [2, 4],
        'min_samples_leaf':  [1, 2],
        'class_weight':      ['balanced'],
    }
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_cv   = GridSearchCV(
        rf_base, rf_params,
        cv=5, scoring='balanced_accuracy',
        n_jobs=-1, verbose=0
    )
    rf_cv.fit(X_tr_sc, y_train)
    rf_best = rf_cv.best_estimator_
    print(f"  Best params: {rf_cv.best_params_}")

    y_pred_rf   = rf_best.predict(X_te_sc)
    acc_rf      = accuracy_score(y_test, y_pred_rf) * 100
    bal_acc_rf  = float(np.mean([
        (y_test[y_pred_rf == i] == i).sum() / max((y_test == i).sum(), 1)
        for i in range(3)
    ]))
    print(f"  Akurasi Test-Set          : {acc_rf:.1f}%")
    print(f"  Balanced Accuracy Test-Set: {bal_acc_rf*100:.1f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=CATEGORIES,
                                zero_division=0))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("  Confusion Matrix (baris=GT, kolom=Pred):")
    print(f"  {'':12s}  " + "  ".join(f"{c:>10s}" for c in CATEGORIES))
    for gi, cat in enumerate(CATEGORIES):
        print(f"  {cat:12s}  " + "  ".join(f"{cm_rf[gi, pi]:>10d}" for pi in range(3)))

    # ---- Feature Importance --------------------------------------------------
    feat_names = (
        [f'canny_{r}' for r in ['mata_kiri','mata_kanan','dahi','pipi_kiri','pipi_kanan']] +
        [f'lbp_{r}_b{b}' for r in ['mata_kiri','mata_kanan','dahi','pipi_kiri','pipi_kanan']
         for b in range(10)] +
        [f'gabor_{r}_l{int(lam)}_t{t}'
         for r in ['mata_kiri','mata_kanan','dahi','pipi_kiri','pipi_kanan']
         for lam in [4, 8] for t in [0, 45, 90, 135]]
    )
    importances = rf_best.feature_importances_
    top10_idx   = np.argsort(importances)[::-1][:10]
    print("\n  Top-10 Feature Importances:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank:2d}. {feat_names[idx]:30s}  {importances[idx]:.4f}")

    return {
        'classifier':       rf_best,
        'scaler':           scaler,
        'classes':          CATEGORIES,
        'accuracy_test':    float(acc_rf),
        'balanced_acc':     float(bal_acc_rf * 100),
        'confusion_matrix': cm_rf.tolist(),
        'best_params':      rf_cv.best_params_,
        'n_train':          int(len(X_train)),
        'n_test':           int(len(X_test)),
        'feature_names':    feat_names,
    }



# -----------------------------------------------------------------------------
# Pelatihan Ridge Regression (estimasi usia numerik)
# -----------------------------------------------------------------------------

def train_ridge_regression(X: np.ndarray, y_age: np.ndarray) -> dict:
    """
    Latih Ridge Regression untuk memetakan fitur kerutan → usia numerik.
    Hanya menggunakan 5 fitur Canny% (kolom 0–4) agar kompatibel
    dengan JSON serialization sederhana.
    """
    X5 = X[:, :5]   # Hanya 5 fitur Canny%
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X5)

    # Cross-validation untuk cari alpha terbaik
    best_alpha    = 1.0
    best_cv_score = -np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model  = Ridge(alpha=alpha)
        scores = cross_val_score(model, X_scaled, y_age,
                                 cv=min(5, len(X5) // 10),
                                 scoring='neg_mean_absolute_error')
        if scores.mean() > best_cv_score:
            best_cv_score = scores.mean()
            best_alpha    = alpha

    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_scaled, y_age)

    y_pred_train = final_model.predict(X_scaled)
    mae_train    = float(np.mean(np.abs(y_age - y_pred_train)))
    cv_mae       = float(-best_cv_score)

    print(f"\n-- Ridge Regression (Estimasi Usia Numerik) -----------------")
    print(f"  Alpha optimal      : {best_alpha}")
    print(f"  MAE (training set) : {mae_train:.1f} tahun")
    print(f"  MAE (CV 5-fold)    : {cv_mae:.1f} tahun")

    # Koefisien dalam ruang ASLI
    coef_original      = final_model.coef_ / scaler.scale_
    intercept_original = (final_model.intercept_
                          - np.dot(coef_original, scaler.mean_))

    return {
        'coef':      coef_original.tolist(),
        'intercept': float(intercept_original),
        'alpha':     best_alpha,
        'mae_cv':    cv_mae,
        'mae_train': mae_train,
        'n_samples': len(y_age),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def calibrate(max_images: int, output_path: str):
    print("=" * 60)
    print("  KALIBRASI & PELATIHAN MODEL DETEKSI KERUTAN WAJAH v2")
    print("=" * 60)
    print()

    # 1. Kumpulkan fitur
    X, y_age, y_cat = collect_features(max_images)

    if len(X) < 30:
        print(f"\n[ERROR] Terlalu sedikit sampel valid ({len(X)}). Perlu minimal 30.")
        return

    # Pastikan semua kelas terwakili
    for i, cat in enumerate(CATEGORIES):
        n = int(np.sum(y_cat == i))
        if n < 5:
            print(f"\n[PERINGATAN] Kelas '{cat}' hanya {n} sampel — akurasi mungkin rendah.")

    print(f"\nDistribusi usia dalam sampel valid:")
    bins = [(0, 20), (20, 30), (30, 40), (40, 55), (55, 100)]
    for lo, hi in bins:
        count = int(np.sum((y_age >= lo) & (y_age < hi)))
        bar   = '#' * (count // 5)
        print(f"  {lo:3d}-{hi:3d} tahun: {count:4d}  {bar}")

    # 2. Latih Classifier ML
    clf_bundle = train_classifier(X, y_cat)

    # 3. Latih Ridge Regression
    ridge_params = train_ridge_regression(X, y_age)

    # 4. Simpan Classifier ke .pkl
    if _JOBLIB_OK:
        joblib.dump(clf_bundle, CLASSIFIER_OUT)
        print(f"\n✓ Classifier disimpan ke: '{CLASSIFIER_OUT}'")
    else:
        print("\n[!] Classifier TIDAK disimpan (joblib tidak tersedia).")

    # 5. Simpan metadata JSON (untuk threshold fallback & Ridge coef)
    output_data = {
        'generated_at':         datetime.now().isoformat(),
        'version':              'v2_ml_classifier',
        'n_training_samples':   int(len(X)),
        # Threshold fallback (digunakan jika pkl tidak tersedia)
        'young_threshold':      12.0,
        'middle_threshold':     22.0,
        'threshold_accuracy':   0.0,   # Tidak relevan lagi
        # Ridge Regression untuk estimasi usia numerik
        **ridge_params,
        'feature_names': [
            'wrinkle_mata_kiri', 'wrinkle_mata_kanan',
            'wrinkle_dahi', 'wrinkle_pipi_kiri', 'wrinkle_pipi_kanan'
        ],
        # Statistik classifier ML
        'classifier_accuracy_test': clf_bundle['accuracy_test'],
        'classifier_file':          CLASSIFIER_OUT,
        'notes': (
            'v2: Classifier ML (Random Forest) menggantikan threshold rule-based. '
            'coef[i] * feature[i] + intercept = estimated_age_years (Ridge fallback). '
            'Load wrinkle_classifier.pkl untuk kategori Muda/Paruh Baya/Tua.'
        )
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Metadata kalibrasi disimpan ke: '{output_path}'")

    print(f"""
{'=' * 60}
  RINGKASAN HASIL KALIBRASI
{'=' * 60}
  Sampel diproses     : {len(X)}
  Akurasi Classifier  : {clf_bundle['accuracy_test']:.1f}%  (test-set 20%)
  MAE Usia (Ridge)    : {ridge_params['mae_cv']:.1f} tahun  (CV 5-fold)

  File disimpan:
    - {CLASSIFIER_OUT}  (Classifier ML / Random Forest)
    - {output_path}     (Metadata & Ridge Regression)

  Langkah berikutnya:
    1. python Cam.py      -> kamera real-time (classifier otomatis dimuat)
    2. python evaluate.py -> evaluasi lengkap pada dataset
{'=' * 60}""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kalibrasi & pelatihan model deteksi kerutan wajah v2')
    parser.add_argument('--max',    type=int, default=0,
                        help='Jumlah maksimum gambar (0 = semua)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Path file JSON output')
    args = parser.parse_args()
    calibrate(max_images=args.max, output_path=args.output)
