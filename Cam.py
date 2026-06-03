"""
Cam.py — Deteksi Kerutan Wajah (Real-time Webcam)
─────────────────────────────────────────────────────────────────────────────
Menggunakan WrinkleEngine (wrinkle_engine.py) dan AgeEstimator secara modular.

Fitur:
  - Deteksi landmark akurat via MediaPipe Face Mesh (468 titik)
  - CLAHE preprocessing untuk meningkatkan kontras kerutan
  - Kalibrasi threshold dari wrinkle_calibration.json (jika ada)
  - Estimasi usia CaffeNet + confidence score (%)
  - Bar kerutan berwarna adaptif (hijau → kuning → merah)
  - Bounding box berwarna sesuai kategori usia
  - Smoothing output antar 10 frame terakhir

Kontrol:
  q — Keluar
─────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from collections import deque

from wrinkle_engine import WrinkleEngine, AgeEstimator

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────────────────

SMOOTHING_WINDOW  = 10
WINDOW_NAME       = "Deteksi Kerutan Wajah"
WINDOW_W, WINDOW_H = 900, 680

# Warna bounding box per kategori (BGR)
CATEGORY_COLOR = {
    'Muda':        (50,  200,  50),   # Hijau
    'Paruh Baya':  (30,  180, 230),   # Kuning-oranye
    'Tua':         (50,   50, 220),   # Merah
}

# ─────────────────────────────────────────────────────────────────────────────
# Fungsi Utilitas UI
# ─────────────────────────────────────────────────────────────────────────────

def draw_wrinkle_bar(frame: np.ndarray, x: int, y: int,
                     pct: float, width: int = 160, height: int = 12):
    """
    Gambar progress bar kerutan di bawah bounding box.
    Warna: hijau (0%) → kuning (50%) → merah (100% dari maks 30%).
    """
    max_pct = 30.0
    ratio   = min(pct / max_pct, 1.0)

    # Warna interpolasi hijau→kuning→merah
    if ratio < 0.5:
        r = int(ratio * 2 * 200)
        g = 200
    else:
        r = 200
        g = int((1 - (ratio - 0.5) * 2) * 200)
    color = (30, g, r)          # BGR

    # Background bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Bar isi
    fill_w = int(width * ratio)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (150, 150, 150), 1)
    # Label %
    cv2.putText(frame, f"{pct:.1f}%", (x + width + 5, y + height - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


def draw_text_with_bg(frame: np.ndarray, text: str,
                      pos: tuple, font_scale: float,
                      text_color: tuple, bg_color: tuple,
                      thickness: int = 1, padding: int = 3):
    """Teks dengan latar belakang gelap agar terbaca di semua kondisi."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def draw_landmark_dots(face_display: np.ndarray, landmarks, color=(0, 255, 100)):
    """Gambar titik tengah setiap area landmark di atas gambar wajah."""
    for (cx, cy), _ in landmarks:
        cv2.circle(face_display, (cx, cy), 3, color, -1)


def draw_confidence_arc(frame: np.ndarray, cx: int, cy: int,
                        radius: int, confidence: float, color: tuple):
    """Gambar busur confidence di sudut kiri-atas bounding box (opsional dekorasi)."""
    angle = int(360 * confidence)
    cv2.ellipse(frame, (cx, cy), (radius, radius),
                -90, 0, angle, color, 2)
    cv2.ellipse(frame, (cx, cy), (radius, radius),
                -90, angle, 360, (60, 60, 60), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Program
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Inisialisasi komponen ─────────────────────────────────────────────────
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if cascade.empty():
        print("[ERROR] Gagal memuat haarcascade_frontalface_default.xml")
        return

    engine        = WrinkleEngine(calibration_path='wrinkle_calibration.json')
    age_estimator = AgeEstimator()

    # -- Buffer smoothing -------------------------------------------------------
    wrinkle_buf    = deque(maxlen=SMOOTHING_WINDOW)   # % kerutan rata-rata
    age_label_buf  = deque(maxlen=SMOOTHING_WINDOW)   # label usia CaffeNet
    category_buf   = deque(maxlen=SMOOTHING_WINDOW)   # kategori kerutan
    clf_method_buf = deque(maxlen=SMOOTHING_WINDOW)   # metode classifier

    # -- Anti-blink: simpan posisi wajah terakhir yang valid -------------------
    # Jika satu/beberapa frame tidak mendeteksi wajah, pakai posisi frame lalu
    # agar bounding box tidak menghilang sesaat (kedip).
    FACE_HOLD_FRAMES  = 8          # Berapa frame posisi lama masih digunakan
    last_faces        = []         # Posisi wajah frame terakhir
    last_face_counter = 0          # Hitungan mundur sebelum posisi kadaluarsa

    # ── Buka kamera ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka kamera.")
        engine.close()
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    print(f"\n[INFO] Sistem aktif. Tekan 'q' untuk keluar.\n")

    # ── Loop utama ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari kamera.")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # minNeighbors=4 + minSize=(110,110): kurangi false positive dari objek
        # latar belakang (bingkai, botol, poster) yang kadang mirip wajah.
        detected = cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(110, 110)
        )

        # Filter false positive: jika terdeteksi >1 wajah, hanya pakai yang
        # TERBESAR (area = w*h). Wajah asli selalu lebih besar dari false positive.
        if len(detected) > 1:
            detected = [max(detected, key=lambda r: r[2] * r[3])]

        # Anti-blink: gunakan deteksi baru jika ada, tahan posisi lama jika tidak
        if len(detected) > 0:
            last_faces        = list(detected)
            last_face_counter = FACE_HOLD_FRAMES
        elif last_face_counter > 0:
            last_face_counter -= 1
            # Pakai posisi wajah dari frame sebelumnya
        else:
            last_faces = []   # Benar-benar tidak ada wajah

        faces = last_faces

        for (fx, fy, fw, fh) in faces:
            face_crop = frame[fy:fy+fh, fx:fx+fw]

            # ── Analisis kerutan (WrinkleEngine) ─────────────────────────────
            wrinkle_result = engine.analyze(face_crop)

            # ── Estimasi usia (CaffeNet) ──────────────────────────────────────
            age_result = age_estimator.predict(face_crop)

            # ── Smoothing ─────────────────────────────────────────────────────
            wrinkle_buf.append(wrinkle_result.avg_wrinkle_pct)
            age_label_buf.append(age_result.age_label)
            category_buf.append(wrinkle_result.age_category)
            clf_method_buf.append(wrinkle_result.classifier_used)

            # Rata-rata kerutan
            smoothed_pct = sum(wrinkle_buf) / len(wrinkle_buf)

            # Label usia: ambil yang paling sering muncul (voting)
            smoothed_age_label = max(set(age_label_buf),
                                     key=age_label_buf.count)

            # Kategori: ambil yang paling sering muncul
            smoothed_category = max(set(category_buf),
                                    key=category_buf.count)

            # Metode classifier yang dominan
            smoothed_clf = max(set(clf_method_buf), key=clf_method_buf.count)

            # ── Warna bounding box adaptif ────────────────────────────────────
            box_color = CATEGORY_COLOR.get(smoothed_category, (200, 200, 200))

            # ── Gambar elemen UI ──────────────────────────────────────────────
            # Bounding box wajah (warna sesuai kategori)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), box_color, 2)

            # Baris 1: Estimasi usia DL + confidence
            conf_pct = int(age_result.confidence * 100)
            line1 = f"Estimasi Usia: {smoothed_age_label}  [{conf_pct}%]"
            draw_text_with_bg(frame, line1,
                              pos=(fx, fy - 38),
                              font_scale=0.48,
                              text_color=(255, 255, 255),
                              bg_color=(30, 30, 30),
                              thickness=1)

            # Baris 2: Kategori kerutan + estimasi usia + ML confidence
            if wrinkle_result.estimated_age is not None:
                age_str = f"  (~{wrinkle_result.estimated_age:.0f} thn)"
            else:
                age_str = ''

            # ML confidence untuk kategori yang diprediksi
            if (wrinkle_result.ml_proba is not None
                    and smoothed_category in ['Muda', 'Paruh Baya', 'Tua']):
                cat_idx  = ['Muda', 'Paruh Baya', 'Tua'].index(smoothed_category)
                ml_conf  = int(wrinkle_result.ml_proba[cat_idx] * 100)
                conf_str = f"  [{ml_conf}%]"
            else:
                conf_str = ''

            line2 = f"Kerutan: {smoothed_category}{age_str}{conf_str}"
            draw_text_with_bg(frame, line2,
                              pos=(fx, fy - 14),
                              font_scale=0.44,
                              text_color=box_color,
                              bg_color=(20, 20, 20),
                              thickness=1)

            # Bar kerutan di bawah bounding box
            bar_y = fy + fh + 8
            draw_wrinkle_bar(frame, fx, bar_y, smoothed_pct, width=fw - 10)

            # Label "Kerutan" kecil di kiri bar
            cv2.putText(frame, "Kerutan",
                        (fx, bar_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (180, 180, 180), 1)

            # Indikator pojok kanan-bawah: MediaPipe (MP/FB) + Classifier (ML/THR)
            if wrinkle_result.used_mediapipe:
                mp_text  = "MP"
                mp_color = (100, 255, 100)
            else:
                mp_text  = "FB"   # Fallback geometris
                mp_color = (100, 100, 255)

            # Classifier indicator
            clf_text  = "ML" if smoothed_clf == 'ml' else "THR"
            clf_color = (100, 255, 200) if smoothed_clf == 'ml' else (80, 180, 255)

            # Tampilkan kedua indikator
            cv2.putText(frame, mp_text,
                        (fx + fw - 42, fy + fh - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, mp_color, 1)
            cv2.putText(frame, clf_text,
                        (fx + fw - 22, fy + fh - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, clf_color, 1)

        # ── Info frame header kecil ───────────────────────────────────────────
        face_count = len(faces)
        header = f"Wajah terdeteksi: {face_count}"
        cv2.putText(frame, header, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    engine.close()
    print("[INFO] Program selesai.")


if __name__ == '__main__':
    main()
