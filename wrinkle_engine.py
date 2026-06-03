"""
-----------------------------------------------------------------------------
Modul inti (engine) yang memisahkan logika deteksi kerutan dan estimasi usia
dari antarmuka kamera/UI. Desain ini memudahkan:
  - Pengujian unit (tanpa kamera)
  - Penggantian model di masa depan
  - Penggunaan ulang kode di evaluate.py dan calibrate.py

Perbaikan v2 (akurasi):
  - Fitur LBP (Local Binary Pattern) ditambahkan per ROI → 55 fitur total
  - Adaptive Canny threshold berbasis median intensitas
  - Classifier Random Forest (dari wrinkle_classifier.pkl) menggantikan
    threshold rule-based yang hanya mencapai 24.9% accuracy
  - Fallback ke threshold jika model pkl belum tersedia
-----------------------------------------------------------------------------
"""

import cv2
import json
import numpy as np

# MediaPipe bersifat opsional — jika DLL gagal diload (umum di subprocess Windows)
# sistem akan otomatis fallback ke landmark geometris
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except (ImportError, OSError):
    _MEDIAPIPE_AVAILABLE = False
    print("[WrinkleEngine] MediaPipe tidak tersedia — pakai landmark geometris")

# joblib untuk load/save model sklearn
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    print("[WrinkleEngine] joblib tidak tersedia — classifier ML tidak bisa dimuat")

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# -----------------------------------------------------------------------------
# Konstanta: Landmark MediaPipe & Ukuran ROI
# -----------------------------------------------------------------------------

LANDMARK_IDX = {
    'mata_kiri':  [33, 160, 158, 133, 153, 144],
    'mata_kanan': [362, 385, 387, 263, 373, 380],
    'dahi':       [10, 151, 9, 8, 107, 336],
    'pipi_kiri':  [234, 116, 93, 132, 58],
    'pipi_kanan': [454, 345, 323, 361, 288],
}

ROI_SIZE = {
    'mata_kiri':  (50, 20),
    'mata_kanan': (50, 20),
    'dahi':       (120, 55),
    'pipi_kiri':  (35, 20),
    'pipi_kanan': (35, 20),
}

REGION_ORDER  = ['mata_kiri', 'mata_kanan', 'dahi', 'pipi_kiri', 'pipi_kanan']
REGION_LABELS = ['Mata Kiri', 'Mata Kanan', 'Dahi', 'Pipi Kiri', 'Pipi Kanan']

# Threshold Canny default (digunakan sebagai fallback; pipeline utama pakai adaptive)
CANNY_THRESHOLDS = [
    (15, 200),  # Mata Kiri
    (15, 200),  # Mata Kanan
    (12, 210),  # Dahi
    (10, 220),  # Pipi Kiri
    (10, 220),  # Pipi Kanan
]

# Batas kategori default (digunakan jika classifier ML tidak tersedia)
DEFAULT_YOUNG_THRESHOLD    = 12.0
DEFAULT_MIDDLE_THRESHOLD   = 22.0

# LBP Parameter
LBP_RADIUS    = 1       # Radius lingkaran LBP
LBP_N_POINTS  = 8      # Jumlah titik sampel LBP
LBP_N_BINS    = 10     # Jumlah bin histogram LBP per ROI
# Gabor filter parameters
GABOR_KSIZE   = 7      # Ukuran kernel Gabor
GABOR_LAMBDAS = [4.0, 8.0]        # 2 skala (frekuensi)
GABOR_THETAS  = [0, 45, 90, 135]  # 4 orientasi (derajat)
# Total fitur = 5 (Canny%) + 5x10 (LBP) + 5x(2x4) (Gabor mean) = 95
GABOR_N_FEATS = len(GABOR_LAMBDAS) * len(GABOR_THETAS)  # 8 per ROI

CLASSIFIER_PATH = 'wrinkle_classifier.pkl'


# -----------------------------------------------------------------------------
# Dataclass Hasil Analisis
# -----------------------------------------------------------------------------

@dataclass
class WrinkleResult:
    """Hasil analisis kerutan wajah."""
    avg_wrinkle_pct:    float           # Rata-rata % kerutan semua area
    per_region_pct:     List[float]     # % kerutan per area (5 nilai)
    age_category:       str             # 'Muda' / 'Paruh Baya' / 'Tua'
    estimated_age:      Optional[float] = None   # Estimasi usia dari Ridge Regression (tahun)
    landmark_positions: Optional[list]  = None   # Posisi landmark untuk visualisasi
    used_mediapipe:     bool = True              # True jika MediaPipe berhasil
    classifier_used:    str  = 'threshold'       # 'ml' atau 'threshold'
    ml_proba:           Optional[List[float]] = None  # Probabilitas kelas [Muda, PB, Tua]


@dataclass
class AgeResult:
    """Hasil estimasi usia dari model CaffeNet."""
    age_label: str          # Label rentang usia, mis. '(25-32)'
    confidence: float       # Probabilitas kelas tertinggi (0–1)
    all_probs: List[float]  # Probabilitas semua 8 kelas


# -----------------------------------------------------------------------------
# WrinkleEngine
# -----------------------------------------------------------------------------

class WrinkleEngine:
    """
    Engine deteksi kerutan wajah (v2 -- akurasi ditingkatkan).

    Pipeline v2:
      1. Resize wajah ke 250x250
      2. Deteksi landmark (MediaPipe -> fallback geometris)
      3. Per ROI:
         a. GaussianBlur -> CLAHE
         b. Adaptive Canny edge detection
         c. Hitung % piksel tepi (5 Canny features)
         d. Hitung histogram LBP (50 LBP features)
         e. Hitung Gabor filter responses (40 Gabor features)
      4. Gabungkan 5+50+40 = 95 fitur
      5. Prediksi kategori usia via Random Forest (ML classifier)
         -> fallback ke threshold rule jika model belum tersedia
      6. Estimasi usia numerik via Ridge Regression
    """

    def __init__(self,
                 calibration_path: Optional[str] = 'wrinkle_calibration.json',
                 classifier_path:  Optional[str] = CLASSIFIER_PATH):
        # Inisialisasi MediaPipe Face Mesh (jika tersedia)
        self._face_mesh = None
        if _MEDIAPIPE_AVAILABLE:
            try:
                _mp = mp.solutions.face_mesh
                self._face_mesh = _mp.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[WrinkleEngine] MediaPipe Face Mesh aktif (468 landmark).")
            except Exception as e:
                print(f"[WrinkleEngine] MediaPipe gagal diinisialisasi ({e}) — pakai geometris")
        else:
            print("[WrinkleEngine] Menggunakan landmark geometris (fallback).")

        # CLAHE (dibuat sekali, bukan tiap frame)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        # Muat kalibrasi Ridge Regression jika tersedia
        self._calibration = None
        self._young_thr   = DEFAULT_YOUNG_THRESHOLD
        self._middle_thr  = DEFAULT_MIDDLE_THRESHOLD
        if calibration_path:
            self._load_calibration(calibration_path)

        # Muat classifier ML jika tersedia
        self._classifier    = None
        self._clf_classes   = ['Muda', 'Paruh Baya', 'Tua']
        self._clf_scaler    = None
        if classifier_path:
            self._load_classifier(classifier_path)

    # -- Public API ----------------------------------------------------------

    def analyze(self, face_bgr: np.ndarray) -> WrinkleResult:
        """
        Analisis kerutan pada citra wajah yang sudah di-crop.
        face_bgr: gambar BGR, ukuran bebas (akan di-resize otomatis)
        """
        face = cv2.resize(face_bgr, (250, 250), interpolation=cv2.INTER_AREA)
        landmarks, used_mp = self._detect_landmarks(face)

        # Ekstraksi fitur lengkap (Canny% + LBP) per ROI
        region_pcts, lbp_features = self._extract_all_features(face, landmarks)
        avg_pct = float(np.mean(region_pcts))

        # Estimasi usia numerik (Ridge Regression)
        estimated_age = self._estimate_age_from_features(region_pcts)

        # Klasifikasi kategori usia
        age_category, clf_used, ml_proba = self._classify(region_pcts, lbp_features)

        return WrinkleResult(
            avg_wrinkle_pct    = avg_pct,
            per_region_pct     = region_pcts,
            age_category       = age_category,
            estimated_age      = estimated_age,
            landmark_positions = landmarks,
            used_mediapipe     = used_mp,
            classifier_used    = clf_used,
            ml_proba           = ml_proba,
        )

    def extract_features_for_training(self, face_bgr: np.ndarray):
        """
        Ekstrak vektor fitur 55-dimensi dari citra wajah.
        Digunakan oleh calibrate.py saat mengumpulkan data training.
        Return: (feature_vector_55d, region_pcts_5d) atau (None, None) jika gagal.
        """
        face = cv2.resize(face_bgr, (250, 250), interpolation=cv2.INTER_AREA)
        landmarks, _ = self._detect_landmarks(face)
        region_pcts, lbp_features = self._extract_all_features(face, landmarks)
        if lbp_features is None or len(lbp_features) == 0:
            return None, None
        feature_vec = np.concatenate([region_pcts, lbp_features])
        return feature_vec, region_pcts

    def get_canny_overlays(self, face_bgr: np.ndarray,
                           landmarks) -> List[Tuple[Tuple, np.ndarray]]:
        """
        Kembalikan (top_left, edges_image) untuk setiap ROI.
        Berguna untuk visualisasi di Cam.py.
        """
        face = cv2.resize(face_bgr, (250, 250), interpolation=cv2.INTER_AREA)
        return self._apply_canny_adaptive(face, landmarks)

    def close(self):
        """Lepaskan resource MediaPipe jika aktif."""
        if self._face_mesh is not None:
            self._face_mesh.close()

    # -- Internal: Landmark Detection ----------------------------------------

    def _detect_landmarks(self, face: np.ndarray):
        """Deteksi landmark wajah. Return (landmarks, used_mediapipe)."""
        h, w = face.shape[:2]

        if self._face_mesh is not None:
            try:
                rgb    = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                result = self._face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    lm  = result.multi_face_landmarks[0].landmark
                    pts = []
                    for region in REGION_ORDER:
                        idx    = LANDMARK_IDX[region]
                        cx     = int(sum(lm[i].x * w for i in idx) / len(idx))
                        cy     = int(sum(lm[i].y * h for i in idx) / len(idx))
                        rw, rh = ROI_SIZE[region]
                        pts.append(((cx, cy), (rw, rh)))
                    return pts, True
            except Exception:
                pass

        # Fallback geometris
        cx, cy = w // 2, h // 2
        fallback = [
            ((cx - 50, cy - 7),  (50, 15)),
            ((cx + 45, cy - 7),  (50, 15)),
            ((cx,      cy - 90), (120, 55)),
            ((cx - 60, cy + 25), (35, 15)),
            ((cx + 70, cy + 25), (35, 15)),
        ]
        return fallback, False

    # -- Internal: Feature Extraction ----------------------------------------

    def _get_roi_gray(self, face: np.ndarray,
                      cx: int, cy: int, lw: int, lh: int) -> Optional[np.ndarray]:
        """Potong ROI dari gambar dan konversi ke grayscale."""
        img_h, img_w = face.shape[:2]
        x1 = max(0, cx - lw // 2)
        y1 = max(0, cy - lh // 2)
        x2 = min(img_w, cx + lw // 2)
        y2 = min(img_h, cy + lh // 2)
        roi = face[y1:y2, x1:x2]
        if roi.size == 0:
            return None, x1, y1
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return gray, x1, y1

    def _preprocess_roi(self, gray: np.ndarray) -> np.ndarray:
        """
        GaussianBlur → CLAHE pada ROI grayscale.
        Blur sebelum CLAHE: menyaring noise/pori-pori sehingga CLAHE
        hanya menonjolkan kerutan nyata yang lebih lebar.
        """
        blurred  = cv2.GaussianBlur(gray, (7, 7), sigmaX=1.2)
        enhanced = self._clahe.apply(blurred)
        return enhanced

    def _canny_adaptive(self, enhanced: np.ndarray,
                        fallback_lo: int, fallback_hi: int) -> np.ndarray:
        """
        Adaptive Canny: threshold dihitung dari median intensitas ROI.
        Lebih robust terhadap variasi pencahayaan dibanding threshold tetap.
        """
        med = float(np.median(enhanced))
        if med > 0:
            lo = max(5,   int(0.60 * med))
            hi = min(255, int(1.40 * med))
        else:
            lo, hi = fallback_lo, fallback_hi
        return cv2.Canny(enhanced, lo, hi)

    def _lbp_histogram(self, gray: np.ndarray) -> np.ndarray:
        """
        Hitung histogram LBP (Local Binary Pattern) dari ROI grayscale.

        LBP membandingkan setiap piksel dengan tetangganya membentuk kode biner.
        Histogram kode LBP merepresentasikan distribusi pola mikro-tekstur,
        sangat efektif untuk membedakan kulit halus (muda) vs. berkerut (tua).

        Implementasi manual (tanpa skimage) untuk portabilitas.
        Menggunakan LBP 8-tetangga radius-1 (LBP_8,1).
        """
        h, w = gray.shape
        if h < 3 or w < 3:
            return np.zeros(LBP_N_BINS, dtype=np.float32)

        g = gray.astype(np.int32)
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        # 8 tetangga: (dy, dx) searah jarum jam dari atas-kiri
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     ( 0,  1),
                     ( 1,  1), ( 1,  0), ( 1, -1),
                     ( 0, -1)]

        center = g[1:-1, 1:-1]
        for bit, (dy, dx) in enumerate(neighbors):
            neighbor = g[1+dy:h-1+dy, 1+dx:w-1+dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)

        # Histogram ternormalisasi
        hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0, 256))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _gabor_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Ekstrak fitur Gabor filter dari ROI grayscale.

        Filter Gabor mendeteksi tekstur berorientasi (seperti kerutan)
        pada berbagai skala dan arah. Setiap filter menghasilkan satu
        nilai (mean response) yang merepresentasikan kekuatan tekstur
        pada orientasi dan frekuensi tersebut.

        Return: array (GABOR_N_FEATS,) = 8 nilai per ROI.
        """
        h, w = gray.shape
        if h < GABOR_KSIZE or w < GABOR_KSIZE:
            return np.zeros(GABOR_N_FEATS, dtype=np.float32)

        feats = []
        img   = gray.astype(np.float32) / 255.0
        for lam in GABOR_LAMBDAS:
            for theta_deg in GABOR_THETAS:
                theta  = np.deg2rad(theta_deg)
                kernel = cv2.getGaborKernel(
                    (GABOR_KSIZE, GABOR_KSIZE),
                    sigma   = lam * 0.56,   # sigma proportional to lambda
                    theta   = theta,
                    lambd   = lam,
                    gamma   = 0.5,
                    psi     = 0,
                    ktype   = cv2.CV_32F,
                )
                filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
                feats.append(float(np.mean(np.abs(filtered))))

        return np.array(feats, dtype=np.float32)

    def _extract_all_features(self, face: np.ndarray,
                               landmarks) -> Tuple[List[float], np.ndarray]:
        """
        Ekstrak semua fitur dari wajah:
          - region_pcts  : [float x5]   -- % Canny edge per ROI
          - extra_features: ndarray (90,) -- LBP(50) + Gabor(40) per ROI

        Digabung menjadi vektor 95 fitur untuk classifier ML.
        """
        region_pcts = []
        lbp_all     = []
        gabor_all   = []

        for i, ((cx, cy), (lw, lh)) in enumerate(landmarks):
            lo_fb, hi_fb = CANNY_THRESHOLDS[i]
            gray, x1, y1 = self._get_roi_gray(face, cx, cy, lw, lh)

            if gray is None:
                region_pcts.append(0.0)
                lbp_all.append(np.zeros(LBP_N_BINS,    dtype=np.float32))
                gabor_all.append(np.zeros(GABOR_N_FEATS, dtype=np.float32))
                continue

            enhanced = self._preprocess_roi(gray)

            # Canny edge percentage
            edges  = self._canny_adaptive(enhanced, lo_fb, hi_fb)
            area   = edges.shape[0] * edges.shape[1]
            pct    = (cv2.countNonZero(edges) / area * 100) if area > 0 else 0.0
            region_pcts.append(pct)

            # LBP histogram
            lbp_all.append(self._lbp_histogram(enhanced))

            # Gabor filter features
            gabor_all.append(self._gabor_features(enhanced))

        extra = np.concatenate(lbp_all + gabor_all) if lbp_all else np.array([])
        return region_pcts, extra

    def _apply_canny_adaptive(self, face: np.ndarray, landmarks) -> list:
        """Kembalikan (top_left, edges) per ROI untuk visualisasi."""
        results  = []
        img_h, img_w = face.shape[:2]

        for i, ((cx, cy), (lw, lh)) in enumerate(landmarks):
            lo_fb, hi_fb = CANNY_THRESHOLDS[i]
            gray, x1, y1 = self._get_roi_gray(face, cx, cy, lw, lh)

            if gray is None:
                results.append(((x1, y1), np.zeros((lh, lw), dtype=np.uint8)))
                continue

            enhanced = self._preprocess_roi(gray)
            edges    = self._canny_adaptive(enhanced, lo_fb, hi_fb)
            results.append(((x1, y1), edges))

        return results

    # -- Internal: Classification & Age Estimation ---------------------------

    def _classify(self, region_pcts: List[float],
                  lbp_features: np.ndarray) -> Tuple[str, str, Optional[List[float]]]:
        """
        Klasifikasi kategori usia.
        Menggunakan ML classifier jika tersedia, fallback ke threshold.
        Return: (category, method_used, proba_list_or_None)
        """
        if self._classifier is not None and lbp_features is not None and len(lbp_features) > 0:
            try:
                feat = np.concatenate([region_pcts, lbp_features]).reshape(1, -1)
                if self._clf_scaler is not None:
                    feat = self._clf_scaler.transform(feat)
                pred  = self._classifier.predict(feat)[0]
                label = self._clf_classes[int(pred)]
                # Probabilitas (untuk tampilan confidence)
                if hasattr(self._classifier, 'predict_proba'):
                    proba = self._classifier.predict_proba(feat)[0].tolist()
                else:
                    proba = None
                return label, 'ml', proba
            except Exception as e:
                print(f"[WrinkleEngine] Classifier error ({e}) — fallback threshold")

        # Fallback threshold
        avg_pct = float(np.mean(region_pcts))
        return self._categorize_threshold(avg_pct), 'threshold', None

    def _categorize_threshold(self, pct: float) -> str:
        """Kategorisasi berbasis threshold (fallback)."""
        if pct < self._young_thr:
            return 'Muda'
        elif pct < self._middle_thr:
            return 'Paruh Baya'
        else:
            return 'Tua'

    def _estimate_age_from_features(self, region_pcts: List[float]) -> Optional[float]:
        """Estimasi usia numerik menggunakan koefisien Ridge Regression dari kalibrasi."""
        if self._calibration is None:
            return None
        coef      = np.array(self._calibration['coef'])
        intercept = self._calibration['intercept']
        age       = float(np.dot(coef, region_pcts) + intercept)
        return max(0.0, age)

    # -- Internal: Model Loading ---------------------------------------------

    def _load_calibration(self, path: str):
        """Muat parameter kalibrasi Ridge Regression dari file JSON."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._calibration = data
            self._young_thr   = data.get('young_threshold',  DEFAULT_YOUNG_THRESHOLD)
            self._middle_thr  = data.get('middle_threshold', DEFAULT_MIDDLE_THRESHOLD)
            print(f"[WrinkleEngine] Kalibrasi dimuat dari '{path}'")
            print(f"  Threshold fallback: Muda < {self._young_thr:.1f}%  |  "
                  f"Paruh Baya < {self._middle_thr:.1f}%")
        except FileNotFoundError:
            print(f"[WrinkleEngine] Kalibrasi '{path}' tidak ditemukan — pakai threshold default")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WrinkleEngine] File kalibrasi tidak valid ({e}) — pakai threshold default")

    def _load_classifier(self, path: str):
        """Muat classifier ML (Random Forest / SVM) dari file .pkl."""
        if not _JOBLIB_AVAILABLE:
            print("[WrinkleEngine] joblib tidak tersedia — tidak bisa load classifier")
            return
        try:
            bundle = joblib.load(path)
            self._classifier  = bundle['classifier']
            self._clf_classes = bundle.get('classes', ['Muda', 'Paruh Baya', 'Tua'])
            self._clf_scaler  = bundle.get('scaler', None)
            clf_name = type(self._classifier).__name__
            acc_test = bundle.get('accuracy_test', None)
            print(f"[WrinkleEngine] Classifier ML dimuat: {clf_name}")
            if acc_test is not None:
                print(f"  Akurasi test-set saat training: {acc_test:.1f}%")
        except FileNotFoundError:
            print(f"[WrinkleEngine] Classifier '{path}' belum ada — jalankan calibrate.py dulu")
        except Exception as e:
            print(f"[WrinkleEngine] Gagal memuat classifier ({e})")


# -----------------------------------------------------------------------------
# AgeEstimator (CaffeNet Wrapper)
# -----------------------------------------------------------------------------

class AgeEstimator:
    """
    Wrapper untuk model estimasi usia CaffeNet (Levi & Hassner 2015).
    Mengenkapsulasi preprocessing blob dan interpretasi output softmax.
    """

    AGE_LIST   = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                  '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    MEAN_BGR   = (78.4263377603, 87.7689143744, 114.895847746)
    INPUT_SIZE = (227, 227)

    def __init__(self,
                 prototxt:   str = 'age/age_deploy.prototxt',
                 caffemodel: str = 'age/age_net.caffemodel'):
        self._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        print("[AgeEstimator] Model CaffeNet berhasil dimuat.")

    def predict(self, face_bgr: np.ndarray) -> AgeResult:
        """
        Prediksi rentang usia dari potongan wajah (BGR, ukuran bebas).
        Return AgeResult dengan label, confidence, dan semua probabilitas.
        """
        resized = cv2.resize(face_bgr, self.INPUT_SIZE)
        blob    = cv2.dnn.blobFromImage(
            resized, 1.0, self.INPUT_SIZE, self.MEAN_BGR, swapRB=False
        )
        self._net.setInput(blob)
        probs = self._net.forward()[0]   # shape: (8,)
        idx   = int(probs.argmax())
        return AgeResult(
            age_label  = self.AGE_LIST[idx],
            confidence = float(probs[idx]),
            all_probs  = probs.tolist(),
        )
