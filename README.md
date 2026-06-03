# Facial Wrinkle Level Classification & Age Estimation

Sistem *Computer Vision* berbasis Python untuk mendeteksi wajah secara *real-time*, mengklasifikasikan tingkat kerutan wajah (Muda / Paruh Baya / Tua), serta memberikan estimasi rentang usia berdasarkan kerutan dan model *Deep Learning*.

## ✨ Fitur Utama
1. **Deteksi Wajah Real-Time:** Menggunakan metode Haar Cascade dengan *buffer anti-blinking* agar bounding box tetap stabil.
2. **Ekstraksi Area Wajah (ROI):** Secara dinamis mendeteksi 5 area utama kerutan (Dahi, Mata Kiri, Mata Kanan, Pipi Kiri, Pipi Kanan) menggunakan *MediaPipe Face Mesh* (468 landmarks).
3. **Analisis Tekstur Kulit:** Menggunakan kombinasi metode CLAHE (Peningkatan Kontras), Gaussian Blur, Canny Edge Detection, Local Binary Pattern (LBP), dan Gabor Filter untuk mendeteksi kerutan secara akurat dan mengabaikan *noise* (seperti kacamata).
4. **Machine Learning Classifier:** Klasifikasi kategori kerutan menggunakan model **Random Forest Classifier** yang telah dilatih secara khusus.
5. **Estimasi Usia Ganda:** 
   - **Deep Learning:** Memprediksi rentang usia menggunakan pre-trained *CaffeNet* model (Levi & Hassner 2015).
   - **Ridge Regression:** Memberikan estimasi usia numerik berbasis rasio kerutan dari model regresi.

## ⚙️ Arsitektur Proyek
- `Cam.py` : Script utama (UI Kamera, *smoothing buffer*, integrasi model).
- `wrinkle_engine.py` : *Engine* inti untuk ekstraksi tekstur kulit (Canny, LBP, Gabor) dan prediksi Machine Learning.
- `calibrate.py` : Script *training* untuk melatih ulang model Random Forest dan Ridge Regression menggunakan *dataset*.
- `evaluate.py` : Script untuk mengukur akurasi dari algoritma (F1-Score & MAE).
- `wrinkle_classifier.pkl` : File "otak" model Machine Learning (Random Forest) yang telah dilatih.
- `wrinkle_calibration.json` : File *metadata* yang menyimpan koefisien regresi & nilai *threshold*.

## 🚀 Prasyarat & Instalasi
Pastikan Python 3.7+ sudah terinstall. Kemudian install pustaka pendukung berikut:

```bash
pip install opencv-python opencv-contrib-python
pip install mediapipe
pip install scikit-learn
pip install numpy
pip install joblib
```

## 💻 Cara Penggunaan
1. Lakukan `git clone` pada repository ini.
2. Buka terminal pada folder proyek.
3. Jalankan script utama untuk menyalakan sistem kamera *real-time*:
```bash
python Cam.py
```
*(Tekan tombol `q` pada keyboard untuk keluar dari aplikasi).*

## 📊 Kinerja Model
Berdasarkan evaluasi terhadap >800 data wajah valid:
- **Akurasi Kategori Kerutan (ML Classifier):** ~53.0% (Akurasi tinggi untuk metode *Classic Computer Vision*)
- **Mean Absolute Error (CaffeNet Usia):** ~16.4 tahun
- **Mean Absolute Error (Ridge Regression):** ~13.6 tahun

---
*Proyek ini dikembangkan menggunakan perpaduan antara pendekatan Computer Vision Klasik (Edge Detection & Pattern Recognition) dan Modern Machine Learning.*
