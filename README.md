# URL Classification with XGBoost

Klasifikasi URL menjadi **benign** atau **malicious** menggunakan fitur leksikal dan XGBoost.

Disusun oleh **Wayan Danu Tirta** (NIM 123200025), Universitas Pembangunan Nasional Veteran Yogyakarta.

## Ringkasan

- Dataset: `balanced_urls.csv`, 633.011 URL unik, komposisi 50/50 benign dan malicious.
- Fitur: 20 fitur leksikal (panjang URL, entropi domain, karakter khusus, TLD, karakter Cyrillic, dan lainnya).
- Model: XGBoost dengan hyperparameter tuning `GridSearchCV`.
- Aplikasi: Streamlit, input URL lalu prediksi benign/malicious.
- Evaluasi: 8-fold cross-validation dan holdout 80/20.

## Hasil

| Eksperimen | Fitur | Hasil |
|---|---|---|
| `training.ipynb` (output tersimpan) | 5 | 8-fold CV 97,8%, ROC AUC 0,991 |
| `training.ipynb` (GridSearch) | 20 | best CV 98,1%, holdout 98% |
| `experiment-no-scheme.ipynb` | 16, tanpa `secure_http` | CV 87,7%, ROC AUC 0,949 |
| `experiment-with-scheme.ipynb` | 17, dengan `secure_http` | CV 98,7%, ROC AUC 0,997 |

Model final yang dipakai aplikasi ada di `models-balanced/` (20 fitur).

## Keterbatasan (penting)

Angka di atas **belum bisa dianggap sebagai performa deteksi phishing yang sebenarnya**. Ada beberapa masalah metodologis:

1. **Bias skema URL di dataset.** Semua URL benign memakai `https://`, sedangkan malicious mayoritas `http://` atau tanpa skema. Aturan sederhana "https = benign" saja sudah menghasilkan akurasi 96,8%. Pada uji ulang, fitur `secure_http` menyumbang sekitar 97,8% feature importance. Tanpa fitur skema, akurasi turun ke sekitar 80,8%.
2. **Kebocoran label dari `TargetEncoder`.** Encoder di-fit pada seluruh data sebelum cross-validation dan sebelum train/test split, lalu dipakai untuk semua fitur numerik. Ini membuat evaluasi terlalu optimistis.
3. **Split belum per domain.** URL dari domain yang sama bisa muncul di train dan test, sehingga CV belum mengukur generalisasi ke domain baru.
4. **Input tanpa skema.** Karena model bergantung pada `secure_http`, URL seperti `google.com` (tanpa `https://`) cenderung diprediksi malicious. Aplikasi sebaiknya selalu menerima URL dengan skema.

Perbaikan yang disarankan: bangun ulang dataset dengan perlakuan skema yang konsisten, hapus `TargetEncoder` untuk fitur numerik, dan evaluasi dengan `GroupKFold` per domain.

## Struktur

```text
.
├── app.py                        # Aplikasi Streamlit
├── train.py                      # Pipeline training (data, fitur, CV, GridSearchCV)
├── training.ipynb                # Notebook pipeline final
├── experiment-no-scheme.ipynb    # Eksperimen tanpa fitur skema
├── experiment-with-scheme.ipynb  # Eksperimen dengan fitur skema
├── constant_datas.py             # Daftar TLD terpercaya
├── requirements.txt
├── balanced_urls.csv             # Dataset final
├── models-balanced/              # Model dan encoder yang dipakai app.py
└── reports/
    └── gridcv_results.csv        # Hasil grid search
```

## Setup

Python yang dipakai: **3.10.15**.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
streamlit run app.py
```

Buka `http://localhost:8501`, masukkan URL lengkap (contoh: `https://www.google.com`), lalu klik **Prediksi**.

## Training ulang

```bash
python train.py
```

Perintah ini memproses seluruh `balanced_urls.csv`, menjalankan GridSearchCV, lalu **menimpa** model di `models-balanced/`. Prosesnya lama karena feature engineering dilakukan per URL dan grid search memakai banyak kombinasi.

Notebook `training.ipynb` bisa dijalankan dengan Jupyter. Output yang tersimpan di notebook berasal dari run eksperimen sebelumnya, jadi angka di dalamnya bisa berbeda dengan source terbaru.

## Dataset

`balanced_urls.csv` berisi 633.011 URL unik dengan kolom `url`, `label`, dan `result`. Dataset ini diturunkan dari **Malicious URLs dataset** di Kaggle (651.191 URL, label benign/defacement/phishing/malware):

<https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset>

File asli `malicious_phish.csv` (45 MB) tidak disertakan di repo. Unduh dari Kaggle lalu letakkan di root folder jika ingin membangun ulang dataset dari awal.
