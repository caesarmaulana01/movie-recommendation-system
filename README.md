# Laporan Proyek Machine Learning - T. Muhammad Caesar Maulana

## Project Overview

Dalam era digital saat ini, sistem rekomendasi telah menjadi fitur penting di berbagai platform streaming seperti Netflix dan Disney+. Proyek ini bertujuan untuk membangun sistem rekomendasi film yang dapat membantu pengguna menemukan film berdasarkan preferensi mereka.

**Referensi:**
- [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-4899-7637-6)
- [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)

---

## Business Understanding

### Problem Statements
1. Bagaimana merekomendasikan film berdasarkan kemiripan konten (genre, sutradara, aktor)?
2. Bagaimana mempersonalisasi rekomendasi berdasarkan preferensi pengguna tertentu?
3. Bagaimana mengukur efektivitas sistem rekomendasi yang dibangun?

### Goals
1. Membangun sistem rekomendasi berbasis konten menggunakan TF-IDF dan Cosine Similarity.
2. Mengimplementasikan collaborative filtering dengan SVD.
3. Mengevaluasi performa model menggunakan RMSE dan MAE.

### Solution Approach
1. **Content-Based Filtering**:
   - Menggunakan fitur genre, sutradara, dan aktor.
   - Teknik TF-IDF dan Cosine Similarity.

2. **Collaborative Filtering**:
   - Matrix Factorization dengan SVD.
   - Memanfaatkan data rating pengguna.

---

## Data Understanding

### Sumber Dataset
Dataset yang digunakan dalam proyek ini berasal dari:
- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

### Variabel-variabel dalam Dataset

#### **Dataset Film (`movies_df`)**
Dataset ini berisi informasi terkait berbagai film, termasuk anggaran, genre, dan rating pengguna.

| No | Nama Kolom             | Tipe Data | Deskripsi |
|----|------------------------|----------|-----------|
| 1  | `budget`               | int64    | Total anggaran produksi film. |
| 2  | `genres`               | object   | Daftar genre film. |
| 3  | `id`                   | int64    | ID unik film. |
| 4  | `original_language`    | object   | Bahasa asli film. |
| 5  | `overview`             | object   | Ringkasan singkat cerita film. |
| 6  | `popularity`           | float64  | Skor numerik popularitas film. |
| 7  | `release_date`         | object   | Tanggal rilis film. |
| 8  | `revenue`              | int64    | Total pendapatan film. |
| 9  | `runtime`              | float64  | Durasi film dalam menit. |
| 10 | `title`                | object   | Judul film. |
| 11 | `vote_average`         | float64  | Rata-rata rating film. |
| 12 | `vote_count`           | int64    | Jumlah ulasan film. |

#### **Dataset Kredit (`credits_df`)**
Dataset ini berisi informasi pemeran dan kru film.

| No | Nama Kolom | Tipe Data | Deskripsi |
|----|-----------|----------|-----------|
| 1  | `movie_id` | int64   | ID unik film. |
| 2  | `title`    | object  | Judul film. |
| 3  | `cast`     | object  | Daftar aktor utama. |
| 4  | `crew`     | object  | Daftar kru film, termasuk sutradara. |

#### **Dataset Rating (`ratings_df`)**
Dataset ini berisi rating yang diberikan oleh pengguna terhadap film tertentu.

| No | Nama Kolom  | Tipe Data | Deskripsi |
|----|------------|----------|-----------|
| 1  | `userId`   | int64    | ID unik pengguna. |
| 2  | `movieId`  | int64    | ID unik film. |
| 3  | `rating`   | float64  | Skor rating (1-5). |
| 4  | `timestamp`| int64    | Waktu pemberian rating. |

### Exploratory Data Analysis (EDA)

- **Distribusi Rating Film**  
  Hasil analisis menunjukkan bahwa rating film cenderung mengikuti distribusi normal dengan mean sekitar 6.09.
  
- **Film dengan Pendapatan Tertinggi**  
  Film dengan pendapatan tertinggi adalah *Avatar* dengan pendapatan sebesar $2.78M.

- **Visualisasi**  
  - Distribusi rating pengguna.
  - Hubungan antara popularitas dan pendapatan film.
  - Jumlah film berdasarkan bahasa asli.
  - Hubungan durasi film dengan rating rata-rata.
  - Perusahaan produksi dengan jumlah film terbanyak.

---

## Data Preparation

Langkah-langkah preprocessing data:
1. **Handling Missing Values**  
   - Mengisi nilai kosong pada kolom `overview` dengan string kosong.
  
2. **Feature Engineering**  
   - Ekstraksi sutradara dari kolom `crew`.
   - Normalisasi teks dengan lowercase dan penghapusan whitespace.
  
3. **Data Transformation**  
   - Pembuatan metadata soup untuk content-based filtering.

---

## Modeling

### 1. Content-Based Filtering
Pendekatan ini menggunakan metode **TF-IDF** untuk mengubah teks menjadi vektor numerik, kemudian menghitung kesamaan antarfilm menggunakan **Cosine Similarity**.

**Contoh Rekomendasi:**
| Film Asli               | Rekomendasi (Score)      |
|-------------------------|-------------------------|
| *The Dark Knight Rises* | *The Dark Knight* (0.70) |

### 2. Collaborative Filtering
Pendekatan ini menggunakan **Singular Value Decomposition (SVD)** untuk menganalisis pola rating pengguna terhadap film.

Evaluasi dilakukan dengan **Cross-Validation** menggunakan metrik RMSE dan MAE.

---

## Evaluation

### **Metrik Evaluasi**
1. **Root Mean Squared Error (RMSE)**
   \[
   RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
   \]
   RMSE mengukur deviasi antara prediksi dan nilai aktual.

2. **Mean Absolute Error (MAE)**
   \[
   MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
   \]
   MAE mengukur rata-rata perbedaan absolut antara prediksi dan aktual.

3. **Cosine Similarity**
   \[
   similarity(A, B) = \frac{A \cdot B}{||A|| ||B||}
   \]
   Digunakan dalam content-based filtering untuk mengukur kesamaan antarfilm.

### **Hasil Evaluasi**
| Model                  | RMSE    | MAE     |
|------------------------|--------|--------|
| Content-Based (TF-IDF) | -      | -      |
| Collaborative (SVD)    | 0.8972 | 0.6912 |

---

## Perbandingan Metode

| Metode                  | Kelebihan                                      | Kekurangan                                      |
|-------------------------|-----------------------------------------------|------------------------------------------------|
| **Content-Based**       | Tidak membutuhkan data pengguna.             | Terbatas pada kemiripan konten.                |
| **Collaborative**       | Personalisasi berbasis preferensi pengguna.  | Membutuhkan data rating yang cukup.            |

---

## Conclusion

- **Sistem rekomendasi berbasis konten** efektif dalam merekomendasikan film yang mirip berdasarkan genre dan aktor.
- **Collaborative filtering dengan SVD** memberikan rekomendasi yang lebih personal tetapi membutuhkan jumlah data rating yang lebih banyak.
- Evaluasi menunjukkan bahwa SVD memiliki RMSE sebesar **0.8972** dan MAE **0.6912**, yang menunjukkan performa yang cukup baik.

**Potensi Peningkatan:**
1. Menggunakan **Hybrid Recommendation System** untuk menggabungkan content-based dan collaborative filtering.
2. Meningkatkan model collaborative filtering dengan teknik **deep learning**.

---
