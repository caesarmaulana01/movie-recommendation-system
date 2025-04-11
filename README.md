# Laporan Proyek Machine Learning - T. Muhammad Caesar Maulana

## Project Overview

Dalam era digital saat ini, sistem rekomendasi telah menjadi komponen kritis dalam berbagai platform digital. Pertumbuhan eksponensial dalam pengumpulan data telah memunculkan sistem informasi yang lebih canggih, dimana Recommendation Systems memainkan peran penting. Sistem ini berfungsi sebagai filter informasi yang meningkatkan kualitas hasil pencarian dengan menyajikan item yang lebih relevan baik berdasarkan query pencarian maupun riwayat pengguna.

Sistem rekomendasi mampu memprediksi rating atau preferensi yang mungkin diberikan pengguna terhadap suatu item. Aplikasinya telah diadopsi secara luas oleh perusahaan teknologi terkemuka seperti Netflix yang mengandalkan efektivitas sistem rekomendasi sebagai inti bisnis

Proyek ini berfokus pada pembangunan sistem rekomendasi film berbasis dataset TMDB 5000 Movie Dataset. Implementasi ini akan mencakup:
1. Sistem rekomendasi berbasis konten (content-based filtering)
2. Sistem rekomendasi kolaboratif (collaborative filtering)
3. Evaluasi performa model menggunakan metrik RMSE dan MAE

**Mengapa proyek ini penting?**
- Meningkatkan user experience dalam platform streaming
- Mengurangi waktu pencarian konten yang relevan
- Meningkatkan engagement pengguna melalui personalisasi

**Referensi:**
- [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-4899-7637-6)
- [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)
- [The Netflix Recommender System](https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8)

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
   - Menggunakan fitur pada metadata film.
   - Teknik TF-IDF dan Cosine Similarity.

2. **Collaborative Filtering**:
   - Matrix Factorization dengan SVD.
   - Memanfaatkan data rating pengguna.

---

## Data Understanding

Proyek ini menggunakan dua dataset berbeda yang saling melengkapi untuk membangun sistem rekomendasi film berbasis *content-based filtering* dan *collaborative filtering*.


### Dataset 1: **TMDB 5000 Movie Dataset**
**Sumber**: [TMDB 5000 Movie Dataset (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

#### Deskripsi
Dataset ini berisi metadata untuk sekitar 5.000 film yang tersedia di The Movie Database (TMDB). Data mencakup informasi tentang pemeran, kru, genre, anggaran, pendapatan, tanggal rilis, bahasa, perusahaan produksi, dan negara produksi.

**Tabel**: `movies_df` dan `credits_df`

#### Ukuran Dataset

| Tabel      | Jumlah Baris | Jumlah Kolom |
|------------|--------------|--------------|
| movies_df  | 4803         | 20           |
| credits_df | 4803         | 4            |

#### Kondisi Data

- **Missing Values:**
  - `homepage`: 3091 nilai kosong
  - `tagline`: 844 nilai kosong
  - `overview`: 3 nilai kosong
  - `runtime`: 2 nilai kosong
  - `release_date`: 1 nilai kosong

- **Data Duplikat**: Tidak ditemukan

- **Outlier (berdasarkan boxplot)**:
![Box Plot `movies_df`](images/boxplot_movies_df.jpg)
  - `budget`, `revenue`, `popularity`, `vote_count`, dan `id`: Banyak nilai outlier
  - `runtime`: Outlier dengan durasi sangat pendek/panjang
  - `vote_average`: Relatif normal, ada nilai ekstrem
  - `id`: Distribusi panjang dengan outlier

![Box Plot `movies_df`](images/boxplot_movies_df.jpg)

#### Struktur Fitur: `movies_df`

| Kolom                | Tipe Data | Deskripsi                                                                 |
|----------------------|-----------|---------------------------------------------------------------------------|
| budget               | int64     | Anggaran produksi film (USD)                                             |
| genres               | object    | Daftar genre film (format JSON string)                                   |
| homepage             | object    | URL resmi film                                                           |
| id                   | int64     | ID unik film                                                             |
| keywords             | object    | Kata kunci isi cerita film (JSON string)                                 |
| original_language    | object    | Bahasa asli film                                                         |
| original_title       | object    | Judul asli saat perilisan                                                |
| overview             | object    | Ringkasan cerita                                                         |
| popularity           | float64   | Skor popularitas berdasarkan TMDB                                        |
| production_companies | object    | Daftar perusahaan produksi (JSON string)                                 |
| production_countries | object    | Daftar negara produksi (JSON string)                                     |
| release_date         | object    | Tanggal rilis                                                            |
| revenue              | int64     | Total pendapatan film (USD)                                              |
| runtime              | float64   | Durasi film (menit)                                                      |
| spoken_languages     | object    | Bahasa yang digunakan (JSON string)                                      |
| status               | object    | Status rilis film, misal: "Released", "Post Production"                  |
| tagline              | object    | Kalimat promosi film                                                     |
| title                | object    | Judul umum film                                                          |
| vote_average         | float64   | Rata-rata skor rating pengguna                                           |
| vote_count           | int64     | Jumlah suara rating dari pengguna                                        |

#### Struktur Fitur: `credits_df`

| Kolom     | Tipe Data | Deskripsi                                                        |
|-----------|-----------|-------------------------------------------------------------------|
| movie_id  | int64     | ID film (relasi dengan `movies_df.id`)                          |
| title     | object    | Judul film                                                       |
| cast      | object    | Daftar pemeran utama (format JSON string)                        |
| crew      | object    | Daftar kru film (termasuk sutradara, format JSON string)         |

---

### Dataset 2: **The Movies Dataset – Ratings**
**Sumber**: [The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

#### 📝 Deskripsi
Dataset ini lebih luas, mencakup metadata untuk 45.000 film yang tercantum dalam Full MovieLens Dataset. Film yang ada di dataset ini dirilis pada atau sebelum Juli 2017. Selain metadata film, dataset ini juga mencakup 26 juta rating dari 270.000 pengguna untuk semua film dalam dataset. Rating diberikan dalam skala 1-5 dan diperoleh dari situs resmi GroupLens.

**Tabel**: `ratings_df`

#### Ukuran Dataset

| Tabel      | Jumlah Baris | Jumlah Kolom |
|------------|--------------|--------------|
| ratings_df | 100,004      | 4            |

#### Kondisi Data

- **Missing Values**: Tidak ada
- **Data Duplikat**: Tidak ditemukan

- **Outlier (berdasarkan boxplot)**:
  - `rating`: Beberapa nilai ekstrem (terutama di bawah 1.0)
  - `movieId`: Distribusi panjang dengan outlier
  - `timestamp`: Relatif normal, sedikit outlier ekstrem

![Box Plot `ratings_df`](images/boxplot_ratings_df.png)

#### Struktur Fitur: `ratings_df`

| Kolom     | Tipe Data | Deskripsi                                           |
|-----------|-----------|-----------------------------------------------------|
| userId    | int64     | ID pengguna yang memberi rating                     |
| movieId   | int64     | ID film yang diberi rating                          |
| rating    | float64   | Skor rating film (rentang 0.5 hingga 5.0)           |
| timestamp | int64     | Waktu rating dalam format UNIX timestamp            |

### Exploratory Data Analysis (EDA)

#### 1. **Ringkasan Statistik Dataset Film**
Berdasarkan analisis statistik, berikut adalah beberapa insight penting dari dataset film:

- **Rata-rata Rating Film**: Rata-rata rating film adalah 6.09, menunjukkan bahwa sebagian besar film mendapatkan rating yang cukup baik dari penonton.
- **Durasi Film Terpanjang**: Durasi film terpanjang dalam dataset adalah 338 menit.
- **Film dengan Pendapatan Tertinggi**: Film dengan pendapatan tertinggi adalah *Avatar* dengan total pendapatan $2,787,965,087.


2. Distribusi Rating Film
Distribusi rata-rata rating film menunjukkan frekuensi rating yang diberikan oleh pengguna. Hasil visualisasi menunjukkan bahwa mayoritas film memiliki rating sekitar 6 hingga 7, dengan distribusi yang cenderung normal.

![Distribusi Rating Film](images/user_rating_distribution.png)


3. Distribusi Rating Pengguna
Distribusi rating pengguna dalam dataset The Movies menunjukkan bahwa sebagian besar rating diberikan dalam rentang 3 hingga 4, yang menunjukkan preferensi pengguna terhadap film yang memiliki rating lebih tinggi.

![Rating Pengguna](images/mean_distribution_vote.png)

4. Hubungan antara Popularitas dan Pendapatan Film
Hubungan antara popularitas dan pendapatan menunjukkan bahwa film dengan popularitas yang lebih tinggi cenderung memiliki pendapatan yang lebih besar, meskipun ada beberapa pengecualian.

![Popularitas dan Pendapatan Film](images/popularity_income.png)

5. Jumlah Film Berdasarkan Bahasa Asli
Jumlah film yang diproduksi dalam berbagai bahasa menunjukkan bahwa bahasa Inggris adalah yang paling dominan, diikuti oleh bahasa-bahasa lain.

![Jumlah Film Berdasarkan Bahasa](images/languange.png)

6. Hubungan antara Durasi Film dan Rata-rata Vote
Analisis hubungan antara durasi film dan rating menunjukkan bahwa film dengan durasi lebih panjang tidak selalu mendapat rating yang lebih baik.


![Durasi Film dan Rata-rata](images/duration_mean.png)

7. Top 10 Production Companies dengan Jumlah Film Terbanyak
Top 10 perusahaan produksi dengan jumlah film terbanyak di dataset menunjukkan perusahaan besar seperti Walt Disney Pictures mendominasi.


![Top 10 Production Companies](images/top_10.png)

8. Heatmap Korelasi antara Variabel Penting
Heatmap ini menunjukkan hubungan antar variabel penting seperti budget, popularitas, pendapatan, runtime, dan rating film. Korelasi antara pendapatan dan popularitas sangat kuat.

![Heatmap Korelasi](images/corellation_heatmap.png)

---

## Data Preparation

Tahapan ini bertujuan untuk membersihkan, menyatukan, dan menyiapkan data sebelum dilakukan pemodelan dan sistem rekomendasi. Berikut adalah langkah-langkah data preparation secara sistematis:

### 1. Penggabungan Dataset

Dataset `movies` dan `credits` masing-masing berisi informasi metadata film dan kru/pemeran. Keduanya digabungkan berdasarkan kolom `id` untuk membentuk satu dataframe terpadu. Proses ini mencakup pengubahan nama kolom untuk konsistensi serta penghapusan duplikasi nama kolom.

### 2. Seleksi Film Populer Berdasarkan Rating

Untuk menyoroti film-film berkualitas tinggi, dilakukan pendekatan kuantitatif berdasarkan jumlah vote dan nilai rata-rata rating film. Tahapannya adalah sebagai berikut:

#### Proses:
- Menghitung rata-rata vote seluruh film (`C`).
- Menentukan ambang batas (`m`) sebagai **kuantil 90%** dari jumlah vote.
- Menggunakan formula **weighted rating** dari IMDb untuk menghitung skor akhir film yang mempertimbangkan baik rating maupun jumlah pemilih.

#### Formula:
```math
\text{Weighted Rating} = \left(\frac{v}{v+m} \times R \right) + \left(\frac{m}{m+v} \times C \right)
```
Dimana:
- **$v$** = Jumlah vote untuk film tersebut  
- **$R$** = Rata-rata rating film  
- **$m$** = Ambang batas jumlah vote (kuantil 90%)  
- **$C$** = Rata-rata rating dari seluruh film  

### 3. Visualisasi Film Terpopuler

Film diurutkan berdasarkan nilai `popularity`, dan enam film terpopuler divisualisasikan menggunakan grafik batang horizontal untuk memberikan gambaran umum tren popularitas.

### 4. Ekstraksi Fitur dengan TF-IDF (Overview)

- TF-IDF (Term Frequency–Inverse Document Frequency) digunakan untuk mengolah kolom `overview`.
- Tujuannya adalah untuk menilai pentingnya kata-kata dalam deskripsi film terhadap keseluruhan dataset.
- Matriks kesamaan antar film dihitung menggunakan **Cosine Similarity** berdasarkan hasil TF-IDF.

### 5. Parsing Data JSON ke Format Python

Beberapa kolom seperti `cast`, `crew`, `keywords`, dan `genres` yang semula berbentuk string JSON diubah menjadi struktur Python (list/dictionary) agar dapat diolah lebih lanjut.

### 6. Ekstraksi Fitur Tambahan

- Sutradara diambil dari kolom `crew`.
- Nama pemeran, genre, dan kata kunci diambil dari masing-masing kolom dan dibatasi hingga tiga elemen teratas untuk menjaga kesederhanaan data.

### 7. Pembersihan dan Normalisasi Data

- Semua elemen dalam kolom `cast`, `keywords`, `genres`, dan `director` diubah menjadi huruf kecil dan spasi dihapus.
- Tujuannya untuk menyamakan format string dalam pemodelan berbasis teks.

### 8. Pembuatan Fitur Gabungan ('Soup')

- Fitur `keywords`, `cast`, `director`, dan `genres` digabung menjadi satu string teks yang disebut **soup**.
- Soup ini digunakan sebagai dasar sistem rekomendasi berbasis konten karena mengandung informasi semantik penting dari film.

### 9. Ekstraksi Fitur dengan Count Vectorizer

- Count Vectorizer digunakan untuk menghitung frekuensi kata dalam `soup`.
- Hasilnya kemudian digunakan untuk menghitung **cosine similarity** antar film berdasarkan kata-kata yang muncul bersama.

### 10. Pemetaan Judul ke Indeks

- Dibuat pemetaan balik antara judul film dengan indeks dataframe untuk memudahkan pencarian dan proses rekomendasi berbasis judul film.

---

## Modeling

### Tujuan Modeling

Tahap modeling bertujuan membangun sistem rekomendasi untuk membantu pengguna menemukan film yang sesuai dengan preferensinya, berdasarkan konten film atau pola interaksi pengguna lain. Sistem ini diharapkan dapat memberikan saran film yang relevan secara otomatis tanpa harus melakukan pencarian manual.


### Skema Sistem Rekomendasi

Dua pendekatan sistem rekomendasi yang diterapkan dalam proyek ini adalah:

1. **Content-Based Filtering**  
2. **Collaborative Filtering (dengan algoritma SVD)**

Kedua pendekatan digunakan untuk menyelesaikan permasalahan dari sisi yang berbeda, dengan tujuan akhir yang sama: memberikan rekomendasi film yang relevan bagi pengguna.


### 1. Content-Based Filtering

#### Definisi:
Sistem ini merekomendasikan film berdasarkan kemiripan deskripsi dan genre dari film yang diberikan sebagai input. Pendekatan ini hanya menggunakan metadata film, tanpa memperhatikan interaksi pengguna.

#### Cara Kerja:
- Deskripsi film dikonversi ke bentuk vektor menggunakan TF-IDF.
- Kemiripan antar deskripsi dihitung menggunakan Cosine Similarity.
- Genre film dibandingkan untuk menambah skor relevansi.
- Film dengan skor tertinggi direkomendasikan.

#### Kelebihan & Kekurangan
| Aspek                 | Kelebihan | Kekurangan |
|-----------------------|-----------|------------|
| **Ketergantungan Data** | Tidak memerlukan rating pengguna | Terbatas pada metadata film |
| **Rekomendasi** | Dapat merekomendasikan film yang mirip | Tidak bisa menangani cold start pengguna baru |
| **Komputasi** | Cepat dan efisien | Kualitas rekomendasi bergantung pada deskripsi film |

#### Output Top-N Recommendation yang Dihasilkan Model:

**Input: The Dark Knight Rises**

| Rank | Judul Film                | Similarity Score | Matched Genres        | Relevance Score |
|------|---------------------------|------------------|------------------------|-----------------|
| 1    | The Dark Knight           | 0.700000         | [drama, crime, action] | 3               |
| 2    | Batman Begins             | 0.700000         | [drama, crime, action] | 3               |
| 3    | Amidst the Devil's Wings | 0.547723         | [drama, crime, action] | 3               |
| 4    | The Prestige              | 0.400000         | [drama]                | 1               |
| 5    | Romeo Is Bleeding        | 0.400000         | [drama, crime, action] | 3               |

**Input: The Godfather**

| Rank | Judul Film               | Similarity Score | Matched Genres      | Relevance Score |
|------|--------------------------|------------------|----------------------|-----------------|
| 1    | The Godfather: Part III | 0.527046         | [drama, crime]       | 2               |
| 2    | The Godfather: Part II  | 0.421637         | [drama, crime]       | 2               |
| 3    | Amidst the Devil's Wings| 0.384900         | [drama, crime]       | 2               |
| 4    | The Son of No One       | 0.377964         | [drama, crime]       | 2               |
| 5    | Apocalypse Now          | 0.333333         | [drama]              | 1               |

---

### 2. Collaborative Filtering (SVD)

#### Definisi:
Sistem ini memanfaatkan data interaksi pengguna berupa rating. Rekomendasi diberikan berdasarkan pola kesamaan perilaku antar pengguna.

#### Cara Kerja:
- Dibuat matriks pengguna-film dari data rating.
- Algoritma SVD digunakan untuk mendekomposisi matriks menjadi representasi laten.
- Model memprediksi rating film yang belum ditonton pengguna.
- Film dengan rating prediksi tertinggi direkomendasikan.

#### Kelebihan & Kekurangan
| Aspek                 | Kelebihan | Kekurangan |
|-----------------------|-----------|------------|
| **Ketergantungan Data** | Mampu menangkap pola preferensi pengguna | Membutuhkan banyak data rating |
| **Rekomendasi** | Dapat memberikan saran yang lebih personal | Kesulitan dalam menangani cold start film baru |
| **Komputasi** | Akurat dalam prediksi rating | Membutuhkan daya komputasi lebih tinggi |

#### Output Top-N Recommendation yang Dihasilkan Model:

**User 2 – Film yang Pernah Ditonton:**

| Title                  | Genres                        | Rating |
|------------------------|-------------------------------|--------|
| The Conversation       | crime, drama, mystery         | 5.0    |
| The Hours              | drama                         | 5.0    |
| Monsters, Inc.         | animation, comedy, family     | 5.0    |
| Terminator 3           | action, thriller, sci-fi      | 4.0    |
| Romeo + Juliet         | drama, romance                | 4.0    |
| Reservoir Dogs         | crime, thriller               | 4.0    |

**Top-10 Rekomendasi untuk User 2:**

| Rank | Title                  | Genres                         | Predicted Rating |
|------|------------------------|---------------------------------|------------------|
| 1    | Scarface               | action, crime, drama            | 4.33             |
| 2    | The Good Thief         | crime, drama, thriller          | 4.32             |
| 3    | Beverly Hills Cop III  | action, comedy, crime           | 4.28             |
| 4    | The Sixth Sense        | mystery, thriller, drama        | 4.26             |
| 5    | Terminator Salvation   | action, sciencefiction, thriller| 4.23             |


### Perbandingan dan Analisis

| Aspek                       | Content-Based Filtering                        | Collaborative Filtering (SVD)               |
|----------------------------|-------------------------------------------------|---------------------------------------------|
| **Sumber Data**            | Metadata film (deskripsi, genre)               | Data interaksi pengguna (rating)            |
| **Kelebihan**              | Tidak butuh data pengguna lain, cocok untuk pengguna baru | Bisa menangkap pola tersembunyi antar pengguna |
| **Kekurangan**             | Terbatas pada kesamaan konten, bisa overfitting preferensi awal | Tidak cocok jika data rating sedikit (cold start) |
| **Cocok untuk**            | Rekomendasi berdasarkan satu film input        | Rekomendasi personal berdasarkan histori     |


### Kesimpulan Modeling

Kedua pendekatan memiliki kelebihan masing-masing dan dapat digunakan saling melengkapi. Content-Based cocok digunakan saat sistem belum memiliki banyak data pengguna, sedangkan Collaborative Filtering memberikan rekomendasi yang lebih bersifat personal. Kombinasi keduanya dapat menghasilkan sistem rekomendasi yang lebih akurat dan fleksibel dalam berbagai kondisi.

---

## Evaluasi

### 1. Evaluasi Content-Based Filtering (NDCG@K)
Pada **Content-Based Filtering**, model rekomendasi dibangun untuk menjawab problem statement pertama, yaitu bagaimana merekomendasikan film berdasarkan kemiripan konten seperti genre, sutradara, dan aktor. Dengan menggunakan teknik **TF-IDF Vectorizer** dan **Cosine Similarity**, sistem mampu mengukur kesamaan antarfilm dan menghasilkan rekomendasi yang relevan.

#### Metrik yang digunakan:
**NDCG@10 (Normalized Discounted Cumulative Gain at 10)** adalah metrik evaluasi yang digunakan untuk mengukur kualitas peringkat hasil rekomendasi hingga posisi ke-10. Metrik ini tidak hanya memperhitungkan relevansi item yang direkomendasikan — dalam hal ini diukur berdasarkan kesamaan genre dengan film input — tetapi juga mempertimbangkan posisi item tersebut dalam daftar. Artinya, item yang lebih relevan dan muncul di posisi lebih atas akan memberikan kontribusi yang lebih besar terhadap skor keseluruhan. NDCG kemudian melakukan normalisasi terhadap DCG (Discounted Cumulative Gain) dengan membandingkannya terhadap IDCG (Ideal DCG), yaitu skor maksimum yang bisa diperoleh jika semua item relevan berada di posisi teratas. Nilai NDCG@10 berada dalam rentang 0 hingga 1, di mana skor 1.0 menunjukkan urutan rekomendasi yang sempurna.

#### Cara kerja: 
Metrik ini membandingkan ranking yang dihasilkan dengan ranking ideal, dimana item dengan relevansi tertinggi berada di posisi teratas.

#### Formula:
```math
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
NDCG@K = \frac{DCG@K}{IDCG@K}
```
Dimana:
- **$`rel_i`$** = Relevansi item pada posisi ke-`i` dalam daftar rekomendasi
- **$`i`$** = Posisi item dalam daftar peringkat, dimulai dari 1 hingga $`K`$
- **$`K`$** = Jumlah item teratas (top-K) yang dievaluasi
- **$`DCG@K`$** = Discounted Cumulative Gain hingga posisi ke-K
- **$`IDCG@K`$** = Ideal DCG, yaitu DCG maksimum yang mungkin jika semua item relevan berada di urutan teratas
- **$`NDCG@K`$** = Normalized DCG, yaitu rasio antara DCG dan IDCG untuk normalisasi skor antara 0 dan 1

#### Hasil Evaluasi:
| Judul Film           | Skor NDCG@10 |
|----------------------|-------------|
| The Dark Knight Rises | 0.9678      |
| The Godfather        | 0.9791      |

#### Analisis Hasil:
- Skor mendekati 1.0 menunjukkan:
  - Rekomendasi sangat relevan (kesamaan genre tinggi)
  - Urutan optimal (film paling mirip di peringkat teratas)
- Untuk "The Dark Knight Rises", sistem berhasil merekomendasikan film Batman terkait di posisi teratas


### 2. Evaluasi Collaborative Filtering (SVD)

Pada **Collaborative Filtering**, model ini berfokus pada problem statement kedua, yaitu bagaimana mempersonalisasi rekomendasi berdasarkan preferensi pengguna tertentu. Dengan menggunakan teknik **Singular Value Decomposition (SVD)**, sistem dapat memprediksi rating film berdasarkan pola rating pengguna lain yang memiliki preferensi serupa.


#### 1. RMSE (Root Mean Squared Error)

##### Definisi:
**RMSE (Root Mean Squared Error)** adalah metrik evaluasi yang digunakan untuk mengukur deviasi standar dari kesalahan prediksi dalam sebuah model. RMSE memberikan gambaran seberapa jauh prediksi model menyimpang dari nilai sebenarnya, dengan memberi penalti lebih besar terhadap kesalahan prediksi yang besar. Oleh karena itu, metrik ini sangat berguna ketika kesalahan besar perlu diminimalkan, karena sifatnya yang sensitif terhadap outlier. Nilai RMSE yang lebih kecil menunjukkan bahwa model memiliki kinerja prediksi yang lebih akurat.
   
##### Cara kerja: 
Menghitung akar kuadrat dari rata-rata kuadrat selisih antara rating prediksi dan rating aktual

##### Formula:
```math
RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
```
Dimana:
- **$`y_i`$** = Rating sebenarnya dari pengguna
- **$`\hat{y}_i`$** = Rating yang diprediksi oleh model
- **$`N`$** = Jumlah sampel

#### 1. MAE (Mean Absolute Error)

**MAE (Mean Absolute Error)** adalah metrik evaluasi yang digunakan untuk mengukur rata-rata dari kesalahan absolut antara rating yang diprediksi oleh model dan rating sebenarnya yang diberikan oleh pengguna. Metrik ini memberikan gambaran seberapa besar rata-rata deviasi prediksi dari nilai sebenarnya tanpa memperhatikan arah kesalahannya (positif atau negatif), sehingga semakin kecil nilai MAE, maka semakin akurat prediksi model tersebut.
   
##### Cara kerja: 
Menghitung rata-rata selisih mutlak antara rating prediksi dan rating aktual

##### Formula:
```math
MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
```
Dimana:
- **$y_i$** = Rating sebenarnya dari pengguna ke-$i$
- **$\hat{y}_i$** = Rating yang diprediksi oleh model untuk pengguna ke-$i$
- **$N$** = Jumlah total data atau sampel yang dievaluasi

Semakin kecil nilai MAE, semakin baik prediksi rating yang dihasilkan oleh model.

##### Hasil Validasi Silang (5 fold):
| Metrik  | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Rata-rata | Std Dev |
|---------|--------|--------|--------|--------|--------|-----------|---------|
| RMSE    | 0.8991 | 0.9058 | 0.8924 | 0.8937 | 0.9003 | 0.8983    | 0.0048  |
| MAE     | 0.6920 | 0.6963 | 0.6881 | 0.6882 | 0.6925 | 0.6914    | 0.0031  |

##### Analisis Hasil:
- **Konsistensi Model**:
  - Standar deviasi kecil (±0.0048) menunjukkan performa stabil
- **Akurasi Prediksi**:
  - RMSE 0.898 → Error prediksi rata-rata ≈0.9 poin rating
  - MAE 0.691 → 68% prediksi memiliki error <1 poin rating
- Memenuhi kebutuhan untuk rekomendasi personalisasi

### Kesimpulan
1. **Content-Based Filtering** telah berhasil dievaluasi menggunakan metrik **NDCG@10** dengan hasil yang sangat memuaskan (**0.9678** untuk The Dark Knight Rises dan **0.9791** untuk The Godfather). Nilai yang mendekati 1.0 ini menunjukkan bahwa sistem mampu memberikan rekomendasi film dengan kesamaan genre yang tinggi dan urutan ranking yang optimal, dimana film paling relevan selalu berada di posisi teratas.
2. **Collaborative Filtering** dengan SVD menunjukkan performa yang solid berdasarkan evaluasi **RMSE (0.8983)** dan **MAE (0.6914)**. Nilai error yang relatif kecil ini mengindikasikan bahwa model memiliki akurasi yang baik dalam memprediksi rating pengguna, sehingga layak digunakan untuk sistem rekomendasi personalisasi.
3. Kedua pendekatan secara terpisah telah memenuhi tujuan bisnis yang ditetapkan - Content-Based untuk menemukan film serupa dan Collaborative untuk rekomendasi personal, dengan bukti metrik evaluasi yang objektif dan konsisten.
