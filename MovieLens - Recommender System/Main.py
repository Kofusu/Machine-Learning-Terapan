#!/usr/bin/env python
# coding: utf-8

# # Sistem Rekomendasi: MovieLens
# ## Nama: Hendratara Pratama
# link dataset: [https://grouplens.org/datasets/movielens/](MovieLens)

# ## 1. Import Modul yang Diperlukan

# In[1]:


# Proses Data Lib
import pandas as pd 
import numpy as np

# Visualisasi Data Lib
import matplotlib.pyplot as plt
import seaborn as sns

# Sistem Rekomendasi Lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# ## 2. Menyiapkan Dataset

# Dataset yang digunakan yaitu dataset rekomendasi film dari MovieLelens yang bisa ditemukan di [GroupLens](https://grouplens.org/datasets/movielens/)

# In[2]:


get_ipython().system('curl https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o ./datasets/movielens.zip')


# Lalu Compress ZIP file data yang telah di download

# In[3]:


get_ipython().system('unzip ./datasets/movielens.zip -d ./datasets/')


# Melihat bentuk pohon dari _current work directory_

# In[5]:


get_ipython().system('tree')


# ## 3. Data Understanding

# Mebuat dataframe dari file 4 file dari MovieLens yang sudah di download lalu lihat panjang dari datanya

# In[6]:


# Memuat semua dataset yang tersedia menjadi datafame
links_df = pd.read_csv('datasets/ml-latest-small/links.csv')
movies_df = pd.read_csv('datasets/ml-latest-small/movies.csv')
ratings_df = pd.read_csv('datasets/ml-latest-small/ratings.csv')
tags_df = pd.read_csv('datasets/ml-latest-small/tags.csv')

# Melihat shape / ukuran dataframe
links_df.shape, movies_df.shape, ratings_df.shape, tags_df.shape


# Dataset berisi:
# - tags.csv -> Semua peringkat terkandung dalam file ratings.csv. Setiap baris file ini setelah baris header mewakili satu peringkat dari satu film oleh satu pengguna, dan memiliki format berikut: userId,movieId,rating,timestamp.
# 
# - movies.csv -> Semua tag terkandung dalam file tags.csv. Setiap baris file ini setelah baris header mewakili satu tag yang diterapkan ke satu film oleh satu pengguna, dan memiliki format berikut: userId,movieId,tag,timestamp.
# 
# - ratings.csv -> Informasi film terkandung dalam file movies.csv. Setiap baris file ini setelah baris header mewakili satu film, dan memiliki format berikut: movieId,title,genres.
# 
# - tags.csv -> Pengidentifikasi yang dapat digunakan untuk menautkan ke sumber data film lainnya terdapat dalam file links.csv. Setiap baris file ini setelah baris header mewakili satu film, dan memiliki format berikut: movieId,imdbId,tmdbId.

# Menampilkan summary dataframe links 

# In[7]:


# Menampilkan summary links dataframe
links_df.info()


# Menampilkan summary dataframe movie

# In[8]:


# Menampilkan summary movies dataframe
movies_df.info()


# Menampilkan summary dataframe ratings

# In[9]:


# Menampilkan summary ratings dataframe
ratings_df.info()


# Menampilkan summary dataframe tags

# In[10]:


# Menampilkan summary tags dataframe
tags_df.info()


# Dari summary data diatas diketahui bahwa:
# - userId -> ID pengguna yang telah dianonimkan | (int64)
# - movieId -> movieId adalah pengidentifikasi untuk film yang digunakan oleh https://movielens.org. Misal film Toy Story memiliki link https://movielens.org/movies/1 | (int64)
# - rating -> Peringkat dibuat dalam skala 5 bintang, dengan peningkatan setengah bintang (0,5 bintang - 5,0 bintang) | (float64)
# - timestamp -> Stempel waktu mewakili detik sejak tengah malam Waktu Universal Terkoordinasi (UTC) tanggal 1 Januari 1970 | (int64)
# - tag -> Metadata buatan dari pengguna tentang film | (object)
# - title -> Judul film | (object)
# - genres -> list genre pipe-separated | (object)
# - imdbId -> imdbId adalah pengidentifikasi untuk film yang digunakan oleh http://www.imdb.com. Misalnya film Toy Story memiliki link http://www.imdb.com/title/tt0114709/ | (int64)
# - tmdbId -> tmdbId adalah pengenal untuk film yang digunakan oleh https://www.themoviedb.org. Misal film Toy Story memiliki link https://www.themoviedb.org/movie/862 | (float64)

# Menampilkan data 5 teratas dari dataframe links

# In[11]:


# Menampilkan data 5 teratas links dataframe
links_df.head()


# Menampilkan data 5 teratas dari dataframe movies

# In[12]:


# Menampilkan data 5 teratas movies dataframe
movies_df.head()


# Menampilkan data 5 teratas dari dataframe ratings

# In[13]:


# Menampilkan data 5 teratas ratings dataframe
ratings_df.head()


# Menampilkan data 5 teratas dari dataframe tags

# In[14]:


# Menampilkan data 5 teratas tags dataframe
tags_df.head()


# Melihat panjang data dari tag yang akan digunakan untuk content based filtering

# In[15]:


# melihat unique dari tags
len(tags_df['tag'].unique())


# ## 4. Data Preprocessing

# Menggabungkan Movies dengan Tags berdasarkan MovieId
# dan menampilkan 5 data teratas

# In[16]:


# Menggabungkan Movies dengan Tags berdasarkan MovieId
tag_movies_df = movies_df.merge(tags_df, on='movieId', how='left')
tag_movies_df.head()


# Melihat total tag unik

# In[17]:


# Melihat isi tags unique
len(tag_movies_df['tag'].unique().tolist())


# Melihat data kosong dari dataframe

# In[18]:


# Melihat data not available / kosong
tag_movies_df.isna().sum()


# menghapus movieId yang duplikat untuk digunakan.

# In[19]:


# Membuang data duplikat
preparation = tag_movies_df.drop_duplicates('movieId')
preparation.head()


# konversi series yang akan digunakan menjadi list lalu akan di cek panjang dari datanya

# In[20]:


# Konversi series menjadi list
movie_id = preparation['movieId'].tolist()
title = preparation['title'].tolist()
tag = preparation['tag'].tolist()

len(movie_id), len(title), len(tag)


# Membuat daragrame baru dari list yang dibuat diatas

# In[21]:


# Membuat Dataframe untuk data 'movie_id', 'title', 'tag'
movie_new = pd.DataFrame({
    'id': movie_id,
    'movie': title,
    'tag': tag
})
movie_new.sample(5)


# Cek missing value pada dataframe yang baru dibuat

# In[22]:


# Cek missing Value
movie_new.isna().sum(), len(movie_new)


# Fill missing value dengan `UnkNown` tag

# In[23]:


# Mengganti Tag yang tidak ada dengan unknown
movie_new.fillna('UnkNown', inplace=True)
movie_new.sample(5)


# Menampilkan bar chart dari 10 tag terbanyak (kecuali ta `UnkNown`)

# In[24]:


# bar chart 10 tag terbanyak
movie_new['tag'].value_counts()[1:11].plot(kind='bar')


# ## 5. Data Preparation

# ### Untuk Content Based Filtering

# Untuk content based fiiltering digunakan tfidf sebelum diproses sebelum modelling

# In[25]:


# Inisialisasi Tfidf Vectorizer
tfidf = TfidfVectorizer()

# Transform Matrix hasil dari TFIDF
tfidf_matrix = tfidf.fit_transform(movie_new['tag']).todense()
tfidf_matrix.shape


# List tag yang diolah leh tfidf vectorizer

# In[26]:


# List Tags
tfidf.get_feature_names_out()


# Membuat dataframe hasil dari tfidf dengan tag sebagai kolom dan index 

# In[27]:


# Membuat DataFrame hasil tfidf
df = pd.DataFrame(tfidf_matrix,
                  columns=tfidf.get_feature_names_out(), 
                  index=movie_new['movie'])
df.sample(5)


# ### Untuk Collaborative Filtering

# Untuk collaborative filtering akan dilakukan dengan user-based, dataset yang digunakan yaitu movies dan ratings

# In[28]:


# Menyiapkan Rating movie dataframe
rating_movie_df = movies_df.merge(ratings_df, on='movieId', how='left')
rating_movie_df.head()


# Cek missing value pada dataframe tersebut

# In[29]:


# Cek Missing Value
rating_movie_df.isna().sum()


# menghapus missing value pada dataframe

# In[30]:


# Menghapus Missing Value
rating_movie_df.dropna(inplace=True)


# Membuang fitur `genres` dan `timestamp`

# In[31]:


# Membuang data yg tidak diperlukan
rating_movie_df.drop(['genres', 'timestamp'], axis=1, inplace=True)

rating_movie_df.head()


# ## 7. Modelling

# ### Dengan Cosine Similarity

# Membuat variable cosine similarity dari hasil tfidf diatas

# In[32]:


# Menghitung cosine similarity 
cosine_sim = cosine_similarity(np.array(tfidf_matrix)) 
cosine_sim, cosine_sim.shape


# Membuat dataframe baru dari hasil cosine similarity dengan movie sebagai index dan kolom

# In[33]:


# Membuat Dataframe dari cosine similarity dengan baris dan kolom beruma movies
cosine_sim_df = pd.DataFrame(cosine_sim, 
                             index=movie_new['movie'], 
                             columns=movie_new['movie'])
cosine_sim_df.sample(5)


# Membuat fungsi untuk rekomendasi content based filtering

# In[34]:


# Fungsi Mendapatkan Rekomendasi
def get_recommendation_cosine(nama_movie, similarity_data=cosine_sim_df, items=movie_new[['movie', 'tag']], k=10):
    # Mengambil data dengan menggunakan argpartition 
    # untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,nama_movie].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop nama_movie agar yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(nama_movie, errors='ignore')
    
    return pd.DataFrame(closest).merge(items.reset_index(drop=True)).head(k)


# Mendapatkan rekomendasi dari `Gintama: The Movie (2010)`

# In[35]:


# Pemanggilan fungsi rekomendasi dengan Cosine similarity
get_recommendation_cosine('Gintama: The Movie (2010)')


# ### Dengan KNN

# Menginisialisasi model NearestNeighbors dengan metric `euclidean` dan melakukan fitting untuk collaborative filtering

# In[36]:


# Inisialisasi Model
knn = NearestNeighbors(metric='euclidean')

# Fitting Data
knn.fit(rating_movie_df.drop(['title', 'movieId'], axis=1))


# Membuat fungsi rekomendasi dengan KNN

# In[187]:


def get_recommendation_knn(nama_movie, top_n=10,data=rating_movie_df):
    # Mengambil ID dari items
    row = data[data['title'] == nama_movie]
    movie_id = row.index.values[0]
    inputan = data.drop(['title', 'movieId'], axis=1).loc[movie_id].values.reshape(1, -1)
    
    # Mencari nilai terdekat (index rating)
    distances, neighbors = knn.kneighbors(inputan,
                                          n_neighbors=top_n)
    # Return Similiar Movie
    output = pd.DataFrame(data.loc[neighbors[0], :])
    output['distance'] = distances[0]
    return output.merge(movie_new, left_on='title', right_on='movie', how='left').drop(['id', 'movie'], axis=1)


# Mendapatkan rekomendasi `Gintama: The Movie (2010)` dari model KNN

# In[188]:


get_recommendation_knn("Gintama: The Movie (2010)")


# ## 8. Evaluasi

# Membuat evaluasi metrik precision

# In[134]:


# Fungsi untuk menghitung nilai presisi dari sistem rekomendasi
def precision(inputan, hasil_rec):
    relevant = 0
    for result in hasil_rec['tag'].values.tolist():
        if inputan['tag'] == result.lower():
            relevant += 1
    return relevant / len(hasil_rec)


# Membuat variable film untuk dilakukan evaluasi metrik

# In[117]:


# Input Evaluasi
input_movie = movie_new.loc[10]
input_movie


# ### Evaluasi KNN

# Mendapatkan rekomendasi dari `American President, The (1995)` dengan KNN

# In[189]:


# Rekomendasi KNN
recommendation_knn = get_recommendation_knn(input_movie['movie'])
recommendation_knn


# Melihat hasil presisi dari KNN

# In[174]:


# Menghitung Presisi KNN
precision(input_movie, recommendation_knn)


# KNN memiliki presisi sebesar 1

# ### Evaluasi Cosine Similarity

# Mendapatkan rekomendasi dari `American President, The (1995)` dengan Cosine Similarity

# In[171]:


# Rekomendasi Cosine Similarity
recommendation_cosine = get_recommendation_cosine(input_movie['movie'])
recommendation_cosine


# Melihat hasil presisi dari Cosine Similarity

# In[172]:


# Menghitung Presisi Cosine Similarity
precision(input_movie, recommendation_cosine)


# Cosine similarity memiliki precision sebesar 9

# ## Penutupan
# ### Referensi
# - https://scikit-learn.org/stable/
# - https://grouplens.org/datasets/movielens/
