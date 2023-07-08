#!/usr/bin/env python
# coding: utf-8

# # Predictive Analytics Titanic by Hendratara Pratama

# ## 1. Data Understanding

# ### Import Exploratory Data Analysis Library 

# In[1]:


# Technical Library
import pandas as pd
import numpy as np

# Visualisation Library
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Menyiapkan Data
# Data yang dipakai berasal dari Kaggle yang dapat dilihat pada [tautan berikut](https://www.kaggle.com/c/titanic/data). Data tersebut disimpan dalam folder "./datasets", dan terdapat 2 dataset yaitu Train set dan Test set. Selanjutnya, Menyiapkan data dari csv lalu menampilkan data 5 teratas.

# In[2]:


df_train = pd.read_csv("datasets/train.csv")
df_train.head()


# ### Exploratory Data Analysis
# 
# Pada dataset Titanic ini terdapat variabel:
# 
# - Survived -> status bertahan hidup (0 = Tidak, 1 = Ya)
# - Pclass -> Kelas Tiket (1 = pertama, 2 = kedua, 3 = ketiga)
# - Name -> Nama Penumpang
# - Sex -> jenis kelamin / gender
# - Age -> umur dalam tahun
# - Sibsp -> Jumlah pasangan / saudara di dalam kapal
# - Parch -> jumlah orangtua / anak di dalam kapal
# - Ticket -> nomor tiket
# - Fare -> Tarif penumpang
# - Cabin -> Nomor Kabin
# - Embarked -> pelabuhan awal naik (C = Cherbourg, Q = Queenstown, S = Southampton)
# 
# dengan detail informasi data pada cell berikut:

# In[3]:


# Shape / Ukuran datasets
print('Train Set:', df_train.shape)


# In[4]:


# Train Dataset
df_train.info()


# Terdapat 7 numerikal fitur yaitu: `PassengerId`, `Survived`, `Pclass`, `Age`, `Sibsp`, `Parch`, dan `Fare` lalu ada 5 kategorikal fitur yaitu: `Name`, `Sex`, `Ticket`, `Cabin`, dan `Embarked`.

# In[5]:


# Train Dataset
df_train.describe()


# keterangan:
# 
# - count adalah jumlah sampel pada data.
# - mean adalah nilai rata-rata.
# - std adalah standar deviasi.
# - min yaitu nilai minimum setiap kolom.
# - 25% adalah kuartil* pertama (Q1)
# - 50% adalah kuartil* kedua (Q2) atau median (nilai tengah).
# - 75% adalah kuartil* ketiga (Q3).
# - Max adalah nilai maksimum
# 
# ** Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.*
# 

# ### Menangani Missing Value

# In[6]:


# Cek Missing Value pada Train Set
df_train.isna().sum()


# Karena banyak missing value, maka kita akan mengubah kategorikal value nya menjadi `U` untuk unknown dan median untuk numerikal value

# In[7]:


# Handling missing value train set [Age, Cabin, Embarked]
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Cabin'] = df_train['Cabin'].fillna('U')
df_train['Embarked'] = df_train['Embarked'].fillna('U')
df_train.head()


# In[8]:


df_train.shape


# ### Univariate Analysis

# Membagi fitur pada dataset menjadi numerical dan kategorikal.

# In[9]:


categorical = ['Embarked', 'Sex', 'Survived']
numerical = ['Pclass', 'SibSp', 'Parch', 'Fare']


# Analysis fitur: Embarked 

# In[10]:


count = df_train[categorical[0]].value_counts()
percent = 100*df_train[categorical[0]].value_counts(normalize=True)
df = pd.DataFrame({
    'Jumlah Sampel': count,
    'Percentase': percent.round(2)
})
print(df)
count.plot(kind='bar', title=categorical[0])


# Analisis fitur: Sex

# In[11]:


count = df_train[categorical[1]].value_counts()
percent = 100*df_train[categorical[1]].value_counts(normalize=True)
df = pd.DataFrame({
    'Jumlah Sampel': count,
    'Percentase': percent.round(2)
})
print(df)
count.plot(kind='bar', title=categorical[1])


# In[12]:


count = df_train[categorical[2]].value_counts()
percent = 100*df_train[categorical[2]].value_counts(normalize=True)
df = pd.DataFrame({
    'Jumlah Sampel': count,
    'Percentase': percent.round(2)
})
print(df)
count.plot(kind='bar', title=categorical[2])


# Analisis fitur numerikal: Pclass, SibSp, Parch, Fare

# In[13]:


df_train[numerical].hist(bins=20, figsize=(14, 8))
plt.show()


# ### Multivariate Analysis

# In[14]:


plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(), annot=True, fmt='.2f')
plt.show()


# In[15]:


for col in categorical:
  sns.catplot(x=col, 
              y="Survived", 
              data=df_train,
              aspect=3,
              height=3,
              palette='Set3',
              dodge=False,
              kind='bar')
  plt.title(f"Rata Rata Survived Relatif Pada {col}")


# ## 2. Data Preparation

# menghapus fitur yang tidak diperlukan seperti `Name`, `PassengerId`, `Ticket`, dan `Cabin`. Lalu replace `Embarked`, dan `Sex` menjadi numerik 

# In[16]:


# Menghapus Fitur Name dan PassengerId
df_train.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)
df_train.head()


# Mengubah fitur menjadi numerikal

# In[17]:


# Mengubah Embarked dengan One Hot Encoding
df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix='Embarked')], axis=1)
df_train.drop(['Embarked'], inplace=True, axis=1)


# In[18]:


# Mengubah Sex dengan One Hot Encoding
df_train = pd.concat([df_train, pd.get_dummies(df_train['Sex'], prefix='Sex')], axis=1)
df_train.drop(['Sex'], inplace=True, axis=1)


# In[19]:


df_train.head()


# Split train data dan valid data dengan train test split

# In[20]:


from sklearn.model_selection import train_test_split

x = df_train.drop(['Survived'], axis=1)
y= df_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, test_size=0.2)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# Melakukan Normalisasi pada fitur `Fare` dan `Age` menggunakan StandardScaler yang disediakan oleh library Scikit Learn. Dengan rumus:
# 
# z = (x - u) / s
# 
# dimana `x` adalah nilai dari tiap tiap data, `u` adalah mean dari data, dan `s` adalah standar deviasi dari data.

# In[21]:


from sklearn.preprocessing import StandardScaler

feat = ['Fare', 'Age']

# Train Set
scaler = StandardScaler()
x_train[feat] = scaler.fit_transform(x_train[feat].values)
x_train.head()


# In[22]:


# Test Set
x_test[feat] = scaler.transform(x_test[feat].values)
x_test.head()


# ## 3. Model Development
# model yang akan digunakan untuk klasifikasi ini yaitu random KNN dan RandmForest

# In[39]:


# Menyiapkan Dataframe Accuracy
models = pd.DataFrame(columns=['Train Acc', 'Test Acc', 'F1', 'Recall', 'Precision'], index=['KNN', 'RandomForest'])
models


# - Metode KNN

# In[40]:


from sklearn.metrics import precision_score, recall_score, f1_score


# In[41]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

models.loc['KNN', 'Train Acc'] = knn.score(x_train, y_train)
models.loc['KNN', 'Test Acc'] = knn.score(x_test, y_test)
models.loc['KNN', 'F1'] = f1_score(y_test, knn.predict(x_test))
models.loc['KNN', 'Recall'] = recall_score(y_test, knn.predict(x_test))
models.loc['KNN', 'Precision'] = precision_score(y_test, knn.predict(x_test))


# - Metode Random Forest

# In[42]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

models.loc['RandomForest', 'Train Acc'] = rf.score(x_train, y_train)
models.loc['RandomForest', 'Test Acc'] = rf.score(x_test, y_test)
models.loc['RandomForest', 'F1'] = f1_score(y_test, rf.predict(x_test))
models.loc['RandomForest', 'Recall'] = recall_score(y_test, rf.predict(x_test))
models.loc['RandomForest', 'Precision'] = precision_score(y_test, rf.predict(x_test))


# Menampilkan hasil baseline model

# In[43]:


models


# In[44]:


fig, ax = plt.subplots(figsize=(10,6))
models.sort_values(by='Test Acc', ascending=False).plot(kind='barh', ax=ax, zorder=3)
plt.show()


# ### Melakukan Hyper Parameter Tuning untuk KNN

# In[45]:


from sklearn.model_selection import GridSearchCV

parameter = {
    'n_neighbors': [2, 4, 6, 8, 10],
    'n_jobs': [-1]
}

knn = KNeighborsClassifier()
clf = GridSearchCV(knn, param_grid=parameter)
clf.fit(x_train, y_train)

clf.best_params_


# In[46]:


knn = KNeighborsClassifier(n_neighbors=6, n_jobs=-1)
knn.fit(x_train, y_train)

knn.score(x_test, y_test)


# ## 4. Evaluasi Model

# Melihat classification report pada kedua base Model

# In[47]:


from sklearn.metrics import classification_report

# Model Random Forest
print(classification_report(y_test, rf.predict(x_test)))


# In[48]:


# Model KNN

print(classification_report(y_test, knn.predict(x_test)))


# Melihat report classification

# In[49]:


y_pred = clf.predict(x_test)


# Hasil prediksi

# In[50]:


y_pred


# In[51]:


print(classification_report(y_test, y_pred))


# Terdapat nilai preicision, recall, dan f1-score untuk metrics yang dipakai pada algoritma KNN

# Kesimpulan:
# 
# Random Forest memiliki tingkat accuracy lebih tinggi dibandingkan dengan KNN untuk klasifikasi survived titanic
