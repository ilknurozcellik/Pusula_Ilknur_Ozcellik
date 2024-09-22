# Gerekli kütüphaneleri import edin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Veri setini yükleme
data = pd.read_csv('veri_dosyasi.csv')

# Veri türlerini ve eksik değerleri kontrol edin
print(data.info())

# Sayısal değişkenler için özet istatistikleri görün
print(data.describe())

# Eksik veri analizi
missing_data = data.isnull().sum()
print(missing_data)

# Veri görselleştirme
# Histogramlar
data.hist(bins=50, figsize=(20, 15))
plt.show()

# Değişkenler arası ilişkiler
sns.pairplot(data)

# Isı haritası
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True)
plt.show()

# Eksik verilerin işlenmesi
# Ortalama ile eksik verileri doldurma örneği
imputer = SimpleImputer(strategy='mean')
data['sütun_adı'] = imputer.fit_transform(data[['sütun_adı']])

# Kategorik verilerin kodlanması
encoder = OneHotEncoder()
kategorik_veri = encoder.fit_transform(data[['kategorik_sütun']])

# Sayısal verilerin standardizasyonu
scaler = StandardScaler()
data[['sayısal_sütun']] = scaler.fit_transform(data[['sayısal_sütun']])

# Veri ön işleme pipeline oluşturma
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['sayısal_sütun']),
        ('cat', OneHotEncoder(), ['kategorik_sütun'])
    ]
)
