# Pusula_DogaMelek_Kusun
Doğa Melek Küsün
# Talent Academy Case Study – Veri Ön İşleme ve EDA

# 1. Proje Amacı
Bu çalışma, Talent Academy tarafından sağlanan hasta verilerini analiz ederek eksik değerlerin giderilmesi, kategorik verilerin sayısallaştırılması ve sayısal değişkenlerin ölçeklenmesini içermektedir.  
Hedef değişkenimiz: `TedaviSuresi`

---

# 2. Veri Seti Yükleme

```python
import pandas as pd

df = pd.read_excel("Talent_Academy_Case_DT_2025.xlsx")
df.head()
# İlk ve son 5 satır
df.head()
df.tail()

# Veri tipleri ve sütun isimleri
df.dtypes
df.columns

# Eksik değer sayısı
df.isnull().sum()

# Eksik değerlerin görselleştirilmesi
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull(), cbar=False)
plt.show()
 # Hedef değişkenin sayısallaştırılması
df["TedaviSuresi"] = pd.to_numeric(df["TedaviSuresi"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

# Uygulama süresinin sayısal hale getirilmesi
df["UygulamaSuresi"] = pd.to_numeric(df["UygulamaSuresi"].astype(str).str.extract(r'(\d+)')[0], errors='coerce')


num_cols = ["Yas", "TedaviSuresi", "UygulamaSuresi"]
df[num_cols].hist(figsize=(10,6), bins=20)
plt.show()


from sklearn.impute import KNNImputer

num_imputer = KNNImputer(n_neighbors=5)
df[num_cols] = num_imputer.fit_transform(df[num_cols])


# One-Hot Encoding
df = pd.get_dummies(df, columns=["Cinsiyet", "KanGrubu", "Uyruk", "Bolum"], drop_first=True)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
for col in ["KronikHastalik", "Alerji", "Tanilar", "UygulamaYerleri"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(columns=["TedaviSuresi", "HastaNo"])
y = df["TedaviSuresi"]

print("Hazır veri seti boyutu:", X.shape, y.shape)

sns.heatmap(df.isnull(), cbar=False)
plt.show()

