#Gözetimsiz Öğrenme ile Müşteri Segmentasyonu
import pandas as pd

#İş Problemi
#FLO müşterilerini segmentlere ayırıp bu segmentlere göre
#pazarlama stratejileri belirlemek istiyor.
#Buna yönelik olarak müşterilerin davranışları
#tanımlanacak ve bu davranışlardaki öbeklenmelere
#göre gruplar oluşturulacak.

#Veri Seti Hikayesi
#Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında
#OmniChannel (hem online hem offline alışveriş yapan) olarak
#yapan müşterilerin geçmiş alışveriş davranışlarından elde
#edilen bilgilerden oluşmaktadır.

#master_id:Eşsiz müşteri numarası
#order_channel:Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
#last_order_channel:En son alışverişin yapıldığı kanal
#first_order_date:Müşterinin yaptığı ilk alışveriş tarihi
#last_order_date:Müşterinin yaptığı son alışveriş tarihi
#last_order_date_online:Müşterinin online platformda yaptığı son alışveriş tarihi
#last_order_date_offline:Müşterinin offline platformda yaptığı son alışveriş tarihi
#order_num_total_ever_online:Müşterinin online platformda yaptığı toplam alışveriş sayısı
#order_num_total_ever_offline:Müşterinin offline'da yaptığı toplam alışveriş sayısı
#customer_value_total_ever_offline:Müşterinin offline alışverişlerinde ödediği toplam ücret
#customer_value_total_ever_online:Müşterinin online alışverişlerinde ödediği toplam ücret
#interested_in_categories_12:Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

#Görev 1: Veriyi Hazırlama

#Adım 1: flo_data_20K.csv verisini okutunuz.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: "%.1f" %x)
from mlxtend.frequent_patterns import apriori, association_rules

df=pd.read_csv("datasets/flo_data_20k.csv")

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.info()
#Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.

#String tarihleri datetime formatına çevirme
#total_order_num: Online ve offline toplam sipariş sayısı.
#total_customer_value: Online ve offline toplam müşteri değeri.
#average_order_value: Ortalama sipariş değeri.
#days_since_last_order: Son siparişten geçen gün sayısı.
#customer_lifetime: Müşterinin aktif olduğu toplam gün sayısı.
#online_order_ratio: Toplam siparişler içinde online siparişlerin oranı.
#offline_order_ratio: Toplam siparişler içinde offline siparişlerin oranı.
#order_frequency: Müşterinin sipariş sıklığı (toplam sipariş sayısı / müşteri ömrü).
#most_frequent_category: En sık ilgi duyulan kategori.

datetime=df['last_order_date'].max()

# Yeni değişkenlerin hesaplanması
df['total_order_num'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['total_customer_value'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']
df['average_order_value'] = df['total_customer_value'] / df['total_order_num']
df['days_since_last_order'] = (datetime.now() - df['last_order_date']).dt.days
df['customer_lifetime'] = (df['last_order_date'] - df['first_order_date']).dt.days
df['online_order_ratio'] = df['order_num_total_ever_online'] / df['total_order_num']
df['offline_order_ratio'] = df['order_num_total_ever_offline'] / df['total_order_num']
df['order_frequency'] = df['total_order_num'] / df['customer_lifetime']


df.head()

targets = df[["total_order_num", "total_customer_value", "days_since_last_order",
           "average_order_value", "customer_lifetime", "offline_order_ratio",
           "online_order_ratio"]]

#Görev 2: K-Means ile Müşteri Segmentasyonu

#Adım 1: Değişkenleri standartlaştırınız.

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(targets)
df[0:5]

#Adım 2: Optimum küme sayısını belirleyiniz.
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

#Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("datasets/flo_data_20k.csv", index_col=0)

df["cluster"] = clusters_kmeans+1

df.head()


#Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.

df.groupby("cluster").agg(["count","mean","median"])


#Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu

#Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme
#sayısını belirleyiniz.

df = pd.read_csv("datasets/flo_data_20k.csv")
hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.7, color='b', linestyle='--')
plt.show()

#Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyini

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################


# Hierarchical Clustering Modeli
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()



cluster = AgglomerativeClustering(n_clusters=25, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1
df["kmeans_cluster_no"] = clusters_kmeans

