
#################################################################
#Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
#################################################################
#################################################################
#İş Problemi
#################################################################
#Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi
#sınıf (average, highlighted) oyuncu olduğunu tahminleme.

#################################################################
#Veri Seti Hikayesi
#################################################################

#Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların
#değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren
#bilgilerden oluşmaktadır.

#task_response_id :Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id:İlgili maçın id'si
#evaluator_id:Değerlendiricinin(scout'un) id'si
#player_id:İlgili oyuncunun id'si
#position_id:İlgili oyuncunun o maçta oynadığı pozisyonun id’si
#1: Kaleci
#2: Stoper
#3: Sağ bek
#4: Sol bek
#5: Defansif orta saha 6: Merkez orta saha 7: Sağ kanat
#8: Sol kanat
#9: Ofansif orta saha 10: Forvet
#analysis_id:Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
#attribute_id:Oyuncuların değerlendirildiği her bir özelliğin id'si
#attribute_value:Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)
#potential_label:Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

#Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

attributes = pd.read_csv('datasets/scoutium_attributes.csv', sep=';')
potential_labels = pd.read_csv('datasets/scoutium_potential_labels.csv', sep=';')



#Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
#("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme
#işlemini gerçekleştiriniz.)

df = pd.merge(attributes, potential_labels, how="inner", on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'])

#Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df=df[df["position_id"]!=1]

#Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
#(below_average sınıfı tüm verisetinin %1'ini oluşturur)

df=df[df["potential_label"]!="below_average"]

#Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo
#oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

#Adım 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id”
#ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot
#table’ı oluşturunuz.

pivot_table=df.pivot_table(index=["player_id","position_id","potential_label"]
                           ,columns="attribute_id", values="attribute_value")

#Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve
#“attribute_id” sütunlarının isimlerini stringe çeviriniz.

pivot_table=pivot_table.reset_index()

current_columns = pivot_table.columns

new_columns = [str(col) if 'attribute_id' in str(col) else col for col in current_columns]

pivot_table.columns = new_columns

#Adım6: LabelEncoderfonksiyonunukullanarak“potential_label”kategorilerini
#(average,highlighted)sayısalolarakifadeediniz.



def label_encoder(dataframe, label_col):
    labelencoder = LabelEncoder()
    dataframe[label_col] = labelencoder.fit_transform(dataframe[label_col])
    return dataframe

df = label_encoder(df, 'potential_label')

#Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir
#listeye atayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # Kategorik kolonları belirleme
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Sayısal kolonları belirleme
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Yorum satırlarını (prints) kaldırdım, ihtiyacınıza göre tekrar ekleyebilirsiniz.
    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')

    # Kategori, sayısal ve kategorik ama kardinal olan kolonları döndürme
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için
#StandardScaler uygulayınız.

X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

#Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların
#potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.
#(Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)
X.isnull().sum()
X.dropna(inplace=True)
def base_models(X, y, scoring="accuracy"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1)),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
base_models(X, y)

# Hyperparameter Optimization


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {
    "max_depth": [8, 15, None],
    "max_features": [5, 7, "sqrt"],  # 'auto' yerine 'sqrt' kullanın
    "min_samples_split": [15, 20],
    "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)




#Adım 10: Değişkenlerin önem düzeyini belirten feature_importance
#fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(RF, X)
plot_importance(GBM, X)
plot_importance(XGBoost, X)
plot_importance(LightGBM, X)







