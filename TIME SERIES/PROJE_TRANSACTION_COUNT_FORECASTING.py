
###############################################################
# İş Problemi
###############################################################

# Iyzico internetten alışveriş deneyimini hem alıcılar hem de satıcılar için kolaylaştıran bir finansal teknolojiler şirketidir.
# E-ticaret firmaları, pazaryerleri ve bireysel kullanıcılar için ödeme altyapısı sağlamaktadır.
# 2020 yılının son 3 ayı için merchant_id ve gün bazında toplam işlem hacmi tahmini yapılması beklenmekte.


###############################################################
# Veri Seti Hikayesi
###############################################################
#  7 üye iş yerinin 2018’den 2020’e kadar olan verileri yer almaktadır.

# Transaction : İşlem sayısı
# MerchantID : Üye iş yerlerinin id'leri
# Paid Price : Ödeme miktarı

###############################################################
# GÖREVLER
###############################################################

# Görev 1 : Veri Setinin Keşfi
            # 1. iyzico_data.csv dosyasını okutunuz. transaction_date değişkeninin tipini date'e çeviriniz.
            # 2.Veri setinin başlangıc ve bitiş tarihleri nedir?
            # 3.Her üye iş yerindeki toplam işlem sayısı kaçtır?
            # 4.Her üye iş yerindeki toplam ödeme miktarı kaçtır?
            # 5.Her üye iş yerinin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.

# Görev 2 : Feature Engineering tekniklerini uygulayanız. Yeni feature'lar türetiniz.
            # Date Features
            # Lag/Shifted Features
            # Rolling Mean Features
            # Exponentially Weighted Mean Features

# Görev 3 : Modellemeye Hazırlık
            # 1.One-hot encoding yapınız.
            # 2.Custom Cost Function'ları tanımlayınız.
            # 3.Veri setini train ve validation olarak ayırınız.

# Görec 4 : LightGBM Modelini oluşturunuz ve SMAPE ile hata değerini gözlemleyiniz.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')



###############################################################
# Görev 1 : Veri Setinin Keşfi
###############################################################

# 1. iyzico_data.csv dosyasını okutunuz. transaction_date değişkeninin tipini date'e çeviriniz.
df = pd.read_csv("time_series/iyzico_m/iyzico_data.csv")
df.drop("Unnamed: 0", axis=1,inplace=True)
df.head()
df.tail()
df["transaction_date"].dtypes
df["transaction_date"] = pd.to_datetime(df["transaction_date"])


# 2.Veri setinin başlangıc ve bitiş tarihleri nedir?
df["transaction_date"].min() # Timestamp('2018-01-01 00:00:00')
df["transaction_date"].max() # Timestamp('2020-12-31 00:00:00')

# 3.Her üye iş yerindeki toplam işlem sayısı kaçtır?
df["merchant_id"].unique()

# 4.Her üye iş yerindeki toplam ödeme miktarı kaçtır?
df.groupby("merchant_id").agg({"Total_Paid":"sum"})

# 5.üye iş yerlerinin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.
for id in df.merchant_id.unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1, title = str(id) + ' 2018-2019 Transaction Count')
    df[(df.merchant_id == id) & ( df.transaction_date >= "2018-01-01" ) & (df.transaction_date < "2019-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3, 1, 2,title = str(id) + ' 2019-2020 Transaction Count')
    df[(df.merchant_id == id) &( df.transaction_date >= "2019-01-01" )& (df.transaction_date < "2020-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.show()


###############################################################
# Görev 2 : Feature Engineering
###############################################################

########################
# Date Features
########################

def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.weekofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] =df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df

df = create_date_features(df, "transaction_date")
df.head()

# Üye iş yerlerinin yıl ve ay bazında işlem sayılarının incelenmesi
df.groupby(["merchant_id","year","month","day_of_month"]).agg({"Total_Transaction": ["sum", "mean", "median"]})

# Üye iş yerlerinin yıl ve ay bazında toplam ödeme miktarlarının incelenmesi
df.groupby(["merchant_id","year","month"]).agg({"Total_Paid": ["sum", "mean", "median"]})


########################
# Lag/Shifted Features
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91,92,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,
                       350,351,352,352,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,
                       538,539,540,541,542,
                       718,719,720,721,722])


########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby("merchant_id")['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720])


########################
# Exponentially Weighted Mean Features
########################

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720]

df = ewm_features(df, alphas, lags)
df.tail()

# Döviz kuru, fiyat teklif ind.

########################
# Black Friday - Summer Solstice
########################

df["is_black_friday"] = 0
df.loc[df["transaction_date"].isin(["2018-11-22","2018-11-23","2019-11-29","2019-11-30"]) ,"is_black_friday"]=1

df["is_summer_solstice"] = 0
df.loc[df["transaction_date"].isin(["2018-06-19","2018-06-20","2018-06-21","2018-06-22",
                                    "2019-06-19","2019-06-20","2019-06-21","2019-06-22",]) ,"is_summer_solstice"]=1




########################
# One-Hot Encoding
########################
df.head()

df = pd.get_dummies(df, columns=['merchant_id','day_of_week', 'month'])
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)


########################
# Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

########################
# Time-Based Validation Sets
########################

import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# 2020'nin 10.ayına kadar train seti.
train = df.loc[(df["transaction_date"] < "2020-10-01"), :]

# 2020'nin son 3 ayı validasyon seti.
val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date', 'id', "Total_Transaction","Total_Paid", "year" ]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

# kontrol
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

########################
# Değişken önem düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()


