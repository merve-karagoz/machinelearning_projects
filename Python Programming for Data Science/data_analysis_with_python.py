###############################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
###############################################
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

#############################################
# NUMPY
#############################################

# Neden NumPy? (Why Numpy?)
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)

#############################################
# Neden NumPy?
#############################################
import numpy as np
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b




#############################################
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
#############################################
import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))

#############################################
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
#############################################
import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

#############################################
# Yeniden Şekillendirme (Reshaping)
#############################################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)


#############################################
# Index Seçimi (Index Selection)
#############################################
import numpy as np
a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3, 5))

m[0, 0]
m[1, 1]
m[2, 3]

m[2, 3] = 999

m[2, 3] = 2.9

m[:, 0]
m[1, :]
m[0:2, 0:3]

#############################################
# Fancy Index
#############################################
import numpy as np

v = np.arange(0, 30, 3)
v[1]
v[4]

catch = [1, 2, 3]

v[catch]

#############################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

#######################
# Klasik döngü ile
#######################
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#######################
# Numpy ile
#######################
v < 3

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]

#############################################
# Matematiksel İşlemler (Mathematical Operations)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)
v = np.subtract(v, 1)

#######################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
#######################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

#############################################
# PANDAS
#############################################

# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look at Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri

#############################################
# Pandas Series
#############################################
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)


#############################################
# Veri Okuma (Reading Data)
#############################################
import pandas as pd

df = pd.read_csv("datasets/advertising.csv")
df.head()
# pandas cheatsheet

#############################################
# Veriye Hızlı Bakış (Quick Look at Data)
#############################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()


#############################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#############################################
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# df = df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis=0, inplace=True)

#######################
# Değişkeni Indexe Çevirmek
#######################

df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

#######################
# Indexi Değişkene Çevirmek
#######################

df.index

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

df.reset_index().head()
df = df.reset_index()
df.head()

#######################
# Değişkenler Üzerinde İşlemler
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())


df[["age"]].head()
type(df[["age"]].head())

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, ~df.columns.str.contains("age")].head()


#######################
# iloc & loc
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3]
df.iloc[0, 0]

# loc: label based selection
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]


#######################
# Koşullu Seçim (Conditional Selection)
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()

#############################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
#############################################

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})


#######################
# Pivot table
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)


#############################################
# Apply ve Lambda
#############################################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# df.loc[:, ["age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()

#############################################
# Birleştirme (Join) İşlemleri
#############################################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)

#######################
# Merge ile Birleştirme İşlemleri
#######################

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3, df4)


#############################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
#############################################

#############################################
# MATPLOTLIB
#############################################

# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: hist, boxplot


#############################################
# Kategorik Değişken Görselleştirme
#############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df['sex'].value_counts().plot(kind='bar')
plt.show()

#############################################
# Sayısal Değişken Görselleştirme
#############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

#############################################
# Matplotlib'in Özellikleri
#############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#######################
# plot
#######################

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()



#######################
# marker
#######################

y = np.array([13, 28, 11, 100])

plt.plot(y, marker='o')
plt.show()

plt.plot(y, marker='*')
plt.show()

markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']

#######################
# line
#######################

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashdot", color="r")
plt.show()

#######################
# Multiple Lines
#######################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

#######################
# Labels
#######################

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
# Başlık
plt.title("Bu ana başlık")

# X eksenini isimlendirme
plt.xlabel("X ekseni isimlendirmesi")

plt.ylabel("Y ekseni isimlendirmesi")

plt.grid()
plt.show()

#######################
# Subplots
#######################

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)
plt.show()


# 3 grafiği bir satır 3 sütun olarak konumlamak.
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()









#############################################
# SEABORN
#############################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df['sex'].value_counts().plot(kind='bar')
plt.show()


#############################################
# Sayısal Değişken Görselleştirme
#############################################

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. Genel Resim
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df = sns.load_dataset("flights")
check_df(df)


#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts()
df["sex"].unique()
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("sdfsdfsdfsdfsdfsd")
    else:
        cat_summary(df, col, plot=True)


df["adult_male"].astype(int)


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)




def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "adult_male", plot=True)





def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")



#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]



df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#############################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# docstring

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)



# BONUS
df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


for col in num_cols:
    num_summary(df, col, plot=True)




#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")


for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

#######################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#######################


df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


target_summary_with_cat(df, "survived", "pclass")


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)



#######################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#######################


df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "survived","age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)



#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

cor_matrix = df.corr().abs()


#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN


upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)

type(adsa)













































