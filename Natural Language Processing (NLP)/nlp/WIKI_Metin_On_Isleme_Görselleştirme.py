# Text Preprocessing and Visualization for WIKIPEDIA Texts

# 1. Text Preprocessing
# 2. Text Visualization

from warnings import filterwarnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 1. Text Preprocessing
df = pd.read_csv("wiki_data.csv")
df.head()

# Normalizing Case Folding
df['text'] = df['text'].str.lower()

# Punctuations
df['text'] = df['text'].str.replace('[^\w\s]', '')

# Numbers
df['text'] = df['text'].str.replace('\d', '')

# Stopwords
import nltk
# nltk.download('stopwords')

sw = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rarewords
temp_df = pd.Series(' '.join(df['text']).split()).value_counts()

drops = temp_df[temp_df <= 2000]

df['text'] = df['text'].apply(lambda  x: " ".join(x for x in x.split() if x not in drops))

# Tokenization
# nltk.download("punkt")

df["text"].apply(lambda x: TextBlob(x).words).head()

# Lemmatization
# nltk.download('wordnet')
df["text"] = df["text"].apply(lambda  x: " ".join([Word(word).lemmatize() for word in x.split()]))


# 2. Text Visualization

# Terim Frekanslarının Hesaplanması
tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

# Barplot

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# Wordcloud
text = " ".join(i for i in df.text)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()





