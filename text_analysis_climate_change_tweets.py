# %% [code] {"jupyter":{"outputs_hidden":false}}
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
%matplotlib inline

# %% [code] {"jupyter":{"outputs_hidden":false}}
tsa = pd.read_csv("../input/twitter-climate-change-sentiment-dataset/twitter_sentiment_data.csv")
tsa.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
tsa.drop("sentiment", axis=1, inplace=True)
tsa.head()

# %% [markdown]
# ### LENGTH OF TWEET

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig = plt.figure(figsize=(10, 10))
tsa['message'].str.len().hist()
fig.savefig("hist-length-of-tweet.png", bbox_inches = 'tight')

# %% [markdown]
# ### NUMBER OF WORDS IN A TWEET

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig = plt.figure(figsize=(10, 10))
tsa['message'].str.split().map(lambda x : len(x)).hist()
fig.savefig("hist-no-words-of-tweet.png", bbox_inches = 'tight')

# %% [markdown]
# ### AVERAGE WORD LENGTH IN A TWEET

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig = plt.figure(figsize=(8, 8))
obj = tsa['message'].str.split().apply(lambda x : [len(i) for i in x])
obj = obj.map(lambda x : np.mean(x)) #map is only for pandas objects
obj.hist()
fig.savefig("hist-avg-wordlen-of-tweet.png", bbox_inches = 'tight')

# %% [markdown]
# ### FREQUENCY OF THE STOP WORDS

# %% [code] {"jupyter":{"outputs_hidden":false}}
# import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
# stop

# %% [code] {"jupyter":{"outputs_hidden":false}}
corpus = []
twt = tsa['message'].str.split()
twt = twt.values.tolist()
corpus = [word for i in twt for word in i]
# corpus

# %% [code] {"jupyter":{"outputs_hidden":false}}
from collections import defaultdict
dic = defaultdict(int)

for word in corpus:
    if word in stop:
        dic[word] = dic[word] + 1
        
# dic

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig = plt.figure(figsize = (20, 10))
top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:20] 
x,y = zip(*top)
plt.bar(x,y)
fig.savefig("top-stopwords-bar.png", bbox_inches="tight")

# %% [markdown]
# ### FREQUENCY OF WORDS OTHER THAN STOPWORDS

# %% [code] {"jupyter":{"outputs_hidden":false}}
import collections
from collections import Counter
import re

def username_filter(word):
    if(re.search("^@", word) == None and re.search("^((http|https)\:\/\/)", word) == None 
       and re.search("^&amp;$", word) == None and word != "RT" and word != "|" and word != "-"):
        return True
    else:
#         print(word)
        return False
    
filtered_corpus = filter(username_filter, corpus)
filtered_corpus = list(filtered_corpus)
# filtered_corpus

# %% [code] {"jupyter":{"outputs_hidden":false}}
counter = Counter(filtered_corpus)
most = counter.most_common()
# most[:100]

# print(len(most))

x, y= [], []
for word,count in most[:125]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
# print(len(x)) 
fig = plt.figure(figsize = (20, 10))
plt.xticks(rotation = 90)
plt.xlabel("Words")
plt.ylabel("Count")
sns.barplot(x = x, y = y)
fig.savefig("freq-words-barplot.png", bbox_inches = "tight")

# %% [markdown]
# ### MOST FREQUENT N-GRAMS

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.feature_extraction.text import CountVectorizer
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bow = vec.transform(corpus)
    sum_words = bow.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:140]

# %% [code] {"jupyter":{"outputs_hidden":false}}
vec = CountVectorizer(ngram_range = (2, 2)).fit(filtered_corpus)
bow = vec.transform(filtered_corpus)
sum_words = bow.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
# words_freq

# %% [code] {"jupyter":{"outputs_hidden":false}}
top_n_bigrams=get_top_ngram(tsa['message'],2)[:30]
print(top_n_bigrams)
x,y=map(list,zip(*top_n_bigrams)) 
fig = plt.figure(figsize = (20, 10))
plt.xticks(rotation = 90)
plt.xlabel("Bigrams")
plt.ylabel("Count")
sns.barplot(x = x, y = y)
fig.savefig("freq-bigrams-barplot.png", bbox_inches = "tight")

# %% [code] {"jupyter":{"outputs_hidden":false}}
top_n_trigrams=get_top_ngram(tsa['message'],3)[:25]
print(top_n_trigrams)
x,y=map(list,zip(*top_n_trigrams)) 
fig = plt.figure(figsize = (20, 10))
plt.xticks(rotation = 90)
plt.xlabel("Trigrams")
plt.ylabel("Count")
sns.barplot(x = x, y = y)
fig.savefig("freq-trigrams-barplot.png", bbox_inches = "tight")

# %% [markdown]
# ### TOPIC MODELLING WITH pyLDAvis

# %% [code] {"jupyter":{"outputs_hidden":false}}
import nltk
import gensim
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import punkt
from nltk import wordnet
# nltk.download('punkt')
# nltk.download('wordnet')

# %% [code] {"jupyter":{"outputs_hidden":false}}
def preprocess_tweets(df):
    corpus = []
    stem = PorterStemmer()
    lem = wordnet.WordNetLemmatizer()
    for tweet in df['message']:
        words = [w for w in word_tokenize(tweet) if ((w.lower() not in stop) and username_filter(w))]
        words = [lem.lemmatize(w) for w in words if len(w) > 2]
        corpus.append(words)
    return corpus

corpus = preprocess_tweets(tsa)
# corpus

# %% [code] {"jupyter":{"outputs_hidden":false}}
dic = gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]

# %% [code] {"jupyter":{"outputs_hidden":false}}
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics = 4, id2word = dic, passes = 10, workers = 2)
lda_model.show_topics()

# %% [markdown]
# ### VISUALIZING RESULTS OF LDA

# %% [code] {"jupyter":{"outputs_hidden":false}}
import pyLDAvis
from pyLDAvis import gensim_models

# %% [code] {"jupyter":{"outputs_hidden":false}}
# fig = plt.figure(figsize = (20, 20))
pyLDAvis.enable_notebook()
vis = gensim_models.prepare(lda_model, bow_corpus, dic)
vis

# %% [code] {"jupyter":{"outputs_hidden":false}}
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()
    fig.savefig("wordcloud.png", bbox_inches = "tight")

show_wordcloud(corpus)

# %% [code] {"jupyter":{"outputs_hidden":false}}
