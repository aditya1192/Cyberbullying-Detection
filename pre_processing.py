
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mp

train=pd.read_csv("tweets.csv")

train.describe()

train.head()

#adding header
col= ['Tweets','Classification']
train.columns=col
train.head()

train

train = train.astype(str)

#Number of Words
train['word_count'] = train['Tweets'].apply(lambda x: len(str(x).split(" ")))
train[['Tweets','Classification','word_count']].head()

#Character Count
train['char_count'] = train['Tweets'].str.len() ## this also includes spaces
train[['Tweets','Classification','char_count']].head()

#Average Word Length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['Tweets'].apply(lambda x: avg_word(x))
train[['Tweets','avg_word']].head()

#Number of stopwords
import nltk
nltk.download('stopwords')# downlaoding the package.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stop = stopwords.words('english')

train['stopwords'] = train['Tweets'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['Tweets','stopwords']].head()

#Number of special characters
train['hastags'] = train['Tweets'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['Tweets','hastags']].head()

#Transform tweets to Lower Case
train['Tweets'] = train['Tweets'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['Tweets'].head()

#Removing Punctuations
train['Tweets'] = train['Tweets'].str.replace('[^\w\s]','')
train['Tweets'].head()

# Removing stopwords
stop = stopwords.words('english')
train['Tweets'] = train['Tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['Tweets'].head()

#Common Word Removal. 
freq = pd.Series(' '.join(train['Tweets']).split()).value_counts()[:10]
freq

train['Tweets'] = train['Tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['Tweets'].head()

train['Tweets'] = train['Tweets'].str.replace("https", " ")
train['Tweets'].head()

freq1 = pd.Series(' '.join(train['Tweets']).split()).value_counts()
freq1

freq1 = pd.Series(' '.join(train['Tweets']).split()).value_counts()[-13186:]
freq1

freq1 = list(freq1.index)
train['Tweets'] = train['Tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in freq1))
train['Tweets'].head()

check= pd.Series(' '.join(train['Tweets']).split()).value_counts()[:]
check

#Spelling Correction
from textblob import TextBlob
train['Tweets'[:5223]].apply(lambda x: str(TextBlob(x).correct()))
train.head()

train



train.describe()

train.to_csv('cleaned_Data', encoding='utf-8', index=False)