
#import libraries
import string
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mp
#create a corpus file.
badwords = []
for line in open("badwords.txt"):
    for word in line.split( ):
        badwords.append(word)

import tweepy
from tweepy import OAuthHandler

# set up api keys
consumer_key = 'Consumer Key'
consumer_secret = 'Consumer Secret Key'
access_token = 'Access Token'
access_token_secret = 'Access Token Secret Key'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

#@title Default title text
search_value = "Kylie Jenner" #@param {type:"string"}

mytweet = tweepy.Cursor(api.search,q=search_value +'-filter:retweets', tweet_mode= "extended")
#mytweet= tweepy.Cursor(api.home_timeline, tweet_mode= "extended")
csvfile=open(search_value + '_test.csv', 'a')
csvWriter=csv.writer(csvfile)

for status in mytweet.items():
    print ("tweet: "+ str(status.full_text.encode('utf-8')))
    # get rid of punctuation
    tweet = status.full_text
    tweet = "".join(l for l in tweet if l not in string.punctuation)
    tweet = tweet.lower()
    bullying = "False"
    for word in tweet.split(" "):
        if word in badwords:
            bullying = "True"
            print (bullying)
            break
    if bullying == "False":
        print (bullying)
    row = tweet +","+ bullying + "\n"
    csvWriter.writerow([tweet,bullying])

df=pd.read_csv("tweets.csv")
#b=pd.read_csv("Kylie Jenner_test.csv")
df = df.astype(str)
col= ['Tweets','Classification']
df.columns=col
df.head()



