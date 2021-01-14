import tweepy #https://github.com/tweepy/tweepy
import csv
import datetime
import re

#Twitter API credentials
consumer_key='xxxx'
consumer_secret='xxxx'
access_key='xxxx'
access_secret='xxxx'

startDate = datetime.datetime(2019, 11, 1)
endDate =   datetime.datetime(2020, 10, 1)
tweets=[]


#get tweets from the tweeter
def get_all_tweets_for_user(screen_name):
    
    
    #initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    #all tweets
    alltweets = []  
    exit=0
    #max get 200 tweets
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
    #preserve most recent tweets
    alltweets.extend(new_tweets)
    
    #get last tweet id
    oldestID = alltweets[-1].id - 1
    
    #get all the tweets till start date
    while len(new_tweets) > 0 and exit==0:
        
        new_tweets = api.user_timeline(screen_name = screen_name,tweet_mode="extended",count=200,max_id=oldestID)
        
        #save recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldestID = alltweets[-1].id - 1
        tweet_date = alltweets[-1].created_at
        print(tweet_date)
       # print(startDate)
        if tweet_date < startDate:
            exit=1

    outtweets=[]
    for tweet in alltweets:
        if(tweet.created_at <= endDate and tweet.created_at >= startDate):
            try:
                text = tweet.full_text
            except AttributeError:
                text = tweet.text
        
            clean_text = re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text)
            outtweets.append([tweet.created_at, " "+clean_text])
    #write the csv  
    with open(f'{screen_name}_tweets_2019.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["created_at","text"])
        writer.writerows(outtweets)
    
    pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
    users = ["@CanadianPM","@Safety_Canada","@CDCgov"]
    #users = ["@CanadianPM"]
    for user in users:
        get_all_tweets_for_user(user)