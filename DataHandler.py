import nltk

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date, timedelta
from pyspark.sql.functions import collect_list, concat_ws, udf
from pyspark.sql.types import StructType, IntegerType, StringType, StructField, DateType, TimestampType, DoubleType
from pyspark import SparkContext, SparkConf, RDD
from pyspark.sql import SparkSession, functions as f
import re
import sys
import utils
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import string, random

# nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
schemaTweets = StructType([
    StructField("Date", DateType(), False),
    StructField("Time", TimestampType(), False),
    StructField("ID1", TimestampType(), False),
    StructField("ID2", StringType(), True),
    StructField("Name", StringType(), True),
    StructField("location", StringType(), True),
    StructField("TweetBody", StringType(), False)
])


def get_spark_session():
    return SparkSession \
        .builder.master("local") \
        .appName('SparkMapReduceExample') \
        .getOrCreate()


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


#
# positive_tweets = twitter_samples.strings('positive_tweets.json')
# negative_tweets = twitter_samples.strings('negative_tweets.json')
# text = twitter_samples.strings('tweets.20150430-223406.json')
# tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
#
# stop_words = stopwords.words('english')
#
# positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
# negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
#
# positive_cleaned_tokens_list = []
# negative_cleaned_tokens_list = []
#
# for tokens in positive_tweet_tokens:
#     positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
#
# for tokens in negative_tweet_tokens:
#     negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
#
# all_pos_words = get_all_words(positive_cleaned_tokens_list)
#
# freq_dist_pos = FreqDist(all_pos_words)
# print(freq_dist_pos.most_common(10))
#
# positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
# negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
#
# positive_dataset = [(tweet_dict, "Positive")
#                     for tweet_dict in positive_tokens_for_model]
#
# negative_dataset = [(tweet_dict, "Negative")
#                     for tweet_dict in negative_tokens_for_model]
#
# dataset = positive_dataset + negative_dataset
#
# random.shuffle(dataset)
#
# train_data = dataset[:7000]
# test_data = dataset[7000:]
#
# classifier = NaiveBayesClassifier.train(train_data)
#
# print("Accuracy is:", classify.accuracy(classifier, test_data))
#
# print(classifier.show_most_informative_features(10))

stop_words = stopwords.words('english')
spark = get_spark_session()
sc = spark.sparkContext
#### Spark Accumulators to count tweets without extra stages  ####
tweetCount = sc.accumulator(0)
negCount = sc.accumulator(0)
posCount = sc.accumulator(0)
neuCount = sc.accumulator(0)


# def classifyer(val):
#     return classifier.classify(dict([token, True] for token in val))


def classifyer_pos(val):
    if val != None:
        return vader.polarity_scores(val)["pos"]
    else:
        val = preprocess_tweet(val)
        return vader.polarity_scores(val)["pos"]


def classifyer_neg(val):
    if val != None:
        return vader.polarity_scores(val)["neg"]
    else:
        val = preprocess_tweet(val)
        return vader.polarity_scores(val)["neg"]


def classifyer_neut(val):
    if val != None:
        return vader.polarity_scores(val)["neu"]
    else:
        val = preprocess_tweet(val)
        return vader.polarity_scores(val)["neu"]


def classifyer_compound(val):
    if val != None:
        return vader.polarity_scores(val)["compound"]
    else:
        val = preprocess_tweet(val)
        return vader.polarity_scores(val)["compound"]


def preprocess_word(word):
    # Remove punctuation

    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def tweet_sentimate_outcome(val):
    if (val <= 1 and val > 0.33):
        posCount.add(1)
        return 1
    if (val <= 0.33 and val >= -0.33):
        neuCount.add(1)
        return 0
    if (val <= -0.33 and val >= -1):
        negCount.add(1)
        return -1
    return null


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    tweetCount.add(1)
    if tweet != None:
        processed_tweet = []
        # Convert to lower case
        tweet = tweet.lower()
        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')
        # Replace emojis with either EMO_POS or EMO_NEG
        tweet = handle_emojis(tweet)
        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)
        words = tweet.split()

        for word in words:
            word = preprocess_word(word)
            if is_valid_word(word):
                if word not in stop_words:
                    word = str(PorterStemmer().stem(word))
                    processed_tweet.append(word)

        return ' '.join(processed_tweet)
    else:
        return ""
def preprocess_tweet_for_wc(tweet):
    tweetCount.add(1)
    if tweet != None:
        processed_tweet = []

        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')
        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)
        words = tweet.split()

        for word in words:
            word = preprocess_word(word)
            if is_valid_word(word):
                if word.lower() not in stop_words:
                    processed_tweet.append(word)

        return ' '.join(processed_tweet)
    else:
        return ""

def main():

    data_frame = spark.read.format("csv").load('OsamaNaridGOT_7_C_before_SA-543000.csv')
    bag_of_Wrods = data_frame.select('_c6')

    # bag_of_Wrods.coalesce(1).write.format("text").option("header", "false").mode("overwrite").save("output.txt")

#     data_frame.show()
#     data_frame.printSchema()
#     tweet_process_udf = udf(preprocess_tweet, StringType())
    tweet_process_for_wc_udf = udf(preprocess_tweet_for_wc, StringType())
#     tweet_analyzer_Neg = udf(classifyer_neg, StringType())
#     tweet_analyzer_Pos = udf(classifyer_pos, StringType())
#     tweet_analyzer_Neut = udf(classifyer_neut, StringType())
#     tweet_analyzer_Comp = udf(classifyer_compound, StringType())
#     tweet_outcome = udf(tweet_sentimate_outcome, StringType())
#     data_frame = data_frame.withColumn('after', tweet_process_udf('_c6'))
    data_frame = data_frame.withColumn('post_pross', tweet_process_for_wc_udf('_c6'))
#     data_frame.show()
#
#     wordcount_df = data_frame.withColumn('word', f.explode(f.split(f.col('after'), ' '))) \
#         .groupBy('word') \
#         .count() \
#         .sort('count', ascending=False)
#     wordcount_df.printSchema()
#     wordcount_df.show()
#   #  wordcount_df.coalesce(1).write.mode('overwrite').csv('vocabulary', header='true')#action
#     data_frame = data_frame.withColumn('sentiment', tweet_analyzer_udf('after'))
#     data_frame = data_frame.withColumn('Negative Score', tweet_analyzer_Neg('_c6'))
#     data_frame = data_frame.withColumn('Positve score', tweet_analyzer_Pos('_c6'))
#     data_frame = data_frame.withColumn('Neutral score', tweet_analyzer_Neut('_c6'))
#     data_frame = data_frame.withColumn('Compound score', tweet_analyzer_Comp('_c6'))
#     data_frame = data_frame.withColumn('outcome', tweet_outcome('Compound score'))
#     data_frame.show()
#     data_frame.printSchema()
# #    data_frame.coalesce(1).write.mode('overwrite').csv('proccesed_tweets', header='true')  # action
#     print("tweet count = " + str(tweetCount.value))
#     print("pos tweet count = " + str(posCount.value))
#     print("neg tweet count = " + str(negCount.value))
#     print("neu tweet count = " + str(neuCount.value))
#
#     natural_tweets = data_frame.filter(data_frame.outcome == 0).groupBy("_c0").count()
#     natural_tweets = natural_tweets.selectExpr("_c0 as Date1", "count as naturalCount")
#     positve_tweets = data_frame.filter(data_frame.outcome == 1).groupBy("_c0").count()
#     positve_tweets = positve_tweets.selectExpr("_c0 as Date", "count as positiveCount")
#     negative_tweets = data_frame.filter(data_frame.outcome == -1).groupBy("_c0").count()
#     negative_tweets = negative_tweets.selectExpr("_c0 as Date2", "count as negativeCount")
#     natural_tweets.printSchema()
#     positve_tweets.printSchema()
#     negative_tweets.printSchema()
#     all_tweets_byday = (natural_tweets.join(positve_tweets, natural_tweets.Date1 == positve_tweets.Date))
#     all_tweets_byday = all_tweets_byday.select("Date", "naturalCount", "positiveCount")
#     all_tweets_byday = all_tweets_byday.join(negative_tweets, negative_tweets.Date2 == all_tweets_byday.Date)
#     all_tweets_byday = all_tweets_byday.select("Date", "naturalCount", "positiveCount", "negativeCount")
#     all_tweets_byday.show()
#     all_tweets_byday.coalesce(1).write.mode('overwrite').csv('alltweetcount', header='true')
#     bag_of_Wrods = data_frame.select('after')
#
#     bag_of_Wrods.coalesce(1).write.format("text").option("header", "false").mode("overwrite").save("output.txt")
#
#     # todo daybyday aggrication
#     # todo count positive negative and natural
#     #
    groupeby_loc_df = data_frame.groupby('_c5').agg(collect_list('post_pross').alias("tweet"))
    groupeby_loc_df2 = data_frame.groupby('_c5').count()
    groupeby_loc_df2 = groupeby_loc_df2.selectExpr("_c5 as Location", "count as count")
    groupeby_loc_df = groupeby_loc_df.join(groupeby_loc_df2, groupeby_loc_df._c5 == groupeby_loc_df2.Location).sort(
        'count', ascending=False)
    groupeby_loc_df = groupeby_loc_df.withColumn("tweet", concat_ws(" ", "tweet"))
    groupeby_loc_df = groupeby_loc_df.limit(3)
    # groupeby_loc_df.show()
    # groupeby_loc_df.printSchema()
    groupeby_loc_df.select('Location', 'tweet', 'count')
    groupeby_loc_df.coalesce(1).write.mode('overwrite').csv('tweets_by_top_3_location',header='true')  ##action

#     natural_tweets = data_frame.filter(data_frame.outcome == 0).groupBy("_c0", "_c5").count()
#     natural_tweets = natural_tweets.selectExpr("_c0 as Date1", "_c5 as Location1", "count as naturalCount")
#     positve_tweets = data_frame.filter(data_frame.outcome == 1).groupBy("_c0", "_c5").count()
#     positve_tweets = positve_tweets.selectExpr("_c0 as Date", "_c5 as Location", "count as positiveCount")
#     negative_tweets = data_frame.filter(data_frame.outcome == -1).groupBy("_c0", "_c5").count()
#     negative_tweets = negative_tweets.selectExpr("_c0 as Date2", "_c5 as Location2", "count as negativeCount")
#     all_tweets_byday = (natural_tweets.join(positve_tweets, (natural_tweets.Date1 == positve_tweets.Date) & (
#                 natural_tweets.Location1 == positve_tweets.Location)))
#     all_tweets_byday = all_tweets_byday.select("Date", "Location1", "naturalCount", "positiveCount")
#     all_tweets_byday = all_tweets_byday.join(negative_tweets, (negative_tweets.Date2 == all_tweets_byday.Date) & (
#                 negative_tweets.Location2 == all_tweets_byday.Location1))
#     all_tweets_byday = all_tweets_byday.select("Date", "Location1", "naturalCount", "positiveCount", "negativeCount")
#     all_tweets_byday.printSchema()
#     all_tweets_byday = all_tweets_byday.join(groupeby_loc_df, all_tweets_byday.Location1 == groupeby_loc_df.Location)
#     all_tweets_byday = all_tweets_byday.select("Date", "Location", "naturalCount", "positiveCount", "negativeCount")
#     all_tweets_byday.show()
#     all_tweets_byday.coalesce(1).write.mode('overwrite').csv('tweets_Counts_by_top_3_location', header='true')  ##action



if __name__ == '__main__':
    main()
