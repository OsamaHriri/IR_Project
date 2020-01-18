from datetime import date, timedelta
from pyspark.sql.types import StructType, IntegerType, StringType, StructField, DateType,TimestampType
from pyspark import SparkContext, SparkConf
from pyspark import RDD
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
import pandas as pd
schemaTweets = StructType([
    StructField("Date", DateType(), False),
    StructField("Time", TimestampType(), False),
    StructField("ID1", TimestampType(), False),
    StructField("ID2", StringType(), True),
    StructField("Name", StringType(), True),
    StructField("location", StringType(), True),
    StructField("TweetBody", StringType(),False)
     ])
import re
import sys
import utils
from nltk.stem.porter import PorterStemmer


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


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

            word = str(PorterStemmer().stem(word))
            processed_tweet.append(word)

    return ' '.join(processed_tweet)


def main():
    porter_stemmer = PorterStemmer()
    spark = get_spark_session()

    data_frame = spark.read.format("csv").load('OsamaNaridGOT_7_C_before_SA-543000.csv')
    data_frame.show()
    data_frame.printSchema()
    tweet_process_udf = udf(preprocess_tweet,StringType())
    data_frame = data_frame.withColumn('after',tweet_process_udf('_c6'))
    data_frame.show()
    data_frame.printSchema()




def get_spark_session():
    return SparkSession \
        .builder.master("local") \
        .appName('SparkMapReduceExample') \
        .getOrCreate()


if __name__ == '__main__':
    main()
