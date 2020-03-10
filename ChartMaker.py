import pandas as pd
import glob
import os
import numpy as np
from PIL import Image
from wordcloud import WordCloud
from scipy.misc import imread
import matplotlib.pyplot as plt

from datetime import datetime
def transform_format(val):

    if val == 0:
        return 255
    else:
        return val
def create_PieChart(pos,neg,neu,locaiton):
    colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0)  # explode 1st slice
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [pos, neg, neu]

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.savefig("img/TweetPieChart_for"+locaiton+".png")


path ="tweets_by_top_3_location/"  # use your path
all_files = glob.glob(path + "/*.csv")
li = []

file_name = (os.path.basename(all_files[0]))
count =0
data = pd.read_csv(path+file_name)
data_as_PD = pd.DataFrame(data)

#twitter_mask = np.array(Image.open("img/twitter_mask.png"))

twitter_mask = imread('img/twitter_mask.png', flatten=True)

transformed_twitter_mask = np.ndarray((twitter_mask.shape[0],twitter_mask.shape[1]), np.int32)
for i in range(len(twitter_mask)):
    # print(twitter_mask[i])
    transformed_twitter_mask[i] = list(map(transform_format, twitter_mask[i]))
wc = WordCloud(background_color="white", max_words=1000, mask=twitter_mask,
             contour_width=3, contour_color='firebrick')
## Word Cloud for 3 top tweet Location ##
# print(data_as_PD.iloc[0,0])
# print(data_as_PD.iloc[0, 1])
# print(data_as_PD.iloc[1,0])
# print(data_as_PD.iloc[1, 1])
# print(data_as_PD.iloc[2,0])
# print(data_as_PD.iloc[2, 1])
#

wordcloud = wc.generate(data_as_PD.iloc[0,1])
wordcloud.to_file("img/1st"+ data_as_PD.iloc[0,0]+"_none_proccesses.png")
wordcloud =wc.generate(data_as_PD.iloc[1,1])
wordcloud.to_file("img/2nd"+ data_as_PD.iloc[1,0]+"_none_proccesses.png")
wordcloud = wc.generate(data_as_PD.iloc[2,1])
wordcloud.to_file("img/3rd"+ data_as_PD.iloc[2,0]+"_none_proccesses.png")


## WoldWide Word Cloud ##
# path ="output.txt/"  # use your path
# all_files = glob.glob(path + "/*.txt")
# file_name = (os.path.basename(all_files[0]))
# file_content=open (path+file_name).read()
# wordcloud = wc.generate(file_content)
# wordcloud.to_file("img/WorldWideWordCloud_none_proccesses.png")
## Pie Chart for Top 3 Locations
# path ="tweets_Counts_by_top_3_location/"  # use your path
# all_files = glob.glob(path + "/*.csv")
# li = []
# file_name = (os.path.basename(all_files[0]))
# data = pd.read_csv(path+file_name)
# data_as_PD = pd.DataFrame(data)
#
# data_as_PD = data_as_PD.groupby(['Location'],as_index=False).sum()
# for index, row in data_as_PD.iterrows():
#     create_PieChart(row['positiveCount'],row['negativeCount'],row['naturalCount'],row['Location'])
#     plt.clf()
#
#
# path ="alltweetcount/"  # use your path
# all_files = glob.glob(path + "/*.csv")
# file_name = (os.path.basename(all_files[0]))
# count =0
# data = pd.read_csv(path+file_name)
# data_as_PD = pd.DataFrame(data)
# colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
# explode = (0.1, 0, 0)  # explode 1st slice
# labels = 'Positive', 'Negative', 'Neutral'
# sizes = [data_as_PD['positiveCount'].sum(),data_as_PD['negativeCount'].sum(),data_as_PD['naturalCount'].sum()]
# print(data_as_PD['positiveCount'].sum())
# plot = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
# autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.savefig("img/TweetPieChart.png")
# plt.clf()



#
#
#
#
#
#
# data_as_PD["Date"] = pd.to_datetime(data_as_PD["Date"])
# data_as_PD = data_as_PD.sort_values(by="Date")
# n_groups = data_as_PD.count()
# print(n_groups[0])
# fig, ax = plt.subplots()
# index = np.arange(n_groups[0])
# bar_width = 0.2
# opacity = 0.8
# rects1 = plt.bar(index, data_as_PD['positiveCount'], bar_width,
# alpha=opacity,
# color='yellowgreen',
# label='Positive')
# rects2 = plt.bar(index + bar_width, data_as_PD['negativeCount'], bar_width,
# alpha=opacity,
# color='lightcoral',
# label='Negative')
# rects3 = plt.bar(index + bar_width + bar_width, data_as_PD['naturalCount'], bar_width,
# alpha=opacity,
# color='lightskyblue',
# label='Natural')
#
#
# plt.xlabel('Date')
# plt.ylabel('Tweet Count')
# plt.title('Day by Day Tweet Count from'+ str(data_as_PD["Date"][0]) )
# plt.xticks(index + bar_width ,index+1)
# plt.legend()
#
# plt.tight_layout()
#
# plt.savefig("img/Day_by_day_BarChart.png")
