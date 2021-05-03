import os
import numpy as np
import pandas as pd
from src.utils import plot_sentiment

preprocessed_dir = os.path.join(os.curdir,'data','preprocessed')
plots_dir = os.path.join(os.curdir,'plots')

feb_tweets = pd.read_csv(os.path.join(preprocessed_dir,'feb_tweets.csv'),index_col=0)
mar_tweets = pd.read_csv(os.path.join(preprocessed_dir,'mar_tweets.csv'),index_col=0)
#apr_tweets = pd.read_csv(os.path.join(preprocessed_dir,'apr_tweets.csv'),index_col=0)

# COVID -19 Confirmed Cases 1/Feb. - 29/Apr.

feb_days = ['{}-feb'.format(x) for x in range(1,30)]
            
feb_cases = [1, 0, 3, 0, 0, 0, 0, 0, 0,0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 36,
             0, 6, 1, 2, 8]

feb_death = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1]

feb_rec = [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0,
           0, 1, 0]

mar_days = ['{}-Mar'.format(x) for x in range(1,32)]

mar_cases = [6, 24, 20, 31, 70, 48, 136, 116, 69, 374, 323, 382, 514, 548,
             807, 1125, 1776, 1344, 5967, 5526, 6326, 7680, 10582, 10063,
             11919, 17992, 18126, 19824, 19124, 21237, 26025]

mar_death = [0, 5, 1, 4, 1, 2, 3, 4, 1, 6, 8, 6, 8, 10, 14, 26, 34, 31, 94,
             91, 92, 145, 199, 225, 309, 406, 543, 475, 676, 776, 1171]

mar_rec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 5, 0, 88, 16, 26, 29,
           2, 0, 170, 13, 320, 188, 203, 1593, 2979, 1380]

apr_days = ['{}-Apr'.format(x) for x in range(1,30)]

apr_cases = [25430, 30406, 31790, 33229, 27775, 29515, 30804, 31533, 34673, 33519, 29930,
             28537, 25311, 27046, 29004, 31307, 32081, 32528, 26219, 25899, 27157, 28486,
             28819, 36188, 32796, 27631, 22412, 24385, 27327]

apr_death = [1144, 1427, 1322, 1610, 1505, 1519, 2297, 2079, 2018, 2069, 2009, 1720, 1784,
             2392, 2498, 2084, 2584, 2347, 1170, 1741, 2400, 2326, 2312, 1769, 2262, 1126,
             1338, 2136, 2612]

apr_rec = [1450, 527, 706, 4945, 2796, 2133, 2182, 1796, 1851, 3380, 2480, 1718, 10494,
           4281, 4333, 2607, 3842, 6295, 5497, 1992, 2875, 2162, 2837, 18876, 1293, 6616,
           4436, 4512, 4784]


feb_tweets['day'] = pd.DatetimeIndex(feb_tweets['datetime']).day
mar_tweets['day'] = pd.DatetimeIndex(mar_tweets['datetime']).day
#apr_tweets['day'] = pd.DatetimeIndex(apr_tweets['datetime']).day


# Sentimet frequency per month extraction
feb_freq = list(feb_tweets.groupby(['label','day'])['day'].count().to_dict().values())
mar_freq = list(mar_tweets.groupby(['label','day'])['day'].count().to_dict().values())
#apr_freq = list(apr_tweets.groupby(['label','day'])['day'].count().to_dict().values())

feb_nuet = feb_freq[0:29]
feb_neg = feb_freq[29:58]
feb_pos = feb_freq[58:]

mar_nuet = mar_freq[0:31]
mar_neg = mar_freq[31:62]
mar_pos = mar_freq[62:]

#apr_nuet = apr_freq[0:29]
#apr_neg = apr_freq[29:58]
#apr_pos = apr_freq[58:]


# Tweets sentiment associated with number of COVID-19 cases in February. 
plot_sentiment(feb_days,
               feb_cases, 
               feb_death,
               feb_rec,
               feb_pos,
               feb_neg,
               feb_nuet,
               os.path.join(plots_dir,'feb_sent.jpg'))


# Tweets sentiment associated with number of COVID-19 cases in March. 
plot_sentiment(mar_days,
               mar_cases, 
               mar_death,
               mar_rec,
               mar_pos,
               mar_neg,
               mar_nuet,
               os.path.join(plots_dir,'mar_sent.jpg'))



# Tweets sentiment associated with number of COVID-19 cases in April. 
#plot_sentiment(apr_days,
#               apr_cases, 
#               apr_death,
#               apr_rec,
#               apr_pos,
#               apr_neg,
#               apr_nuet,
#               os.path.join(plots_dir,'apr_sent.jpg'))

