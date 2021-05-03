import os
import nltk
import datetime
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils import *



nltk.download('vader_lexicon')
tqdm.pandas()

raw_dir = os.path.join(os.curdir,'data','raw')
intermediate_dir = os.path.join(os.curdir,'data','intermediate')
preprocessed_dir = os.path.join(os.curdir,'data','preprocessed')
plots_dir = os.path.join(os.curdir,'plots')

data = pd.read_csv(os.path.join(raw_dir,'US COVID-19 Tweets.csv'))

data = data[['text','datetime','hashtags']]

import numpy as np
idx = np.arange(data.shape[0])
np.random.shuffle(idx)
data = data.iloc[0:13000]

# Apply Cleaning
data['clean_text']=data['text'].progress_apply(lambda x: process_all_text(x))
print('cleaning completed \n')

# Apply stemming
data['clean_stemmed']=data['clean_text'].progress_apply(lambda x: stem_tweet(x))
print('stemming completed \n')

# Calculate polarity
data['polarity']=data['clean_stemmed'].progress_apply(lambda x: get_polarity(x))
print('polarity calculation completed \n')

# Extract Label
data['label']=data['polarity'].progress_apply(lambda x: get_label(x))
print('labelling completed \n')

# Tweet language detection
data['language']=data['clean_text'].progress_apply(lambda x: detect_lang(x))
print('language detection completed \n')

# Tweet Length Extraction
data['length']=data['clean_stemmed'].progress_apply(lambda x: twt_len(x))
print('tweets length calculation completed \n')


# Non-English tweets removal 
data = data[data.language == 'en']

# Short tweets removal
data = data[data.length > 3]


# Labels distribution 
label_dist={'positive':len(data.loc[data.label =='positive']),
               'negative':len(data.loc[data.label =='nigative']),
               'neutral':len(data.loc[data.label =='neutral'])
               }

plt.figure(figsize=(5,5))
plt.bar(label_dist.keys(),label_dist.values(),)
plt.xlabel('Labeles',fontdict=font)
plt.ylabel('Frequency',fontdict=font)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir,'label_dist.jpg'),dpi=300)


# Data sorting by date
data = data.sort_values(by='datetime')
data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True)

# Month extraction
data['month'] = pd.DatetimeIndex(data['datetime']).month

# Data splitting based on month
feb_tweets = data[data.month == 2]
mar_tweets = data[data.month == 3]
apr_tweets = data[data.month == 4]

data.to_csv(os.path.join(intermediate_dir,'cleaned_data.csv'))

feb_tweets.to_csv(os.path.join(preprocessed_dir,'feb_tweets.csv'))
mar_tweets.to_csv(os.path.join(preprocessed_dir,'mar_tweets.csv'))
apr_tweets.to_csv(os.path.join(preprocessed_dir,'apr_tweets.csv'))

print('preprocessing completed \n')