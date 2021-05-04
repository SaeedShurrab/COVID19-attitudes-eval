import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import most_freq_topic_words
import warnings
warnings.filterwarnings("ignore")

preprocessed_dir = os.path.join(os.curdir,'data','preprocessed')
tables_dir = os.path.join(os.curdir,'tables')

feb_tweets = pd.read_pickle(os.path.join(preprocessed_dir,'feb_tweets.pkl'))
mar_tweets = pd.read_pickle(os.path.join(preprocessed_dir,'mar_tweets.pkl'))
apr_tweets = pd.read_pickle(os.path.join(preprocessed_dir,'apr_tweets.pkl'))



# Data splitting for LDA purposes.
feb_pos = feb_tweets[feb_tweets.label == 'positive']
feb_neg = feb_tweets[feb_tweets.label == 'nigative']
mar_pos = mar_tweets[mar_tweets.label == 'positive']
mar_neg = mar_tweets[mar_tweets.label == 'nigative']
apr_pos = apr_tweets[apr_tweets.label == 'positive']
apr_neg = apr_tweets[apr_tweets.label == 'nigative']



# TF-IDF vectorizer instanciation
vectorizer_fp = TfidfVectorizer(analyzer='word',       
                             min_df = 10,
                             max_df = 0.9,                        
                             stop_words='english',             
                             token_pattern='[a-zA-Z0-9]{3,}',           
                            )

vectorizer_fn = TfidfVectorizer(analyzer='word',       
                             min_df = 10,
                             max_df = 0.9,                        
                             stop_words='english',             
                             token_pattern='[a-zA-Z0-9]{3,}',           
                            )

vectorizer_mp = TfidfVectorizer(analyzer='word',       
                             min_df = 10,
                             max_df = 0.9,                        
                             stop_words='english',             
                             token_pattern='[a-zA-Z0-9]{3,}',           
                            )

vectorizer_mn = TfidfVectorizer(analyzer='word',       
                             min_df = 10,
                             max_df = 0.9,                        
                             stop_words='english',             
                             token_pattern='[a-zA-Z0-9]{3,}',           
                            )

vectorizer_ap = TfidfVectorizer(analyzer='word',       
                             min_df = 10,
                             max_df = 0.9,                        
                             stop_words='english',             
                             token_pattern='[a-zA-Z0-9]{3,}',           
                            )

vectorizer_an = TfidfVectorizer(analyzer='word',       
                             min_df = 10,
                             max_df = 0.9,                        
                             stop_words='english',             
                             token_pattern='[a-zA-Z0-9]{3,}',           
                            )



# TF-IDF vectrorizer fitting
feb_pos_vec = vectorizer_fp.fit_transform(list(feb_pos['clean_stemmed']))
feb_neg_vec = vectorizer_fn.fit_transform(list(feb_neg['clean_stemmed']))
mar_pos_vec = vectorizer_mp.fit_transform(list(mar_pos['clean_stemmed']))
mar_neg_vec = vectorizer_mn.fit_transform(list(mar_neg['clean_stemmed']))
apr_pos_vec = vectorizer_ap.fit_transform(list(apr_pos['clean_stemmed']))
apr_neg_vec = vectorizer_an.fit_transform(list(apr_neg['clean_stemmed']))




# LDA model instanciation
lda_model_fp = LatentDirichletAllocation(n_components=4,            
                                      max_iter=20,               
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=128,            
                                      evaluate_every = -1,       
                                      n_jobs = -1,               
                                     )

lda_model_fn = LatentDirichletAllocation(n_components=4,            
                                      max_iter=20,               
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=128,            
                                      evaluate_every = -1,       
                                      n_jobs = -1,               
                                     )

lda_model_mp = LatentDirichletAllocation(n_components=4,            
                                      max_iter=20,               
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=128,            
                                      evaluate_every = -1,       
                                      n_jobs = -1,               
                                     )

lda_model_mn = LatentDirichletAllocation(n_components=4,            
                                      max_iter=20,               
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=128,            
                                      evaluate_every = -1,       
                                      n_jobs = -1,               
                                     )

lda_model_ap = LatentDirichletAllocation(n_components=4,            
                                      max_iter=20,               
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=128,            
                                      evaluate_every = -1,       
                                      n_jobs = -1,               
                                     )

lda_model_an = LatentDirichletAllocation(n_components=4,            
                                      max_iter=20,               
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=128,            
                                      evaluate_every = -1,       
                                      n_jobs = -1,               
                                     )




# LDA model fitting
feb_pos_tm = lda_model_fp.fit_transform(feb_pos_vec)
print('February positive topic modelling completed')
feb_neg_tm = lda_model_fn.fit_transform(feb_neg_vec)
print('February negative topic modelling completed')
mar_pos_tm = lda_model_mp.fit_transform(mar_pos_vec)
print('March positive topic modelling completed')
mar_neg_tm = lda_model_mn.fit_transform(mar_neg_vec)
print('march negative topic modelling completed')
apr_pos_tm = lda_model_ap.fit_transform(apr_pos_vec)
print('April positive topic modelling completed')
apr_neg_tm = lda_model_an.fit_transform(apr_neg_vec)
print('April negative topic modelling completed')



fp = most_freq_topic_words(lda_model_fp,vectorizer_fp,10)
fp.to_csv(os.path.join(tables_dir,'most-frequent-positive-words-February'))

fn = most_freq_topic_words(lda_model_fn,vectorizer_fn,10)
fn.to_csv(os.path.join(tables_dir,'most-frequent-negative-words-February'))

mp = most_freq_topic_words(lda_model_mp,vectorizer_mp,10)
mp.to_csv(os.path.join(tables_dir,'most-frequent-positive-words-March'))

mn = most_freq_topic_words(lda_model_mn,vectorizer_mn,10)
mn.to_csv(os.path.join(tables_dir,'most-frequent-negative-words-March'))

ap = most_freq_topic_words(lda_model_ap,vectorizer_ap,10)
ap.to_csv(os.path.join(tables_dir,'most-frequent-positive-words-April'))

an = most_freq_topic_words(lda_model_an,vectorizer_an,10)
an.to_csv(os.path.join(tables_dir,'most-frequent-negative-words-April'))