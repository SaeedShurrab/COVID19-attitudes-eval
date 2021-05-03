
import re
import os
import matplotlib.pyplot as plt
from langdetect import detect
from nltk.stem import PorterStemmer
from langdetect.lang_detect_exception import LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# font specifications
font = {'family': 'Times New Roman',
        'color':  'k',
        'weight': 'normal',
        'size':8
        }



# hashtags and mentions extraction and cleaning function
def parse_hashtags_mentions(text):
    hashtags = re.findall('#\s?\w+', text)
    hashtags = [hashtag[1:] for hashtag in hashtags]
    mentions = re.findall("@\s?\w+", text)
    mentions = [mention[1:] for mention in mentions]
    text = clean_text = re.sub('@\s?\w+','',text)
    text = clean_text = re.sub('#\s?\w+','',text)
    words = text.split(" ")
    clean_text = " ".join([word for word in words if len(word) > 0 and not word.startswith("#") and not word.startswith("@") ])
    clean_text = "".join(re.findall("[\w\s]", clean_text))
    clean_text = re.sub("\s{2,}", " ", clean_text)
    return clean_text, hashtags, mentions



# Text cleaning function
def clean_text(text:str) -> str:
    text, _, _  = parse_hashtags_mentions(text)
    image_clean = re.sub("pic.[\w/.]+", "", text)
    punctionation_clean = re.sub("""[!?.'"-]""", "", image_clean)
    numbers_clean = re.sub("\d", "", punctionation_clean)
    skip_clean = re.sub("\\n|\\t", " ", numbers_clean)
    link_clean = re.sub("https.+", "", skip_clean)
    underscore_clean = re.sub('_','',link_clean)
    return underscore_clean



# Stop words cleaning function
def clean_stopwords(text:str) -> str:
    text =text.lower()
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                  'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 
                  'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                  'hers', 'herself', 'it', "it's", 'its','itself','they',
                  'them', 'their', 'theirs', 'themselves', 'what', 'which',
                  'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                  'am', 'is', 'are', 'was', 'were', 'be','been','being','have',
                  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a','an',
                  'the', 'and', 'but', 'if', 'or', 'because', 'as','until','while',
                  'of', 'at', 'by','for', 'with', 'about','between','into','through',
                  'during', 'before', 'after', 'above','below','to','from','up','down',
                  'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',  'then',
                  'once', 'here', 'there', 'when', 'where', 'why','how','all','any','both',
                  'each', 'other', 'such', 'only', 'own', 'same','s','t','can','will','just', 
                  'should', "should've", 'now', 'd',  'll',  'm', 'o', 're', 've', 'y', 'than',]
    tokens = text.split(" ")
    clean = [token for token in tokens if token not in stop_words]
    return " ".join(clean).lower()



# Tweets stemming function
def stem_tweet(text:str) -> str:
    stemmer = PorterStemmer()
    tokens = text.split(" ")
    stemmed = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed)




# All cleaning operations function
def process_all_text(text:str)->str:
    clean = clean_stopwords(text)
    return clean_text(clean)




# Polarity calculation function
def get_polarity(text:str) -> float:
    analyzer = SentimentIntensityAnalyzer()
    results = analyzer.polarity_scores(text)
    return results.get("compound")



# Labels assinment function
def get_label(x:float) -> str:
    if   x > 0.05:
        return 'positive'
    elif x < -.05:
        return 'nigative'
    else:
        return 'neutral'



#  Language detection function
def detect_lang(text:str) -> str:
    lang = text
    try:
        lang = detect(text)
    except LangDetectException:
        return 'None'
    return lang



# Tweet length extraction function
def twt_len(text: str) -> int:
    return len(text.split())

#sentiment plotting function
def plot_sentiment(days, cases, death, recovered, positive, negative, nuetral, path):

    fig = plt.figure(figsize=(5,7))
    sub1 = fig.add_subplot(2,1,1)
    sub1.plot(days,cases,'r-*',
             days,death,'k-*',
             days,recovered,'g-*')

    sub1.set_xticklabels(())
    sub1.set_title("Number of COVID-19 Cases",loc = 'center',fontdict=font)
    sub1.set_ylabel('Number of Cases',fontdict=font)
    sub1.legend(['Confirmed','Death','Recovered'],loc ='upper left',prop={'size': 7})
    sub1.grid()

    sub2 = fig.add_subplot(2,1,2)
    sub2.plot(days,positive,'g-*',
              days,negative,'r-*',
              days,nuetral,'k-*'
              )  
    sub2.set_xticklabels(days, rotation=60)
    sub2.set_title("Sentiment Labels Frequency",loc = 'center',fontdict=font)  
    sub2.set_ylabel('Number of Tweets',fontdict=font)
    sub2.set_xlabel('Days',fontdict=font)  
    sub2.legend(['Positive','Negative','Neutral'],loc ='upper left',prop={'size': 7})
    sub2.grid()

    plt.tight_layout()
    plt.savefig(path,dpi =600)
    plt.show()

# most frequent topic words extraction function
def most_freq_topic_words(lda_model : sklearn.decomposition._lda.LatentDirichletAllocation,
                          vectorizer : sklearn.feature_extraction.text.TfidfVectorizer,
                          n_words : int) -> pd.DataFrame:
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []

    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))

    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    return df_topic_keywords.T