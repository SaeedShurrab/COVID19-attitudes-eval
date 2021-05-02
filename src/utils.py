
import re
from nltk.stem import PorterStemmer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer





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


# font specifications
font = {'family': 'Times New Roman',
        'color':  'k',
        'weight': 'normal',
        'size':8
        }