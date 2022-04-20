from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
from scipy.sparse import hstack
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from scipy.sparse import hstack
import re
from bs4 import BeautifulSoup
import string 
from fastapi import FastAPI
import gradio as gr
app=FastAPI()


tfidf_X1=joblib.load('tfidf_X1')
tfidf_X2=joblib.load('tfidf_X2')
reg=joblib.load('svc')
binarizer=joblib.load('binarizer')

token=ToktokTokenizer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = BeautifulSoup(text, 'html.parser')
    text = text.get_text()
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)    
    text = text.strip(' ')
    punct = set(string.punctuation) 
    text = "".join([ch for ch in text if ch not in punct])
    stop_words = set(stopwords.words('english'))
    
    words=token.tokenize(text)
    
    text=[word for word in words if not word in stop_words]
    text=' '.join(map(str, text))
    text=token.tokenize(text)
    lemm_list=[]
    for word in text:
        x=lemmatizer.lemmatize(word, pos='v')
        lemm_list.append(x)
    text=' '.join(map(str, lemm_list))
    return text

def tags_normalization(text):
    text=text.replace('<','').replace('>', ' ')
    return text



class Item(BaseModel):
    content : str
    title : str

TITRE = "How to enumerate an enum"
CONTENU = "How can you enumerate an enum in C#? E.g. the following code does not compile:"


@app.get("/")
def welcome():
    return "welcome to tag predictionAPI"



@app.post("/")
def tag_predict(item:Item):
    unseen_data={'Title': preprocess(item.title), 'Body': preprocess(item.content)}
    unseen_data=pd.DataFrame(data=unseen_data, index=[0])
    tfidf_X=tfidf_X1
    tfidf_Y=tfidf_X2
    tfidf_Y=tfidf_Y.transform(unseen_data.Title)
    tfidf_X=tfidf_X.transform(unseen_data.Body)
    tfidf_unseen=hstack([tfidf_X, tfidf_Y])
    y_pred=reg.predict(tfidf_unseen)
    pred_list=binarizer.inverse_transform(y_pred)
    return list(sum(pred_list, ()))
  
