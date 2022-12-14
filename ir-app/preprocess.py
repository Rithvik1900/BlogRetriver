import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
import regex as re
from gensim.parsing.preprocessing import STOPWORDS
import streamlit as st
import nltk

nltk.download('punkt')

@st.cache
def remove_specials(documents):
    cleaned = list()
    for text in documents:
        word = re.sub("[0-9a-zA-Z]"," ",text)
        cleaned.append(word)
    return cleaned

@st.cache
def remove_stopwords(documents):
    cleaned = list()
    for text in documents:
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in STOPWORDS]
        str_t = " ".join(tokens_without_sw)
        cleaned.append(str_t)
    return cleaned

@st.cache
def preprocess(documents):
    cleaned = remove_specials(documents)
    cleaned = remove_stopwords(documents)
    return cleaned
