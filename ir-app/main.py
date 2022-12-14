import os
import numpy as np
import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi
from preprocess import preprocess

st.set_page_config(page_title="Rithvik IR APP", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 24px;
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache
def process_documents():
    path = "ir-app/data/data.csv"
    df = pd.read_csv(path)
    titles = df["title"].values.tolist()
    documents = df["content"].values.tolist()
    urls = df["url"].values.tolist()
    documents_cleaned = preprocess(documents)
    bm25_index = prepare_indices(documents_cleaned)
    return documents,titles,urls,bm25_index

def prepare_indices(documents):
    
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index

def recommend(text,top_k,threshold):
    text = preprocess([text])[0]
    tokenized_query = text.split(" ")
    scores = bm25_index.get_scores(tokenized_query)

    recs = np.argsort(scores)[::-1]
    recs = [rec for rec in recs if scores[rec]>threshold]
    scores = [scores[rec] for rec in recs]
    sel_titles = [titles[i] for i in recs]
    sel_documents = [documents[i] for i in recs]
    sel_documents = [document[:500] for document in sel_documents]
    sel_urls = [urls[i] for i in recs]
    
    for i in range(min(top_k,len(sel_urls))): 
        st.metric(label=sel_urls[i],value=sel_titles[i], delta=str(round(scores[i],4)),help=sel_documents[i]+'....')


documents,titles,urls,bm25_index = process_documents()

query_text = st.sidebar.text_input('Enter the Search',value="data",key="mit")
st.session_state["query"] = query_text

top_k = st.sidebar.number_input(
        "View Top-k",
        min_value = 1,
        max_value=50,
        value= st.session_state.top_k if st.session_state.get("top_k",None) else 3,
        key = "topk"
    )
#st.session_state["top_k"] = top_k

threshold = st.sidebar.number_input(
        "Minimum Score",
        min_value = 0.0,
        value= st.session_state.thres if st.session_state.get("thres",None) else 0.01,
        key = "threshold"
    )
#st.session_state["thres"] = threshold

recommend(query_text,top_k,threshold)
