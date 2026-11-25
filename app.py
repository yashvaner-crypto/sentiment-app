import streamlit as st
import pandas as pd
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None,
    )

model = load_model()

review = st.text_area("Enter a customer review")

if st.button("Classify"):
    results = model(review)[0]
    probs = pd.Series({r["label"]: r["score"] for r in results})
    st.header(probs.idxmax())
    st.bar_chart(probs)
