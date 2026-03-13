import streamlit as st
import pickle
import numpy as np
from gensim.models import Word2Vec


# Load model
@st.cache_resource
def load_model():

    with open("/kaggle/working/XGBoost_model.pkl", "rb") as f:
        model = pickle.load(f)

    w2v_model = Word2Vec.load("/kaggle/working/word2vec.model")

    return model, w2v_model


model, w2v_model = load_model()


# UI
st.title("Question Similarity Checker")
st.write("Enter two questions to calculate similarity")

q1 = st.text_input("Question 1")
q2 = st.text_input("Question 2")


if st.button("Check Similarity"):

    if q1.strip() and q2.strip():

        # Example dummy feature creation
        features = np.random.rand(1,100)

        score = model.predict(features)[0]

        st.success(f"Similarity Score: {score}")

    else:
        st.warning("Please enter both questions")
