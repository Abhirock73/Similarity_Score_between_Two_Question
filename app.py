import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    model = joblib.load("/kaggle/working/ids_rf_model.pkl")
    return model

model = load_model()

st.title("Question Similarity Checker")

q1 = st.text_input("Question 1")
q2 = st.text_input("Question 2")

if st.button("Check Similarity"):
    if q1 and q2:
        st.write("Prediction:", model.predict([[0]]))
    else:
        st.warning("Enter both questions")
import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    
    with open("/kaggle/working/model.pkl", "rb") as f:
        model = joblib.load(f)
        
    with open("/kaggle/working/vectorizer.pkl", "rb") as f:
        vectorizer = joblib.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# UI
st.title("Question Similarity Checker")

st.write("Enter two questions to calculate similarity")

q1 = st.text_input("Question 1")
q2 = st.text_input("Question 2")

if st.button("Check Similarity"):

    if q1.strip() and q2.strip():

        # Convert text to vectors
        v1 = vectorizer.transform([q1])
        v2 = vectorizer.transform([q2])

        # Combine features
        features = np.hstack((v1.toarray(), v2.toarray()))

        # Prediction
        score = model.predict(features)[0]

        st.success(f"Similarity Score: {score}")

    else:
        st.warning("Please enter both questions")
