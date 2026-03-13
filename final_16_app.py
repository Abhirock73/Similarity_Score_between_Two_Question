

import streamlit as st
import pickle
import numpy as np
import re
import distance
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

stemmer = PorterStemmer() 
from scipy.spatial.distance import cosine, cityblock, euclidean, jaccard

# -----------------------------
# Load models
# -----------------------------

def load_model():

    # load XGBoost model
    with open("XGBoost_model.pkl", "rb") as f:
        model = pickle.load(f)

    # load Word2Vec model
    w2v_model = Word2Vec.load("word2vec.model")

    return model, w2v_model


# call the function
model, w2v_model = load_model()


def sentence_to_vector(sentence, model, vector_size=100):

    words = sentence.split()
    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)


def create_223_features(q1, q2, w2v_model):

    features = []

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    q1_words = set(q1_tokens)
    q2_words = set(q2_tokens)

    common_words = q1_words & q2_words
    common_chars = set(q1) & set(q2)

    # -------------------------
    # Token overlap features
    # -------------------------

    features.append(len(common_words)/(min(len(q1_words),len(q2_words))+0.0001))
    features.append(len(common_words)/(max(len(q1_words),len(q2_words))+0.0001))

    features.append(len(common_chars)/(min(len(q1),len(q2))+0.0001))
    features.append(len(common_chars)/(max(len(q1),len(q2))+0.0001))

    features.append(len(q1_tokens)/(len(q2_tokens)+0.0001))
    features.append(len(q2_tokens)/(len(q1_tokens)+0.0001))

    features.append(1 if q1_tokens[-1:] == q2_tokens[-1:] else 0)
    features.append(1 if q1_tokens[:1] == q2_tokens[:1] else 0)

    features.append(abs(len(q1_tokens)-len(q2_tokens)))
    features.append((len(q1_tokens)+len(q2_tokens))/2)

    features.append(len(q1))
    features.append(len(q2))

    features.append(len(common_words))
    features.append(len(common_chars))

    # -------------------------
    # Fuzzy Features
    # -------------------------

    features.append(fuzz.QRatio(q1,q2))
    features.append(fuzz.partial_ratio(q1,q2))
    features.append(fuzz.token_sort_ratio(q1,q2))
    features.append(fuzz.token_set_ratio(q1,q2))

    # -------------------------
    # NEW FEATURES (missing ones)
    # -------------------------

    # Jaccard similarity
    features.append(len(common_words)/(len(q1_words|q2_words)+0.0001))

    # Longest common substring ratio
    substr = list(distance.lcsubstrings(q1, q2))
    if len(substr) > 0:
        features.append(len(substr[0])/(min(len(q1),len(q2))+0.0001))
    else:
        features.append(0)

    # -------------------------
    # Word2Vec vectors
    # -------------------------

    v1 = sentence_to_vector(q1, w2v_model, 100)
    v2 = sentence_to_vector(q2, w2v_model, 100)

    features.extend(v1)
    features.extend(v2)

    # -------------------------
    # Distance metrics
    # -------------------------

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
      features.append(1)
      features.append(0)
      features.append(0)
    else:
      features.append(cosine(v1,v2))
      features.append(cityblock(v1,v2))
      features.append(euclidean(v1,v2))

    features = np.array(features)

    return features.reshape(1,-1)

#preprocess
def preprocess(q):

    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q, "html.parser")
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    # ADD STEMMING HERE
    stemmed_words = []
    for word in q.split():
        stemmed_words.append(stemmer.stem(word))

    q = " ".join(stemmed_words)


    return q



   

# UI
st.title("Question Similarity Checker")
st.write("Enter two questions to calculate similarity")

q1 = st.text_input("Question 1")
q2 = st.text_input("Question 2")
q1_process = preprocess(q1)
q2_process = preprocess(q2)



if st.button("Check Similarity"):

    if q1.strip() and q2.strip():

        # Example dummy feature creation
        features = create_223_features(q1_process,q2_process,w2v_model) 

        score = model.predict_proba(features)[0][1]

        st.success(f"Similarity Score: {score}")

    else:
        st.warning("Please enter both questions")
