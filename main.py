import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from keras._tf_keras.keras.datasets import imdb
from keras._tf_keras.keras.preprocessing import sequence
#from tensorflow.keras.models import load_model
from keras._tf_keras.keras.models import load_model
word_index=imdb.get_word_index()
reverse_word_index= {value: key for key, value in word_index.items()}
model=load_model('SimpleRnnIMDB.keras')
#helper function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

## function to preprocess the text
def preprocess_text(text):
    words =text.lower().split()
    encoded_review =[word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review
## prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment= "Positive" if prediction[0][0] >0.5 else "Negative"
    return sentiment,prediction[0][0]

#st.title("IMDB Movie Review Sentiment Analysis")
#st.write("Enter a movie review to classify it as positive and negative.")
#user_input=st.text_area("Movie Review")

#if st.button('Classify'):
    #preprocess_input=preprocess_text(user_input)

    #prediction=model.predict(preprocess_input)
    #sentiment= "Positive" if prediction[0][0] >0.5 else "Negative"

    #st.write(f'Sentiment: {sentiment}')
    #st.write(f'Prediction score: {prediction[0][0]}')
#else:
 #   st.write('Please enter a movie review')
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")

# Title and intro
st.markdown("<h1 style='text-align: center; color: #333;'>IMDB Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a movie review below to classify it as <b>Positive</b> or <b>Negative</b>.</p>", unsafe_allow_html=True)

# Text input
user_input = st.text_area("Movie Review", height=150, placeholder="Type your movie review here...")

# Classify button
if st.button('Classify', disabled=not user_input.strip()):
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.info('Please enter a movie review to classify.')
