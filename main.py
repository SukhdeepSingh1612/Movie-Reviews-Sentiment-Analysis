# Step 1 : Import the necessary libraries and load model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Step 2 : Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value : key for (key, value) in word_index.items()}


# Load the pre-trained model with Relu Activation
model = load_model('simplernn_model.h5')

# Step 3 : Helper functions
# Functions to decode the reviews

def decode_review(encoded_review):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    return decoded_review

# Function to preprocess the user input

def preprocess_input(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Step 4:  Prediction Function
def predict_sentiment(review):
    # Preprocess the review
    preprocessed_review = preprocess_input(review)

    # Make prediction
    prediction = model.predict(preprocessed_review)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    return sentiment, prediction[0][0]


import streamlit as st
## Streamlit app
st.title("IMDB Movie Reviews Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative):")


## User input
user_input = st.text_area("Review", "Type your review here...")

if st.button("Classify"):

    preprocess_input = preprocess_input(user_input)

    prediction = model.predict(preprocess_input)

    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
   
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
    st.write("Please enter a movie review.")

   