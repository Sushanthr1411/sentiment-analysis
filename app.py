import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np

# Load model and preprocessing objects once
@st.cache(allow_output_mutation=True)
def load_model_tokenizer():
    model = tf.keras.models.load_model('sentiment_model_3class.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_tokenizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"@\w+", '', text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # Remove punctuation/numbers
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def predict_sentiment(text, neutrality_threshold=0.15):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=75, padding='post')

    pred = model.predict(padded)[0]  # shape (3,)

    class_labels = label_encoder.classes_  # E.g. ['negative', 'neutral', 'positive']
    sorted_indices = np.argsort(pred)[::-1]
    top_idx = sorted_indices[0]
    second_idx = sorted_indices[1]

    top_prob = pred[top_idx]
    second_prob = pred[second_idx]
    prob_diff = top_prob - second_prob

    if prob_diff < neutrality_threshold:
        predicted_label = "neutral"
    else:
        predicted_label = class_labels[top_idx]

    return predicted_label

emoji_map = {
    "positive": "😊",
    "negative": "😠",
    "neutral": "😑"
}

st.title("Sentiment Analysis")

user_input = st.text_area("Enter your review text here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.markdown(f"### Predicted Sentiment: {sentiment.capitalize()} {emoji_map.get(sentiment, '')}")
    else:
        st.warning("Please enter some text to analyze.")
