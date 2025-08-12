import tensorflow as tf
import pickle
import re
import numpy as np

# 1. Load the saved model and preprocessing objects
model = tf.keras.models.load_model('sentiment_model_3class.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 2. Text cleaning function (consistent with training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"@\w+", '', text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # Remove punctuation/numbers
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# 3. Predict function with neutrality threshold
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

# 4. Input and clean output with emoji
if __name__ == "__main__":
    user_input = input("Enter a review text: ")
    sentiment = predict_sentiment(user_input)

    emoji_map = {
        "positive": "😊",
        "negative": "😠",
        "neutral": "😑"
    }

    print(f"Predicted Sentiment: {sentiment.capitalize()} {emoji_map.get(sentiment, '')}")
