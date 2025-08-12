# train_sentiment.py
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import random
import nltk
from nltk.corpus import wordnet

# Download WordNet data (run once)
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Load dataset
df = pd.read_csv("Sentiment_dataset.csv", encoding='latin-1')
df = df[['target', 'text']]

# 2. Ensure all targets are strings, then map sentiment values
df['target'] = df['target'].astype(str)
df['target'] = df['target'].replace({'0': 'negative', '2': 'neutral', '4': 'positive'})

# 3. Basic text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r"@\w+", '', text)  # remove mentions
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # remove punctuation/numbers
    text = re.sub(r"\s+", ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# 4. Synonym replacement augmentation
def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == random_word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:  # only replace n words
            break
    return ' '.join(new_words)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

# Augment the dataset by creating augmented sentences for training set only
augmented_texts = []
augmented_labels = []

for idx, row in df.iterrows():
    text = row['text']
    label = row['target']
    # Add original
    augmented_texts.append(text)
    augmented_labels.append(label)
    # Add augmented version
    augmented_text = synonym_replacement(text, n=2)
    augmented_texts.append(augmented_text)
    augmented_labels.append(label)

df_aug = pd.DataFrame({'text': augmented_texts, 'target': augmented_labels})

# 5. Encode labels on augmented dataset
label_encoder = LabelEncoder()
df_aug['label'] = label_encoder.fit_transform(df_aug['target'])
y = to_categorical(df_aug['label'], num_classes=3)

# 6. Tokenize text on augmented dataset
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_aug['text'])
sequences = tokenizer.texts_to_sequences(df_aug['text'])
padded_sequences = pad_sequences(sequences, maxlen=75, padding='post')

# 7. Train-test split on augmented data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, y, test_size=0.2, random_state=42
)

# 8. Build model with dropout (same as before)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=75),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 9. Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 10. Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    batch_size=64,
    callbacks=[early_stop]
)

# 11. Save model & tokenizer
model.save("sentiment_model_3class_augmented.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
