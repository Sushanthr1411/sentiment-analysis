<img width="972" height="570" alt="image" src="https://github.com/user-attachments/assets/ffd80080-4638-422f-acb5-2b8160dc8749" />

# Sentiment Analysis

## About Project

This project is a machine learning-based sentiment analysis tool designed to classify text data into three sentiment categories: positive, negative, and neutral. It leverages deep learning models and natural language processing techniques to analyze user input and predict sentiment. The project includes scripts for training the model, making predictions, and a simple application interface for user interaction. It is suitable for applications such as social media monitoring, customer feedback analysis, and general text sentiment classification.


## Working of the Project

1. **Data Preprocessing:**
	- The raw text data is cleaned by removing URLs, mentions, punctuation, and converting to lowercase.
	- Synonym replacement is used for data augmentation to improve model generalization.

2. **Label Encoding:**
	- Sentiment labels (negative, neutral, positive) are mapped and encoded for model training.

3. **Tokenization & Padding:**
	- Texts are tokenized and converted to sequences, then padded to a fixed length for input to the neural network.

4. **Model Training:**
	- A Bidirectional LSTM neural network is trained on the processed and augmented data to classify sentiment.
	- Early stopping is used to prevent overfitting.

5. **Saving Artifacts:**
	- The trained model, tokenizer, and label encoder are saved for later use in prediction.

6. **Prediction:**
	- For new input text, the same preprocessing and tokenization steps are applied.
	- The model predicts sentiment, and a neutrality threshold is used to handle uncertain cases.

7. **User Interface:**
	- The Streamlit app provides a simple UI for users to input text and view sentiment predictions with emojis.

---
## Tech Stack Used

**Programming Language:**
- Python 3.11

**Libraries & Frameworks:**
- TensorFlow / Keras
- scikit-learn
- NLTK
- NumPy
- Pandas
- Streamlit
- Pickle (standard library)
- re (standard library)
- random (standard library)

**Artifacts & Files:**
- `.h5` files: Trained Keras/TensorFlow models
- `.pkl` files: Saved tokenizer and label encoder
- `Sentiment_dataset.csv`: Dataset for training/testing

---
Feel free to add setup instructions, usage examples, or more details as needed.
