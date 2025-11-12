# app.py
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Clean input text
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', str(tweet))
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    tweet = tweet.lower().split()
    lm = WordNetLemmatizer()
    tweet = [lm.lemmatize(word) for word in tweet if word not in stopwords.words('english')]
    return ' '.join(tweet)

# Streamlit UI
st.title("💬 Twitter Sentiment Analysis App")
st.write("Analyze the sentiment of any tweet as Positive or Negative.")

user_tweet = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    cleaned = clean_tweet(user_tweet)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    st.subheader(f"Predicted Sentiment: 🎯 **{prediction.capitalize()}**")
