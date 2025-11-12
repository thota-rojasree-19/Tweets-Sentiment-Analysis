# model.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tqdm import tqdm  # for progress bar

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# ---------------- STEP 1: Load the Dataset ----------------
data = pd.read_csv('data/twitter_sentiment.csv', encoding='latin-1', header=None)
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'tweet']

print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)

# Convert numeric sentiment (0 = negative, 4 = positive)
data['sentiment'] = data['sentiment'].map({0: 'negative', 4: 'positive'})

# Drop any missing rows
data = data.dropna(subset=['tweet', 'sentiment'])

# (Optional) Use a smaller sample for faster training
data = data.sample(50000, random_state=42)
print("✅ Using 50,000 tweets for faster processing and model training!")

# ---------------- STEP 2: Clean the Tweets ----------------
stop_words = set(stopwords.words('english'))
lm = WordNetLemmatizer()

def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', str(tweet))      # remove URLs
    tweet = re.sub(r'@\w+', '', tweet)              # remove mentions
    tweet = re.sub(r'#', '', tweet)                 # remove hashtags
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)       # remove punctuation/symbols
    tweet = tweet.lower().split()
    tweet = [lm.lemmatize(word) for word in tweet if word not in stop_words]
    return ' '.join(tweet)

print("🧹 Cleaning tweets... please wait (this may take a few minutes)")

tqdm.pandas()  # add progress bar
data['cleaned_tweet'] = data['tweet'].progress_apply(clean_tweet)

print("✅ Tweets cleaned successfully!")

# ---------------- STEP 3: Feature Extraction (TF-IDF) ----------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_tweet']).toarray()
y = data['sentiment']

print("✅ Text vectorization completed!")

# ---------------- STEP 4: Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data split into training and testing sets!")

# ---------------- STEP 5: Train the Model ----------------
print("⚙️ Training Logistic Regression model...")
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# ---------------- STEP 6: Evaluate the Model ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ---------------- STEP 7: Save Model and Vectorizer ----------------
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("\n✅ Model and vectorizer saved successfully!")
print("🎉 Training complete! You can now run your Streamlit app.")
