# 💬 Twitter Sentiment Analysis App

An intelligent **Machine Learning + NLP** project that analyzes tweets and classifies their sentiment as **Positive**, **Negative**, or **Neutral**.  
This project combines **Python, Data Science, and Streamlit** to create an interactive web app for real-time sentiment prediction. 🚀  

---

## 🎯 **Objective**

The goal of this project is to automatically determine the **emotion or opinion** expressed in a tweet —  
whether it’s **positive**, **negative**, or **neutral** — using Natural Language Processing (NLP) and Machine Learning.

---

## 🧠 **Key Features**

✅ Preprocessed and cleaned 50,000+ tweets from the Sentiment140 dataset  
✅ Used **TF-IDF Vectorization** for converting text to numerical features  
✅ Trained **Logistic Regression** and **Naive Bayes** models for sentiment classification  
✅ Achieved up to **87% model accuracy**  
✅ Developed an interactive **Streamlit Web App** for live tweet sentiment prediction  
✅ Supported **neutral sentiment detection** for balanced emotion analysis  

---

## 🛠️ **Tech Stack & Tools**

| Category | Technologies Used |
|-----------|-------------------|
| 💻 Programming | Python |
| 📊 Libraries | Pandas, NumPy, NLTK, Scikit-learn |
| 🧠 NLP | TF-IDF Vectorizer, Lemmatization, Stopword Removal |
| 🌐 Web Framework | Streamlit |
| 🖼️ Visualization | Matplotlib, WordCloud |
| 💾 Model | Logistic Regression |
| 📚 Dataset | [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) |

---

## ⚙️ **How It Works**

1. **Data Preprocessing**
   - Remove URLs, mentions, hashtags, and special characters  
   - Convert text to lowercase and lemmatize words  
   - Remove stopwords like “the”, “and”, “is”, etc.  

2. **Feature Extraction**
   - Convert tweets into numerical vectors using **TF-IDF**  

3. **Model Training**
   - Train a **Logistic Regression model** on labeled sentiment data  

4. **Prediction**
   - Predict whether a given tweet is **Positive**, **Negative**, or **Neutral**  

5. **Web App**
   - Built using **Streamlit**, allowing users to type a tweet and get instant analysis  

---

## 🧩 **Project Structure**
```
Tweets/
│
├── app.py # Streamlit web app
├── model.py # Model training and preprocessing
├── sentiment_model.pkl # Trained ML model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore # Files and folders ignored by Git
└── data/ # (Optional) Dataset folder (ignored in Git
```
2️⃣ Create and activate virtual environment
python -m venv sentiment_env
sentiment_env\Scripts\activate    # for Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py


🌍 Deployment

This project can be deployed on Streamlit Community Cloud easily.
Simply connect your GitHub repo and deploy the app online.

Live Demo (Example):
🔗 https://twitter-sentiment-analysis.streamlit.app
 (replace with your actual deployed URL)


 💡 Skills Demonstrated

Natural Language Processing (NLP)
Text Cleaning & Preprocessing
Feature Engineering (TF-IDF)
Machine Learning Model Building
Model Evaluation & Visualization
Web Application Development (Streamlit)
Communication & Presentation Skills


🚀 Future Enhancements

🧠 Implement Deep Learning (BERT / LSTM) models
💬 Add live tweet scraping using Tweepy
📊 Visualize sentiment trends over time
🌍 Support for multilingual sentiment analysis
