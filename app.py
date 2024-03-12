from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

df = pd.read_csv('cleaned_tweets.csv')

X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['bully_status'], random_state=42)
    
@app.route("/")
def index():
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # Get the tweet from the user
        tweet = request.form['tweet']
        # Make a prediction
        clf = pickle.load(open('NB_Model.sav', 'rb'))
        vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
        
        input = vectorizer.transform([tweet])
        predictions = clf.predict(input)
        prediction = "Bullying" if predictions[0] == 1 else "Not Bullying"
        return render_template('index.html', prediction=prediction)
        