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

df = pd.read_csv('bullying_tweets.csv')

X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['bully_status'], random_state=42)
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the tweet from the user
        tweet = request.form['tweet']
        # Make a prediction
        file = 'bullying_model2.sav'
        loaded_model = pickle.load(open(file, 'rb'))
        # print(stopwords.words('english')[:20])
        tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=1000)
        # print(df['cleaned_tweet_text_without_hashtags'])
        # tweetText = tfidf_vectorizer.fit_transform(df['tweet_text'])
        input = [tweet]
        input_data = tfidf_vectorizer.transform(input)

        predictions = loaded_model.predict(input_data)
        prediction = "Bullying" if predictions[0] == 1 else "Not Bullying"
        return render_template('index.html', prediction=prediction)


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # Get the tweet from the user
        tweet = request.form['tweet']
        # Make a prediction
        filename = 'bully_model_NB.sav'
        clf = pickle.load(open(filename, 'rb'))
        count_vector = CountVectorizer(stop_words = 'english', lowercase = True, ngram_range=(1,1))
        training_data = count_vector.fit_transform(X_train)
        input = count_vector.transform([tweet])
        predictions = clf.predict(input)
        prediction = "Bullying" if predictions[0] == 1 else "Not Bullying"
        return render_template('index.html', prediction=prediction)
        
        