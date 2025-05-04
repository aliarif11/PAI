
from flask import Flask, render_template, request
import random
import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Prepare data
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression()
model.fit(X, tags)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    X_test = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_test)[0]

    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."

if __name__ == "__main__":
    app.run(debug=True)
