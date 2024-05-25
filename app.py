from flask import Flask, request, jsonify, render_template # type: ignore
import re
import pickle
# from spellchecker import SpellChecker
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
import pandas as pd # type: ignore
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove all extra whitespaces
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load the pre-trained voting model
model = pickle.load(open("youtube_model4.pkl", "rb"))

# Initialize the spell checker
# spell = SpellChecker()

# Create a Flask application instance
app = Flask(__name__)

def contains_special_characters(comment):
    return bool(re.search(r'[^a-zA-Z0-9\s,.;"\'-]', comment))

# def has_spelling_errors(comment):
#     words = comment.split()
#     misspelled_words = spell.unknown(words)
#     return len(misspelled_words) > 0

def predict_sentiment(comment):
    if contains_special_characters(comment):
        return "Special characters not accepted."
    # if has_spelling_errors(comment):
    #     return "Spelling errors detected."
    
    comment_cleaned = clean_text(comment)
    sentiment = model.predict([comment_cleaned])
    return sentiment[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form
    comment = data['comment']
    predicted_sentiment = predict_sentiment(comment)

    if predicted_sentiment in ["Special characters not accepted."]:
        result = predicted_sentiment
    else:
        result = predicted_sentiment

    return jsonify({'comment': comment, 'sentiment': result})

if __name__ == "__main__":
    app.run(debug=True)
