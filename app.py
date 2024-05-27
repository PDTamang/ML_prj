from flask import Flask, request, jsonify, render_template
import re
import pickle
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize necessary components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def clean_text(text):
    # Keep common punctuation and remove others
    text = re.sub(r'[^\w\s.,\'?-]', ' ', text)  # Keep common punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove all extra whitespaces
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word.lower() not in stop_words])
    return text.strip()

# Load the pre-trained voting model
model = pickle.load(open("youtube_model7.pkl", "rb"))

def contains_special_characters(comment):
    # Allow common punctuation and filter out only non-standard characters
    return bool(re.search(r'[^a-zA-Z0-9\s,.;"\'-]', comment))

# def has_spelling_errors(comment):
#     # Remove common punctuation before checking for spelling errors
#     words = comment.split()
#     misspelled_words = spell.unknown(words)
#     return len(misspelled_words) > 0

def has_spelling_errors(comment):
    # Define regular expression pattern to match common punctuation
    punctuation_pattern = r'[,.]'
    
    # Remove punctuation from the comment
    comment_without_punctuation = re.sub(punctuation_pattern, '', comment)
    
    # Split the comment into words
    words = comment_without_punctuation.split()
    
    # Get misspelled words
    misspelled_words = spell.unknown(words)
    
    # Filter out any misspelled words that are single characters (likely false positives)
    misspelled_words = [word for word in misspelled_words if len(word) > 1]
    
    return len(misspelled_words) > 0

def predict_sentiment(comment):
    if contains_special_characters(comment):
        return "Special characters not accepted."
    if has_spelling_errors(comment):
        return "Spelling errors detected."
    
    comment_cleaned = clean_text(comment)
    sentiment = model.predict([comment_cleaned])
    return sentiment[0]

# Create a Flask application instance
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form
    comment = data['comment']
    predicted_sentiment = predict_sentiment(comment)

    if predicted_sentiment in ["Special characters not accepted.", "Spelling errors detected."]:
        result = predicted_sentiment
    else:
        result = predicted_sentiment

    return jsonify({'comment': comment, 'sentiment': result})

# if _name__ == "_main_":
#     app.run(debug=True)