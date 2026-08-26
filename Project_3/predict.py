import re
import pickle
import nltk

from nltk.corpus import stopwords


# ---------------------------------------
# Load NLTK stopwords
# ---------------------------------------

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


# ---------------------------------------
# Load Model
# ---------------------------------------

with open(
    "model/sentiment_model.pkl",
    "rb"
) as file:

    model = pickle.load(file)


with open(
    "model/tfidf_vectorizer.pkl",
    "rb"
) as file:

    tfidf = pickle.load(file)


# ---------------------------------------
# Text Cleaning
# ---------------------------------------

def clean_text(text):

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ---------------------------------------
# Stopword Removal
# ---------------------------------------

def remove_stopwords(text):

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ---------------------------------------
# Prediction
# ---------------------------------------

def predict_sentiment(review):

    cleaned = clean_text(review)

    processed = remove_stopwords(cleaned)

    vectorized = tfidf.transform(
        [processed]
    )

    prediction = model.predict(
        vectorized
    )[0]

    if prediction == 1:
        return "Positive"

    return "Negative"


# ---------------------------------------
# User Input
# ---------------------------------------

review = input(
    "\nEnter a movie review: "
)

result = predict_sentiment(
    review
)

print(
    "\nPredicted Sentiment:",
    result
)