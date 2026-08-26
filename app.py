import re
import pickle
import nltk
import streamlit as st

from nltk.corpus import stopwords


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# ---------------------------------------
# NLTK
# ---------------------------------------

nltk.download("stopwords")

stop_words = set(
    stopwords.words("english")
)


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
# Clean Text
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
# Remove Stopwords
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

    processed = remove_stopwords(
        cleaned
    )

    vectorized = tfidf.transform(
        [processed]
    )

    prediction = model.predict(
        vectorized
    )[0]

    probabilities = model.predict_proba(
        vectorized
    )[0]

    if prediction == 1:

        return "Positive", probabilities[1]

    else:

        return "Negative", probabilities[0]


# ---------------------------------------
# User Interface
# ---------------------------------------

st.title(
    "🎬 IMDb Sentiment Analysis"
)

st.write(
    "Enter a movie review and the model "
    "will predict whether it is positive "
    "or negative."
)


review = st.text_area(
    "Movie Review",
    height=200,
    placeholder=(
        "Example: This movie was amazing..."
    )
)


if st.button("Analyze Sentiment"):

    if review.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        sentiment, confidence = (
            predict_sentiment(review)
        )

        st.subheader(
            "Prediction"
        )

        st.write(
            f"Sentiment: **{sentiment}**"
        )

        st.write(
            f"Confidence: **{confidence:.2%}**"
        )

        if sentiment == "Positive":

            st.success(
                "😊 The review is Positive!"
            )

        else:

            st.error(
                "😞 The review is Negative!"
            )