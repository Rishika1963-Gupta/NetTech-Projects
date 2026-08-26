import pandas as pd
import re
import nltk
import pickle
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------
# Load Dataset

DATA_PATH = "dataset/IMDB Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# ---------------------------------------
# Text Cleaning

def clean_text(text):

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text) #html tag
    text = text.lower() #lower case 
    text = re.sub(r"[^a-zA-Z\s]", " ", text) #spcl character
    text = re.sub(r"\s+", " ", text).strip() #extra space

    return text

df["clean_review"] = df["review"].apply(clean_text)

print("\nOriginal review:")
print(df["review"].iloc[0])

print("\nCleaned review:")
print(df["clean_review"].iloc[0])


# ---------------------------------------
# Remove Stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def remove_stopwords(text):

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["processed_review"] = df["clean_review"].apply(
    remove_stopwords
)

print("\nProcessed review:")
print(df["processed_review"].iloc[0])


# ---------------------------------------
# Create Numerical Labels

df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print("\nLabel distribution:")
print(df["label"].value_counts())


# ---------------------------------------
# Train/Test Split

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["processed_review"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))

# ---------------------------------------
# TF-IDF


tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print("\nTF-IDF training shape:", X_train.shape)
print("TF-IDF testing shape:", X_test.shape)


# ---------------------------------------
# Train Logistic Regression

model = LogisticRegression(
    max_iter=1000
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------------
# Predictions
y_pred = model.predict(X_test)

# ---------------------------------------
# Accuracy


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ---------------------------------------
# Save Model

os.makedirs("model", exist_ok=True)

with open(
    "model/sentiment_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


with open(
    "model/tfidf_vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tfidf,
        file
    )


print("\nModel saved successfully!")