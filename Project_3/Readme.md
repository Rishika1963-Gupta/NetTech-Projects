# Trained Model

The trained model files are generated automatically by `train_model.py`.

The generated `.pkl` files are not included in this repository.


# 🎬 IMDb Sentiment Analysis

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning
to classify IMDb movie reviews as positive or negative.

## Objective

The objective is to build a machine learning system that can automatically
analyze a movie review and determine whether the sentiment is positive or
negative.

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Dataset

The project uses the IMDb Movie Reviews dataset containing 50,000 reviews.

Each review is labeled as either:

- Positive
- Negative

The dataset is not included in this repository because of its large size.

## Machine Learning Workflow

IMDb Dataset
↓
Text Cleaning
↓
Stopword Removal
↓
TF-IDF Feature Extraction
↓
Logistic Regression
↓
Sentiment Prediction

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
