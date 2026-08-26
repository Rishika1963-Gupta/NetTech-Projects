import pandas as pd

# Dataset location
file_path = "dataset/IMDB Dataset.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())