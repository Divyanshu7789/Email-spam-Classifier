import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['target', 'text']

# Convert labels
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Vectorize
tfidf = TfidfVectorizer(stop_words='english')

X = tfidf.fit_transform(df['text'])
y = df['target']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save files
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")