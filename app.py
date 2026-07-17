import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Page configuration
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title("📧 Email & SMS Spam Detector")
st.markdown("Detect whether a message is **Spam** or **Not Spam** using Machine Learning.")

st.divider()

# Input
input_sms = st.text_area(
    "✍️ Enter your message",
    height=180,
    placeholder="Type your email or SMS here..."
)

# Statistics
if input_sms:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Characters", len(input_sms))
    with col2:
        st.metric("Words", len(input_sms.split()))

st.write("")

# Prediction
if st.button("🔍 Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]

        probability = model.predict_proba(vector_input)
        confidence = max(probability[0]) * 100

        st.divider()

        if result == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")

        st.progress(confidence / 100)

        st.write(f"### Confidence: **{confidence:.2f}%**")