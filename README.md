# Email & SMS Spam Detector

A Machine Learning web application that classifies Email and SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

## Features

- Detect Spam and Non-Spam messages
- Prediction confidence score
- Character and word count
- Interactive Streamlit interface
- Fast real-time predictions

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy

## Project Structure

```
Email-spam-Classifier/
│── app.py
│── train_model.py
│── model.pkl
│── vectorizer.pkl
│── spam.csv
│── requirements.txt
│── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Divyanshu7789/Email-spam-Classifier.git
```

Move to the project directory:

```bash
cd Email-spam-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Machine Learning Pipeline

1. Data Cleaning
2. Text Preprocessing
3. TF-IDF Vectorization
4. Multinomial Naive Bayes Classification
5. Spam Prediction

## 📸 Application Preview

Then display it using:

```markdown
![Application Screenshot](images/demo.png)
```

## Model

- **Vectorizer:** TF-IDF
- **Algorithm:** Multinomial Naive Bayes

## Future Improvements

- Support multiple ML models
- Email file upload
- Dark mode UI
- Deploy on Streamlit Cloud

## Author

**Divyanshu Tiwari**

GitHub: https://github.com/Divyanshu7789