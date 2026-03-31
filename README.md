# Fake News Detector

A simple web app that tells whether a news article is **Real** or **Fake** using Machine Learning.

---

## What is this project?

This project detects fake news by analyzing the text of the article. I built a machine learning model that learns from real and fake news examples and then predicts new articles.

I also made a simple web interface using Streamlit so anyone can use it easily.

---

## Features

- Cleans the news text automatically
- Uses TF-IDF and Logistic Regression model
- Simple and clean web interface
- Fast prediction
- Easy to run on your computer

---

## Project Files

- `train_model.py` → Trains the model and saves it
- `app.py` → The main web application
- `Fake.csv` & `True.csv` → Dataset files
- `LR_model.joblib` → Trained model
- `vectorizer.joblib` → Text converter
- `requirements.txt` → List of packages needed

---

## How to Run Locally

### Step 1: Clone or Download the project

### Step 2: Open terminal in the project folder and run:

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
