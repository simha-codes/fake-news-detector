# Project Report

**Title:** Fake News Detection Using Machine Learning

**Submitted by:** Vishwa  
**Date:** March 2026

---

## Introduction

In today's world, fake news spreads very quickly through social media and websites. It creates confusion among people and sometimes affects important things like elections and public opinion. Checking every news manually is almost impossible. 

So, I decided to build a system that can automatically detect whether a given news article is **Real** or **Fake** using Machine Learning.

---

## Objective

- To develop a model that can classify news as Real or Fake
- To clean and prepare the text data properly
- To build a simple and user-friendly web application
- To achieve good accuracy in prediction

---

## Dataset

I used two separate CSV files:
- `Fake.csv` → Contains fake news articles (Label = 0)
- `True.csv` → Contains real news articles (Label = 1)

I combined both files into one dataset. The dataset was taken from Kaggle.

---

## Methodology

First, I cleaned the news text by:
- Converting all text to lowercase
- Removing URLs, HTML tags, punctuation, and numbers

After cleaning, I used **TF-IDF Vectorizer** to convert the text into numerical features. Then I trained a **Logistic Regression** model on this data. 

After training, I saved the model and vectorizer using `joblib` so they can be used in the web app.

---

## Implementation

I created two main Python files:

1. **`train_model.py`**  
   - Loads the dataset
   - Cleans the text
   - Trains the Logistic Regression model
   - Saves the model and vectorizer

2. **`app.py`**  
   - Built using Streamlit
   - Provides a simple web interface
   - User can paste any news text and get instant prediction

---

## Results

The model achieved approximately **98% accuracy** on the test data.  
The web application works smoothly and gives fast predictions.

---

## Technologies Used

- Python
- Pandas (for data handling)
- Scikit-learn (for TF-IDF and Logistic Regression)
- Streamlit (for web app)
- Joblib (for saving model)

**IDE:** Visual Studio Code

---

## Conclusion

This project helped me understand how text classification works in Machine Learning. I learned important steps like data cleaning, feature extraction using TF-IDF, model training, and building a web application.

Overall, it was a great learning experience.

---

## Future Improvements

- Add confidence/probability score for each prediction
- Try advanced models like Naive Bayes or BERT
- Improve the design of the web app
- Deploy the app online so anyone can use it

---

## References

- YouTube Tutorial by Tensor Titans
- Scikit-learn Official Documentation
- Kaggle - Fake and Real News Dataset

---

**Thank You!**
