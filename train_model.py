import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load datasets (change filenames if needed)
print("Loading datasets...")
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# Add labels
fake['label'] = 0   # 0 = Fake
true['label']  = 1  # 1 = Real

# Combine both
data = pd.concat([fake, true], axis=0).reset_index(drop=True)
data = data[['text', 'label']]   # Keep only needed columns

print("Cleaning text...")
data['text'] = data['text'].apply(clean_text)

# Features and target
X = data['text']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF Vectorizer
print("Vectorizing text...")
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
accuracy = model.score(X_test_tfidf, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
print("Saving model and vectorizer...")
joblib.dump(model, 'LR_model.joblib')
joblib.dump(tfidf, 'vectorizer.joblib')

print("Training completed successfully!!!")
