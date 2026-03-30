import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('LR_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_model()

# Same cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's **Real** or **Fake**.")

user_input = st.text_area("Paste News Text Here", height=300)

if st.button("🔍 Check News"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            transformed = vectorizer.transform([cleaned])
            prediction = model.predict(transformed)[0]
            
            if prediction == 1:
                st.success("✅ The news appears to be **REAL**")
            else:
                st.error("❌ The news appears to be **FAKE**")
    else:
        st.warning("Please enter some text to analyze.")
