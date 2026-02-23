import streamlit as st
import joblib
import pandas as pd

# Navigation Sidebar
st.sidebar.title("ML Portfolio 🚀")
choice = st.sidebar.radio("Project Select Karein:", ["Titanic Survival", "Sentiment Analysis"])

# --- TITANIC SECTION ---
if choice == "Titanic Survival":
    st.title("🚢 Titanic Survival Predictor")
    # Yahan apna purana Titanic wala logic (inputs aur prediction) paste kar dein
    st.info("Passenger details enter karein taake survival predict kiya ja sakay.")

# --- SENTIMENT SECTION ---
elif choice == "Sentiment Analysis":
    st.title("🤖 AI Sentiment Analyzer")
    st.write("Amazon Reviews par train kiya gaya model.")

    # Load NLP Model & Vectorizer
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    user_text = st.text_area("Review likhein (e.g. 'I love this product'):")

    if st.button("Analyze"):
        if user_text:
            # Transformation
            data = vectorizer.transform([user_text])
            prediction = model.predict(data)
            
            # Result Display
            if prediction[0] == 1:
                st.success("Positive Review! 😊")
            else:
                st.error("Negative Review! ☹️")
        else:
            st.warning("Pehle kuch text likhein!")
