import streamlit as st
import joblib
import pandas as pd
import re

# Navigation Sidebar
st.sidebar.title("ML Portfolio 🚀")
choice = st.sidebar.radio("Project Select Karein:", ["Titanic Survival", "Sentiment Analysis"])

# --- TITANIC SECTION ---
if choice == "Titanic Survival":
    st.title("🚢 Titanic Survival Predictor")
    st.info("Passenger details enter karein taake survival predict kiya ja sakay.")

    # 1. Inputs (Ye missing thay tumhare code mein)
    pclass = st.selectbox("Ticket Class (1=Elite, 3=Economy):", [1, 2, 3])
    sex = st.selectbox("Gender:", ["male", "female"])
    age = st.slider("Age:", 1, 80, 25)
    family_size = st.number_input("Family Members on Board:", 0, 10, 0)
    
    # Sex ko numeric karna (male=0, female=1)
    sex_numeric = 1 if sex == "female" else 0

    if st.button("Predict Survival"):
        try:
            # Model load karein
            model_titanic = joblib.load('titanic_model.pkl')
            # Prediction (5 features: Pclass, Sex, Age, FamilySize, Title_Numeric)
            # Hum Title_Numeric ko filhal 1 (Mr/Miss) default de rahe hain
            prediction = model_titanic.predict([[pclass, sex_numeric, age, family_size, 1]])
            
            if prediction[0] == 1:
                st.success("Result: Is passenger ke bachne ke chances zyada hain! 🎉")
            else:
                st.error("Result: Is passenger ke bachne ke chances kam hain. ☹️")
        except Exception as e:
            st.error(f"Error loading Titanic model: {e}")

# --- SENTIMENT SECTION ---

     # --- SENTIMENT SECTION ---
elif choice == "Sentiment Analysis":
    st.title("🤖 AI Sentiment Analyzer")
    
    # 1. PEHLE LOAD KARO (Don't put this inside the button)
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except Exception as e:
        st.error(f"Files load nahi ho sakein: {e}")

    user_text = st.text_area("Review likhein:")

    if st.button("Analyze"):
        if user_text:
            # 2. AB USE KARO (Ab NameError nahi aayega)
            clean_text = user_text.lower()
            data = vectorizer.transform([clean_text])
            prediction = model.predict(data)
            
            if prediction[0] == 1:
                st.success("Positive Review! 😊")
            else:
                st.error("Negative Review! ☹️")

