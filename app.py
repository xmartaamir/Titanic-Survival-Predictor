import streamlit as st
import joblib
import numpy as np
import sklearn

# Model load karo
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("AI se check karein ke kya aap Titanic par bach paate?")

# User se input lena
pclass = st.selectbox("Ticket Class (1=Ameer, 3=Ghareeb)", [1, 2, 3])
sex = st.radio("Gender", ["Male", "Female"])
age = st.slider("Aapki Umar", 1, 100, 25)
family = st.number_input("Family members (Sath kitne thay?)", 0, 10)
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

# Data ko numbers mein badalna (Encoding)
sex_num = 0 if sex == "Male" else 1
title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
title_num = title_map[title]

# Prediction Button
if st.button("Predict"):
    features = np.array([[pclass, sex_num, age, family, title_num]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("Mubarak ho! AI kehta hai aap BACH JAYENGE! 🎉")
    else:
        st.error("Afsos! AI kehta hai aap NAHI BACH PAYENGE. 💔")