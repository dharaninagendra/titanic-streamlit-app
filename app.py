import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚢 Titanic Survival Prediction App")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [1 if sex == 'male' else 0],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_C': [1 if embarked == 'C' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0]
})

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger **survived**.")
    else:
        st.error("❌ The passenger **did not survive**.")
