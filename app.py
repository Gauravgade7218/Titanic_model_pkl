import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction")

def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(input_data)
    return "Survived" if prediction[0] == 1 else "Not Survived"

# User input fields
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.radio("Sex", ["Male", "Female"])
Sex = 1 if Sex == "Male" else 0
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 30.0)
Embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])
Embarked = 0 if Embarked == "C" else 1 if Embarked == "Q" else 2

if st.button("Predict Survival"):
    result = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    st.success(f"Prediction: {result}")


