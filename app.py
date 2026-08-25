import streamlit as st
import pandas as pd
import joblib

# Load the saved model pipeline from Week 4
model = joblib.load('titanic_pipeline.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below to check if they would have survived the Titanic disaster.")

# User inputs matching the pipeline features
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0.0, 100.0, 28.0)
fare = st.slider("Fare (£)", 0.0, 500.0, 32.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Predict button
if st.button("Predict Survival"):
    # Engineer the features inside the app to match our training pipeline
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    # Create input dataframe matching the exact column structure
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Embarked': [embarked],
        'FamilySize': [family_size],
        'IsAlone': [is_alone]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 This passenger likely **SURVIVED**! (Survival Probability: {probability*100:.2f}%)")
    else:
        st.error(f"❌ This passenger likely **DID NOT SURVIVE**. (Probability: {(1-probability)*100:.2f}%)")
