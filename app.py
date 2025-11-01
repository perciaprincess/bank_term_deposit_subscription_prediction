import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/best_classifier.pkl")

st.set_page_config(page_title="Term Deposit Prediction", layout="centered")

# App Title
st.title("🏦 Bank Term Deposit Subscription Prediction")
st.write("Predict whether a client will subscribe to a term deposit offer.")

# --- Sidebar Information ---
st.sidebar.header("User Input Features")

# Collect user input
def user_input_features():
    age = st.sidebar.slider("Age", 18, 95, 35)
    job = st.sidebar.selectbox("Job", ['admin.', 'technician', 'services', 'management', 'retired', 
                                       'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'student', 'self-employed'])
    marital = st.sidebar.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.sidebar.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.sidebar.selectbox("Credit in Default?", ['yes', 'no'])
    balance = st.sidebar.number_input("Account Balance", min_value=-5000, max_value=100000, value=500)
    housing = st.sidebar.selectbox("Housing Loan", ['yes', 'no'])
    loan = st.sidebar.selectbox("Personal Loan", ['yes', 'no'])
    day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)
    month = st.sidebar.selectbox("Last Contact Month", 
                                 ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    duration = st.sidebar.number_input("Last Contact Duration (sec)", 0, 5000, 200)
    campaign = st.sidebar.slider("Number of Contacts During Campaign", 1, 50, 2)
    pdays = st.sidebar.slider("Days Since Last Contact (-1 if never)", -1, 999, 999)
    previous = st.sidebar.slider("Previous Contacts", 0, 50, 0)
    poutcome = st.sidebar.selectbox("Previous Campaign Outcome", ['unknown', 'failure', 'other', 'success'])
    
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Show user input
st.subheader("🔍 Entered Client Data")
st.write(input_df)

# Prediction
if st.button("Predict Subscription"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1][0]
    st.error({prediction[0] })
    if prediction[0] == 'yes':
        st.success(f"✅ Client is **likely to subscribe** to a term deposit. (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Client is **not likely to subscribe**. (Probability: {probability:.2f})")

st.markdown("---")
st.caption("Developed by Percia Princess | Data Science Project | Day 7 – Streamlit Deployment")
