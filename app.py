import streamlit as st
import numpy as np
import joblib

# ✅ Load model and scaler
model = joblib.load("salary_model.joblib")
scaler = joblib.load("scaler.joblib")

# 🎨 Page config
st.set_page_config(page_title="Salary Prediction", layout="centered")
st.title("💼 Salary Prediction Web App")
st.markdown("Estimate if income is >50K or ≤50K based on key metrics.")

# 🧮 UI Inputs
age = st.slider("Age", 18, 75, 30)
education_num = st.slider("Education Level (1–16)", 1, 16, 10)
occupation = st.selectbox("Occupation", [
    'Prof-specialty', 'Exec-managerial', 'Sales', 'Other-service',
    'Craft-repair', 'Adm-clerical', 'Machine-op-inspct', 'Transport-moving',
    'Handlers-cleaners', 'Tech-support', 'Protective-serv', 'Farming-fishing',
    'Priv-house-serv', 'Armed-Forces', 'others'
])
workclass = st.selectbox("Workclass", [
    'Private', 'Self-emp-not-inc', 'Local-gov',
    'State-gov', 'Self-emp-inc', 'Federal-gov', 'Notlisted'
])
hours_per_week = st.slider("Hours Worked Per Week", 1, 100, 40)

# Add inputs for the remaining features
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
relationship = st.selectbox("Relationship", ['Husband', 'Own-child', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Portugal', 'Iran', 'Nicaragua', 'Greece', 'Peru', 'Ecuador', 'France', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Honduras', 'Hungary', 'Holand-Netherlands'])


# 🔡 Encode dropdowns manually
def encode(value, options):
    return options.index(value)

occupation_encoded = encode(occupation, [
    'Prof-specialty', 'Exec-managerial', 'Sales', 'Other-service',
    'Craft-repair', 'Adm-clerical', 'Machine-op-inspct', 'Transport-moving',
    'Handlers-cleaners', 'Tech-support', 'Protective-serv', 'Farming-fishing',
    'Priv-house-serv', 'Armed-Forces', 'others'
])

workclass_encoded = encode(workclass, [
    'Private', 'Self-emp-not-inc', 'Local-gov',
    'State-gov', 'Self-emp-inc', 'Federal-gov', 'Notlisted'
])

marital_status_encoded = encode(marital_status, ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
relationship_encoded = encode(relationship, ['Husband', 'Own-child', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'])
race_encoded = encode(race, ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender_encoded = encode(gender, ['Male', 'Female'])
native_country_encoded = encode(native_country, ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Portugal', 'Iran', 'Nicaragua', 'Greece', 'Peru', 'Ecuador', 'France', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Honduras', 'Hungary', 'Holand-Netherlands'])


# 🔢 Format input for prediction - Ensure all 13 features are present in the correct order
user_input = np.array([[
    age,
    workclass_encoded,
    0, # Placeholder for fnlwgt (removed during preprocessing)
    education_num,
    marital_status_encoded,
    occupation_encoded,
    relationship_encoded,
    race_encoded,
    gender_encoded,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country_encoded
]])

# Debugging: Print the shape of the user input
print("Shape of user_input:", user_input.shape)

# 🧮 Scale input
input_scaled = scaler.transform(user_input)

# 🎯 Predict
if st.button("Predict Income Category"):
    pred = model.predict(input_scaled)[0]
    result = ">50K" if pred == 1 else "≤50K"
    st.success(f"💰 Predicted Income: {result}")
#2


