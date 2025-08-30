import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("💼 Employee Attrition Prediction - IT Companies")

st.sidebar.header("Enter Employee Details")

# Input fields
age = st.sidebar.number_input("Age", 18, 60, 30)
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
department = st.sidebar.selectbox("Department", ("IT", "HR", "Finance"))
education = st.sidebar.selectbox("Education", ("Graduate", "Masters", "PhD"))
years = st.sidebar.slider("Years at Company", 0, 20, 3)
satisfaction = st.sidebar.slider("Job Satisfaction (1-5)", 1, 5, 3)
income = st.sidebar.number_input("Monthly Income", 10000, 100000, 30000)

# Encode inputs (same as train.py)
gender_val = 1 if gender == "Male" else 0
dept_val = {"IT": 2, "HR": 0, "Finance": 1}[department]
edu_val = {"Graduate": 0, "Masters": 1, "PhD": 2}[education]

# Create dataframe
input_data = pd.DataFrame([[age, gender_val, dept_val, edu_val, years, satisfaction, income]],
                          columns=["Age","Gender","Department","Education","YearsAtCompany","JobSatisfaction","MonthlyIncome"])

# Prediction
if st.sidebar.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    result = "⚠️ Likely to Leave" if prediction == 1 else "✅ Likely to Stay"
    st.subheader(result)
