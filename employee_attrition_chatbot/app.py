
import streamlit as st
import pandas as pd
import pickle
from typing import List, Dict, Any

st.set_page_config(page_title="Employee Attrition Chatbot", page_icon="🤖", layout="centered")

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipe = load_model()

st.title("🤖 Employee Attrition Prediction — Chatbot")
st.caption("Answer a few questions and I'll estimate if the employee is likely to stay or leave.")

# Define the ordered questions and simple validators
QUESTIONS = [
    ("Age", "Please enter age (18-65)", lambda x: x.isdigit() and 18 <= int(x) <= 65, "Enter a whole number between 18 and 65."),
    ("Gender", "Gender? (Male/Female)", lambda x: x.strip().lower() in {"male","female"}, "Type Male or Female."),
    ("Department", "Department? (IT/HR/Finance)", lambda x: x.strip().lower() in {"it","hr","finance"}, "Choose one: IT, HR, Finance."),
    ("Education", "Education level? (Graduate/Masters/PhD)", lambda x: x.strip().lower() in {"graduate","masters","phd"}, "Choose one: Graduate, Masters, PhD."),
    ("YearsAtCompany", "Years at company? (0–40)", lambda x: x.isdigit() and 0 <= int(x) <= 40, "Enter a whole number between 0 and 40."),
    ("JobSatisfaction", "Job satisfaction (1–5)", lambda x: x.isdigit() and 1 <= int(x) <= 5, "Enter a whole number between 1 and 5."),
    ("MonthlyIncome", "Monthly income?", lambda x: x.replace('_','').replace(',','').replace('.','',1).isdigit(), "Enter a numeric value (e.g., 45000)."),
]

def normalize_value(key: str, val: str):
    v = val.strip()
    if key in {"Gender","Department","Education"}:
        return v.title()
    if key in {"Age","YearsAtCompany","JobSatisfaction"}:
        return int(float(v))
    if key == "MonthlyIncome":
        v = v.replace(",","").replace("_","")
        return float(v)
    return v

# Session state
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []
if "answers" not in st.session_state:
    st.session_state.answers: Dict[str, Any] = {}
if "step" not in st.session_state:
    st.session_state.step = 0
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def ask_next_question():
    if st.session_state.step < len(QUESTIONS):
        key, prompt, _, _ = QUESTIONS[st.session_state.step]
        with st.chat_message("assistant"):
            st.markdown(prompt)

def make_prediction():
    ans = st.session_state.answers
    row = {
        "Age": ans["Age"],
        "Gender": ans["Gender"],
        "Department": ans["Department"],
        "Education": ans["Education"],
        "YearsAtCompany": ans["YearsAtCompany"],
        "JobSatisfaction": ans["JobSatisfaction"],
        "MonthlyIncome": ans["MonthlyIncome"],
    }
    X = pd.DataFrame([row])

    pred = pipe.predict(X)[0]
    proba = getattr(pipe, "predict_proba", lambda x: [[None, None]])(X)[0]

    label = "⚠️ Likely to Leave" if pred == 1 else "✅ Likely to Stay"

    prob_text = ""
    if proba is not None and len(proba) > 0:
        prob_text = f" (confidence: {max(proba):.2f})"

    with st.chat_message("assistant"):
        st.markdown(f"**Prediction:** {label}{prob_text}")

    st.session_state.predicted = True


# Intro if first load
if st.session_state.step == 0 and not st.session_state.messages:
    st.session_state.messages.append({"role":"assistant", "content": "Hi! Let's get started."})

# Show next question (if needed)
if not st.session_state.predicted:
    ask_next_question()

# Chat input
user_input = st.chat_input("Type your answer and press Enter")
if user_input is not None and user_input.strip() != "":
    step = st.session_state.step
    if step < len(QUESTIONS):
        key, prompt, validator, err = QUESTIONS[step]
        if validator(user_input):
            value = normalize_value(key, user_input)
            st.session_state.answers[key] = value
            st.session_state.messages.append({"role":"user", "content": user_input})
            st.session_state.step += 1
        else:
            st.session_state.messages.append({"role":"assistant", "content": f"❗ {err}"})
    if st.session_state.step < len(QUESTIONS):
        ask_next_question()
    else:
        if not st.session_state.predicted:
            make_prediction()

# Utility sidebar
with st.sidebar:
    st.header("ℹ️ Info")
    st.write("This chatbot uses a scikit-learn pipeline (OneHotEncoder + LogisticRegression).")
    if st.button("Reset conversation"):
        st.session_state.clear()
        st.rerun()
