
# Employee Attrition Chatbot (Streamlit)

A simple end-to-end example that trains a Logistic Regression model and serves predictions via a **chat-style** Streamlit app.

## 🗂️ Project Structure
```
employee_attrition_chatbot/
├── app.py
├── train.py
├── model.pkl          # created after training
├── data/
│   └── employee_data.csv
├── requirements.txt
└── Procfile
```

## 🚀 Quickstart (Local)

1. **Create a virtual environment & install deps**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python train.py
   ```
   This will create `model.pkl` (a scikit-learn Pipeline including preprocessing).

3. **Run the chatbot**
   ```bash
   streamlit run app.py
   ```

4. Open the URL printed by Streamlit (usually http://localhost:8501).

## 🧠 Model
- Preprocessing: `OneHotEncoder(handle_unknown="ignore")` for categorical columns, passthrough for numerics.
- Classifier: `LogisticRegression(class_weight="balanced")`.
- Target: `Attrition` (Yes/No).

## 🌐 Deploy on Render/Heroku (optional)
- Use the `Procfile` with the command:
  ```
  web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
  ```

## ✨ Notes
- You can replace `data/employee_data.csv` with your own, as long as it has these columns:
  `Age, Gender, Department, Education, YearsAtCompany, JobSatisfaction, MonthlyIncome, Attrition`.
- Labels must be `Yes`/`No`.
