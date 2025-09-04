
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "employee_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def train_model():
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df.dropna().copy()
    # Ensure correct dtypes
    numeric_cols = ["Age","YearsAtCompany","JobSatisfaction","MonthlyIncome"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # Target
    if "Attrition" not in df.columns:
        raise ValueError("Column 'Attrition' missing from dataset.")
    y = df["Attrition"].astype(str).str.strip().str.title().map({"Yes":1, "No":0})
    if y.isna().any():
        bad = df.loc[y.isna(), "Attrition"].unique().tolist()
        raise ValueError(f"Unexpected labels in Attrition: {bad}. Expected Yes/No.")
    # Check at least two classes
    if y.nunique() < 2:
        counts = y.value_counts().to_dict()
        raise ValueError(f"Need at least two classes in target. Got distribution: {counts}")

    # Features
    X = df.drop(columns=["Attrition"])

    cat_cols = ["Gender","Department","Education"]
    num_cols = ["Age","YearsAtCompany","JobSatisfaction","MonthlyIncome"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    model = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)

    pipe = Pipeline([("pre", pre), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("✅ Trained LogisticRegression pipeline")
    print("Accuracy:", round(acc, 3))
    print("Report:\n", classification_report(y_test, preds, digits=3))

    # Save entire pipeline (preprocessing + model)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"📦 Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
