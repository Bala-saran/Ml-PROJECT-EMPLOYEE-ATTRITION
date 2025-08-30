import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv("../data/employee_data.csv")

# Encode categorical columns
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Department"] = le.fit_transform(data["Department"])
data["Education"] = le.fit_transform(data["Education"])
data["Attrition"] = le.fit_transform(data["Attrition"])  # Yes=1, No=0

# Features & Target
X = data.drop("Attrition", axis=1)
y = data["Attrition"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Model trained with accuracy:", acc)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("📦 Model saved as model.pkl")
