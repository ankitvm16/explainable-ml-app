import os

# Define folder and files structure
structure = {
    "app": [
        "__init__.py",
        "data_preprocessing.py",
        "model_training.py",
        "evaluation_metrics.py",
        "explainability.py",
        "utils.py"
    ],
    ".": [
        "streamlit_app.py",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]
}

# Boilerplate code for each file
boilerplate = {
    "app/__init__.py": "",
    "app/data_preprocessing.py": '''import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df.dropna(inplace=True)
    return df
''',
    "app/model_training.py": '''from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test
''',
    "app/evaluation_metrics.py": '''from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return acc, report
''',
    "app/explainability.py": '''import shap

def explain_with_shap(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    return shap_values
''',
    "app/utils.py": '''import os

def get_dataset_path():
    return os.path.join(os.getcwd(), 'data', 'bank-additional-full.csv')
''',
    "streamlit_app.py": '''import streamlit as st
from app.data_preprocessing import load_and_preprocess_data
from app.model_training import train_model
from app.model_evaluation import evaluate_model
from app.explainability import explain_with_shap
from app.utils import get_dataset_path

st.title("Explainable ML App")

file_path = get_dataset_path()
df = load_and_preprocess_data(file_path)

st.write("Data Preview", df.head())

y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
X = df.drop('y', axis=1)

model, X_test, y_test = train_model(X, y)
acc, report = evaluate_model(model, X_test, y_test)

st.write("Model Accuracy:", acc)
st.text(report)

st.subheader("SHAP Explanation")
shap_values = explain_with_shap(model, X_test[:5])
shap.plots.bar(shap_values)
''',
    "requirements.txt": '''streamlit
pandas
scikit-learn
shap
matplotlib
''',
    "README.md": '''# Explainable ML App

This Streamlit app offers explainability for tabular ML models using SHAP.

## Getting Started
- Clone this repo
- Create a virtual environment
- Install dependencies from `requirements.txt`
- Run with: `streamlit run streamlit_app.py`
''',
    ".gitignore": '''venv/
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
.env
'''
}

# Create the folder structure and write the files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        full_path = os.path.join(folder, file) if folder != "." else file
        with open(full_path, 'w') as f:
            content = boilerplate.get(os.path.join(folder, file), "")
            f.write(content)

print("✅ Project files created successfully with content.")
