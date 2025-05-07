import streamlit as st
import pandas as pd
import io
import csv
import pickle

from app.data_preprocessing import preprocess_data
from app.model_training    import train_model
from app.model_evaluation  import evaluate_model
from app.explainability    import shap_explanation, lime_explanation

from app.llm_explain import llm_explain, make_metrics_prompt, make_shap_prompt

st.set_page_config(page_title="Explainable ML App", layout="wide")
st.title("Explainable Machine Learning App")

mode = st.radio(
    "Select the mode",
    ["Option 1: Train the classification model and receive explainable metrics",
     "Option 2: Generate explainable metrics by uploading model weights and input dataset"]
)

if mode == "Option 1: Train the classification model and receive explainable metrics":
    st.info("Upload your dataset (in CSV format) to train and evaluate the trained model")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        content = uploaded_file.read()
        try:
            first_line = content.decode("utf-8").splitlines()[0]
            dialect = csv.Sniffer().sniff(first_line)
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=delimiter)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        target_column = st.selectbox("Select the target column", df.columns)

        if st.button("Run Analysis"):
            st.info("Preprocessing data...")
            (X_train, X_test, y_train, y_test), encoders = preprocess_data(df, target_column)

            st.info("🏋️‍♂️ Training model...")
            model = train_model(X_train, y_train)

            st.info("Evaluating model...")
            report, matrix, accuracy = evaluate_model(model, X_test, y_test)

            st.subheader("Model Evaluation Results")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write("**Classification Report:**")
            st.json(report)
            st.write("**Confusion Matrix:**")
            st.dataframe(matrix)

            st.info("Generating SHAP explanation...")
            shap_explanation(model, X_train, X_test)

            st.info("Generating LIME explanation...")
            lime_explanation(model, X_train, X_test)

            st.info("Generating summary of metrics…")
            metrics_prompt = make_metrics_prompt(accuracy, report)
            metrics_explanation = llm_explain(metrics_prompt)
            st.markdown("### Model Performance Explanation")
            st.write(metrics_explanation)

            st.info("Explaining SHAP visualizations…")
            shap_prompt = make_shap_prompt()
            shap_explanation_text = llm_explain(shap_prompt)
            st.markdown("### SHAP Charts Explanation")
            st.write(shap_explanation_text)

elif mode == "Option 2: Generate explainable metrics by uploading model weights and input dataset":
    st.info("Upload a pre-trained model and the full dataset to run evaluation & explainability")
    model_file = st.file_uploader("Upload Pickled Model (.pkl)", type=["pkl"])
    data_file  = st.file_uploader("Upload Full Dataset CSV", type=["csv"])

    df = None
    if data_file:
        raw = data_file.read().decode("utf-8")
        try:
            delim = csv.Sniffer().sniff(raw.splitlines()[0]).delimiter
        except csv.Error:
            delim = ","
        df = pd.read_csv(io.StringIO(raw), sep=delim)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    target_column = None
    if df is not None:
        target_column = st.selectbox("Select the target column", df.columns)

    if st.button("Run Evaluation & Explain") and model_file and df is not None and target_column:
        st.info("Preprocessing & splitting data...")
        (X_train, X_test, y_train, y_test), encoders = preprocess_data(df, target_column)

        st.info("Loading pre-trained model...")
        model_file.seek(0)
        model = pickle.load(model_file)

        st.info("Evaluating model...")
        report, matrix, accuracy = evaluate_model(model, X_test, y_test)

        st.subheader("Model Evaluation Results")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write("**Classification Report:**")
        st.json(report)
        st.write("**Confusion Matrix:**")
        st.dataframe(matrix)

        st.info("Generating SHAP explanation...")
        shap_explanation(model, X_train, X_test)

        st.info("Generating LIME explanation...")
        lime_explanation(model, X_train, X_test)

        st.info("Generating summary of metrics…")
        metrics_prompt = make_metrics_prompt(accuracy, report)
        metrics_explanation = llm_explain(metrics_prompt)
        st.markdown("### 🧾 Model Performance Explanation")
        st.write(metrics_explanation)

        st.info("Explaining SHAP visualizations…")
        shap_prompt = make_shap_prompt()
        shap_explanation_text = llm_explain(shap_prompt)
        st.markdown("### SHAP Charts Explanation")
        st.write(shap_explanation_text)
