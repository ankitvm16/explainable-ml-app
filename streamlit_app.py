import streamlit as st
import pandas as pd
import io
import csv

from app.data_preprocessing import preprocess_data
from app.model_training import train_model
from app.model_evaluation import evaluate_model
from app.explainability import shap_explanation, lime_explanation

st.set_page_config(page_title="Explainable ML App", layout="wide")
st.title("🔍 Explainable Machine Learning App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Try detecting the delimiter (defaults to comma)
    content = uploaded_file.read()
    try:
        # Detect delimiter from first line
        first_line = content.decode('utf-8').splitlines()[0]
        dialect = csv.Sniffer().sniff(first_line)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ','  # fallback

    df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=delimiter)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_column = st.selectbox("🎯 Select the target column", df.columns)

    if st.button("Run Analysis"):
        st.info("🔄 Preprocessing data...")
        (X_train, X_test, y_train, y_test), le_dict = preprocess_data(df, target_column)

        st.info("🏋️‍♂️ Training model...")
        model = train_model(X_train, y_train)

        st.info("📈 Evaluating model...")
        report, matrix, accuracy = evaluate_model(model, X_test, y_test)

        st.subheader("✅ Model Evaluation Results")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write("**Classification Report:**")
        st.json(report)
        st.write("**Confusion Matrix:**")
        st.dataframe(matrix)

        st.info("💡 Generating SHAP explanation...")
        shap_explanation(model, X_train, X_test)

        st.info("💡 Generating LIME explanation...")
        lime_explanation(model, X_train, X_test)

        st.success("🎉 Analysis Complete!")