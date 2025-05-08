import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    labels = sorted(y_test.unique())
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generating classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Generating confusion_matrix
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    return report_df, matrix_df, accuracy

def display_evaluation_results(report_df, matrix_df, accuracy, key_prefix=""):
    st.header("Model Evaluation Results")

    # Displaying Accuracy
    st.subheader("Model Accuracy")
    st.metric("Accuracy:", f"{accuracy:.2%}")

    # Generating classification report chart
    st.subheader("Classification Report")
    st.dataframe(report_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
        'support': '{:.0f}'
    }).background_gradient(cmap='Blues', axis=0))

    # Generating confusion_matrix chart
    st.subheader("Confusion Matrix")
    fig = px.imshow(
        matrix_df,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig, key=f"{key_prefix}_confusion_matrix")
