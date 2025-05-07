import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit as st

def shap_explanation(model, x_train, x_test):
    st.subheader("SHAP Explanation")

    # Limit sample size
    x_sample = x_test[:100]

    # Choose appropriate explainer
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_sample)

    # For classification: shap_values has multiple outputs
    if isinstance(shap_values.values, list) or len(shap_values.values.shape) == 3:
        st.write("Detected classification task - using SHAP values for positive class.")
        shap_values = shap_values[:, :, 1]  # use class 1 (positive) explanations

    # Beeswarm Plot
    st.write("Feature importance (SHAP Beeswarm Plot):")
    fig_summary = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig_summary)

    # Bar Plot
    st.write("SHAP Bar Plot:")
    fig_bar = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig_bar)

def lime_explanation(model, x_train, x_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train.values,
        feature_names=x_train.columns,
        class_names=['No', 'Yes'],
        mode='classification'
    )
    exp = explainer.explain_instance(
        data_row=x_test.iloc[0].values,
        predict_fn=model.predict_proba
    )
    st.subheader("LIME Explanation for First Test Instance")
    st.pyplot(exp.as_pyplot_figure())