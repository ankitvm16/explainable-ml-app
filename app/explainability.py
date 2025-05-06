import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit as st

def shap_explanation(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

def lime_explanation(model, X_train, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['No', 'Yes'],
        mode='classification'
    )
    exp = explainer.explain_instance(
        data_row=X_test.iloc[0].values,
        predict_fn=model.predict_proba
    )
    st.subheader("LIME Explanation for First Test Instance")
    st.pyplot(exp.as_pyplot_figure())
