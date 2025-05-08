import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import PartialDependenceDisplay
from alibi.explainers import ALE
import numpy as np

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

def pdp_explanation(model, x_train, features):
    st.subheader("Partial Dependence Plot (PDP)")
    display = PartialDependenceDisplay.from_estimator(model, x_train, features)
    plt.title("PDP for selected features")
    st.pyplot(display.figure_)

def ale_explanation(model, X_train, feature):
    # st.subheader("Accumulated Local Effects (ALE)")


    try:
        st.subheader(f"ALE Plot for feature: {feature}")
        X_np = X_train.to_numpy()
        feature_index = X_train.columns.get_loc(feature)

        ale = ALE(model.predict, feature_names=X_train.columns.tolist())
        exp = ale.explain(X_np)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(exp.feature_values[feature_index], exp.ale_values[feature_index])
        ax.set_title(f"ALE Plot for {feature}")
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("ALE Value")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to generate ALE plot: {e}")
        st.write(f"ALE plot for feature: {feature} not implemented correctly yet.)")


def counterfactual_explanation(model, X_train, y_train, X_test):
    st.subheader("Counterfactual Explanation")
    try:
        import dice_ml
        from dice_ml import Dice

        # Build the training DataFrame with target
        df_train = X_train.copy()
        df_train[y_train.name] = y_train

        # Identify continuous features
        continuous_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Create DiCE data and model objects
        dice_data = dice_ml.Data(
            dataframe=df_train,
            continuous_features=continuous_feats,
            outcome_name=y_train.name
        )
        dice_model = dice_ml.Model(
            model=model,
            backend="sklearn"
        )
        explainer = Dice(dice_data, dice_model)

        # Pick the first test row
        query_instance = X_test.iloc[0:1]

        # Generate a single counterfactual in the opposite class
        dice_exp = explainer.generate_counterfactuals(
            query_instance,
            total_CFs=1,
            desired_class="opposite"
        )

        # Display counterfactuals as a DataFrame
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        st.session_state.cf_original = X_test.iloc[0:1].reset_index(drop=True)
        st.session_state.cf_counterfactual = cf_df.reset_index(drop=True)
        # st.dataframe(cf_df)

    except Exception as e:
        st.error(f"Counterfactual explanation failed: {e}")
        st.write("Counterfactual explanations not implemented correctly yet.")
