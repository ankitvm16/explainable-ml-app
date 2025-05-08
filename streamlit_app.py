import streamlit as st
import pandas as pd
import numpy as np
import shap as _shap
import io, csv, pickle
from lime.lime_tabular import LimeTabularExplainer
from app.data_preprocessing import preprocess_data
from app.model_training    import train_model
from app.evaluation_metrics import evaluate_model, display_evaluation_results
from app.explainability    import (
    shap_explanation,
    lime_explanation,
    pdp_explanation,
    ale_explanation,
    counterfactual_explanation
)

from app.llm_explain import llm_explain, make_metrics_prompt, make_charts_prompt

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Explainable AI Dashboard", layout="wide")
st.title("Explainable AI Dashboard")

# ─── Reset Function ─────────────────────────────────────────────────────────────
def reset_app():
    st.session_state.clear()
    st.rerun()

# Always show reset at bottom:
show_reset = True

# ─── Session State Init ────────────────────────────────────────────────────────
if "phase" not in st.session_state:
    st.session_state.phase = "init"
if "charts" not in st.session_state:
    st.session_state.charts = []

# ─── Mode Selection ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Select Mode",
    [
        "Option 1: Train a model & generate model explainability metrics",
        "Option 2: Upload a dataset and model & generate model explainability metrics"
    ]
)

# ─── PHASE 1: Train or Upload ───────────────────────────────────────────────────
if st.session_state.phase == "init":
    if mode == "Option 1: Train a model & generate model explainability metrics":
        st.info("Upload CSV to train a model and get explanations")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            raw = uploaded.read().decode("utf-8")
            try:
                delim = csv.Sniffer().sniff(raw.splitlines()[0]).delimiter
            except csv.Error:
                delim = ","
            df = pd.read_csv(io.StringIO(raw), sep=delim)

            st.subheader("Data Preview")
            st.dataframe(df.head())

            target = st.selectbox("Select the Target Column to train the model and run the analysis", df.columns)
            if st.button("Run Evaluation & Explain"):
                # train/evaluate
                # st.session_state.phase = "trained"
                (X_train, X_test, y_train, y_test), _ = preprocess_data(df, target)
                model = train_model(X_train, y_train)

                # store
                st.session_state.model   = model
                st.session_state.X_train = X_train
                st.session_state.X_test  = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test  = y_test

                # evaluate
                report_df, matrix_df, accuracy = evaluate_model(model, X_test, y_test)
                st.session_state.report_df  = report_df
                st.session_state.matrix_df  = matrix_df
                st.session_state.accuracy   = accuracy

                st.session_state.phase = "trained"
                st.rerun()

                # show metrics
                # display_evaluation_results(report_df, matrix_df, accuracy)

                # LLM summary of metrics
                st.subheader("Generating LLM Summary of the Model Performance Report")
                prompt = make_metrics_prompt(accuracy, report_df)
                expl = llm_explain(prompt)
                st.markdown("### Model Performance Explanation")
                st.write(expl)

    else:  # Upload & Explain
        st.info("Upload a pickled model and a CSV to explain")
        model_file = st.file_uploader("Model (.pkl)", type=["pkl"])
        data_file  = st.file_uploader("Data CSV",    type=["csv"])
        if model_file and data_file:
            raw = data_file.read().decode("utf-8")
            try:
                delim = csv.Sniffer().sniff(raw.splitlines()[0]).delimiter
            except csv.Error:
                delim = ","
            df = pd.read_csv(io.StringIO(raw), sep=delim)

            st.subheader("Data Preview")
            st.dataframe(df.head())

            target = st.selectbox("Select Target Column", df.columns)
            if st.button("Run Evaluation & Explain"):
                # st.session_state.phase = "trained"
                (X_train, X_test, y_train, y_test), _ = preprocess_data(df, target)

                model_file.seek(0)
                model = pickle.load(model_file)

                # store
                st.session_state.model   = model
                st.session_state.X_train = X_train
                st.session_state.X_test  = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test  = y_test

                report_df, matrix_df, accuracy = evaluate_model(model, X_test, y_test)
                st.session_state.report_df  = report_df
                st.session_state.matrix_df  = matrix_df
                st.session_state.accuracy   = accuracy

                st.session_state.phase = "trained"
                st.rerun()

                # display_evaluation_results(report_df, matrix_df, accuracy)

                # LLM summary of metrics
                st.subheader("Generating LLM Summary of the Model Performance Report")
                prompt = make_metrics_prompt(accuracy, report_df)
                expl = llm_explain(prompt)
                st.markdown("### Model Performance Explanation")
                st.write(expl)

# ─── PHASE 2: Explainability Charts ──────────────────────────────────────────────
if st.session_state.phase == "trained":
    report_df = st.session_state.report_df
    matrix_df = st.session_state.matrix_df
    accuracy = st.session_state.accuracy
    display_evaluation_results(report_df, matrix_df, accuracy, key_prefix="after_charts")
    st.markdown("## Select Explainability Charts")
    with st.form("charts_form"):
        st.session_state.charts = st.multiselect(
            "Charts to generate",
            ["SHAP", "LIME", "PDP", "ALE", "Counterfactual"],
            default=[]
        )
        # Only show feature selectors if PDP/ALE chosen
        if "PDP" in st.session_state.charts:
            st.session_state.pdp_features = st.multiselect(
                "Select PDP feature(s)",
                st.session_state.X_train.columns.tolist(),
                default=st.session_state.pdp_features if "pdp_features" in st.session_state else [
                    st.session_state.X_train.columns[0]]
            )
        if "ALE" in st.session_state.charts:
            st.session_state.ale_feature = st.selectbox(
                "Select ALE feature",
                st.session_state.X_train.columns.tolist(),
                index=st.session_state.X_train.columns.get_loc(st.session_state.ale_feature)
                if "ale_feature" in st.session_state else 0
            )
        submit = st.form_submit_button("Generate Charts")

    if submit:
        model   = st.session_state.model
        X_train = st.session_state.X_train
        X_test  = st.session_state.X_test
        y_train = st.session_state.y_train

        chart_data = {}

        for chart in st.session_state.charts:
            st.markdown(f"### {chart}")
            if chart == "SHAP":
                shap_explanation(model, X_train, X_test)
                expl = _shap.Explainer(model, X_train)
                shap_vals = expl(X_train[:100])
                vals = shap_vals.values
                if isinstance(vals, np.ndarray):
                    if vals.ndim == 3:
                        # vals shape: (n_samples, n_classes, n_features)
                        # pick positive class (index 1) if it exists, else class 0
                        class_idx = 1 if vals.shape[1] > 1 else 0
                        arr = vals[:, class_idx, :]
                    else:
                        # vals is already (n_samples, n_features)
                        arr = vals
                elif isinstance(vals, (list, tuple)):
                    # legacy SHAP: list of arrays [class0, class1]
                    arr = np.array(vals[1]) if len(vals) > 1 else np.array(vals[0])
                else:
                    # fallback: cast to array
                    arr = np.array(vals)

                mean_abs = np.abs(arr).mean(axis=0)
                top_idx = np.argsort(-mean_abs)[:5]
                tops = [(X_train.columns[i], float(mean_abs[i])) for i in top_idx]
                chart_data["SHAP"] = {"top_features": tops}


            elif chart == "LIME":
                lime_explanation(model, X_train, X_test)
                explr = LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                              class_names=['0', '1'], mode='classification')
                exp = explr.explain_instance(X_test.iloc[0].values, model.predict_proba)
                weights = exp.as_list()[:5]  # top 5
                chart_data["LIME"] = {"lime_weights": [(feat, wt) for feat, wt in weights]}

            elif chart == "PDP":
                features = st.session_state.get("pdp_features", [])
                if features:
                    pdp_explanation(model, X_train, features)
                    pdp_ranges = {
                        feat: (float(np.min(X_train[feat])), float(np.max(X_train[feat])))
                        for feat in features
                    }
                    chart_data["PDP"] = {"pdp_ranges": pdp_ranges}

                else:
                    st.warning("Please select at least one feature for PDP.")

            elif chart == "ALE":
                feature = st.session_state.get("ale_feature", None)
                if feature:
                    ale_explanation(model, X_train, feature)
                    chart_data["ALE"] = {
                        "ale_range": (
                            float(np.min(X_train[feature])),
                            float(np.max(X_train[feature]))
                        )
                    }

                else:
                    st.warning("Please select a feature for ALE.")

            elif chart == "Counterfactual":
                counterfactual_explanation(model, X_train, y_train, X_test)
                orig = st.session_state.get("cf_original", None)
                cf = st.session_state.get("cf_counterfactual", None)

                if orig is not None and cf is not None:
                    # Build side-by-side DataFrame
                    df_compare = pd.concat(
                        [orig.add_prefix("orig_"), cf.add_prefix("cf_")],
                        axis=1
                    )

                    target_col = st.session_state.y_train.name
                    df_compare = df_compare.drop(
                        columns=[f"cf_{target_col}"], errors="ignore"
                    )

                    # Highlight only the columns that changed
                    def highlight_changes(row):
                        styles = []
                        for col in df_compare.columns:
                            if col.startswith("cf_"):
                                orig_key = row["orig_" + col[3:]]
                                if orig_key in row.index and row[col] != row[orig_key]:
                                    styles.append("background-color: lightgreen")
                                else:
                                    styles.append("")

                            else:
                                styles.append("")
                        return styles


                    st.subheader("Counterfactual: Before vs. After")
                    st.dataframe(
                        df_compare.style.apply(highlight_changes, axis=1),
                        use_container_width=True
                    )

            # elif chart == "Counterfactual":
            #     counterfactual_explanation(model, X_train, y_train, X_test)
            #     cf_df = st.session_state.get("last_cf_df", None)
            #     if cf_df is not None:
            #         diffs = {}
            #         orig = X_test.iloc[0]
            #         cf = cf_df.iloc[0]
            #         for col in orig.index:
            #             if orig[col] != cf[col]:
            #                 diffs[col] = (float(orig[col]), float(cf[col]))
            #         chart_data["Counterfactual"] = {"cf_changes": diffs}

        # LLM explanation for the selected charts
        # if chart_data:
        #     st.subheader("Generating LLM summary of selected charts")
        #     prompt = make_charts_prompt(st.session_state.charts, chart_data)
        #     explanation = llm_explain(prompt)
        #     st.markdown("### Charts Interpretation (LLM)")
        #     st.write(explanation)


# ─── Reset App Button ───────────────────────────────────────────────────────────
if show_reset:
    st.markdown("---")
    if st.button("Reset & Start Over"):
        reset_app()
