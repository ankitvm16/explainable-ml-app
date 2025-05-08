import streamlit as st
from llama_cpp import Llama
import pandas as pd

# 1) Load the local LLM pipeline once and cache it
@st.cache_resource
def load_llm_local_gguf(model_path: str = "./models/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
    try:
        st.info(f"Loading the LLM: Mistral-7b quantized GGUF model..")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_batch=64,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load GGUF model from '{model_path}': {e}")
        return None

# 2) Generate natural language explanation
def llm_explain(prompt: str, max_tokens: int = 512) -> str:
    llm = load_llm_local_gguf()
    if not llm:
        return "LLM not available. Please check the GGUF model path or environment setup."

    with st.expander("🔍 Prompt sent to LLM"):
        st.code(prompt, language="text")

    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1
        )
        text = output["choices"][0]["text"].strip()
        if not text:
            return "LLM returned no text. Consider adjusting the prompt or parameters."
        return text

    except Exception as e:
        return f"Error generating explanation: {e}"



# 3) Build prompt for classification report
def make_metrics_prompt(accuracy: float, report: pd.DataFrame) -> str:
    """
    Build a concise prompt using only the true class labels (excluding aggregates).
    """
    # Identify only the actual class labels by excluding meta-rows
    class_labels = [lab for lab in report.index
                    if lab not in ("accuracy", "macro avg", "weighted avg")]

    lines = []
    for label in class_labels:
        p = report.at[label, "precision"]
        r = report.at[label, "recall"]
        f = report.at[label, "f1-score"]
        lines.append(f"- {label}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")

    metrics_summary = "\n".join(lines)

    prompt = (
        "I trained a binary classification model.\n\n"
        f"Overall accuracy: {accuracy:.2%}\n"
        "Per-class performance:\n"
        f"{metrics_summary}\n\n"
        "In plain English, interpret what these numbers tell us about:\n"
        "1. How well the model detects each class.\n"
        "2. Any notable strengths or weaknesses.\n"
        "3. Its ability to distinguish between the two outcomes."
    )
    return prompt

# 4) Build prompt for XAI charts
def make_charts_prompt(selected_charts: list, chart_data: dict) -> str:

    sections = []
    for chart in selected_charts:
        data = chart_data.get(chart, {})
        if chart == "SHAP":
            # data: {'top_features': [('feat1', 0.45), ...]}
            lines = ["## SHAP Feature Importances"]
            for feat, val in data.get("top_features", []):
                lines.append(f"- {feat}: mean SHAP = {val:.3f}")
        elif chart == "LIME":
            # data: {'lime_weights': [('featA', 0.12), ...]}
            lines = ["## LIME Local Explanation"]
            for feat, w in data.get("lime_weights", []):
                lines.append(f"- {feat}: weight = {w:.3f}")
        elif chart == "PDP":
            # data: {'pdp_ranges': {feat: (min, max)}}
            lines = ["## Partial Dependence"]
            for feat, (mn, mx) in data.get("pdp_ranges", {}).items():
                lines.append(f"- {feat}: feature values range from {mn:.2f} to {mx:.2f}")

        elif chart == "ALE":
            # data: {'ale_range': (min, max)}
            lines = ["## Accumulated Local Effects"]
            mn, mx = data.get("ale_range", (None, None))
            if mn is not None:
                lines.append(f"- feature values range from {mn:.2f} to {mx:.2f}")

        elif chart == "Counterfactual":
            # data: {'cf_changes': {'featZ': (orig, cf), ...}}
            lines = ["## Counterfactual Explanation"]
            for feat, (orig, cf) in data.get("cf_changes", {}).items():
                lines.append(
                    f"- To flip prediction: {feat} change {orig:.3f} → {cf:.3f}"
                )
        else:
            continue
        sections.append("\n".join(lines))

    prompt = (
        "Below are the key data points for each explainability chart on my banking dataset.\n\n"
        + "\n\n".join(sections)
        + "\n\nFor each section, interpret what the data tells us about model behavior, "
          "highlight any non-intuitive findings, and suggest practical next steps."
    )
    return prompt
