import streamlit as st
from typing import Dict
from llama_cpp import Llama

# 1) Load the local LLM pipeline once and cache it
@st.cache_resource
def load_llm_local_gguf(model_path: str = "./models/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
    try:
        st.info(f"Loading quantized GGUF model from: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,           # You can tune this based on available RAM
            n_threads=8,          # Set according to your CPU cores
            n_batch=64,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load GGUF model from '{model_path}': {e}")
        return None

# 2) Generate natural language explanation
def llm_explain(prompt: str, max_tokens: int = 200) -> str:
    llm = load_llm_local_gguf()
    if not llm:
        return "LLM not available. Please check the GGUF model path or environment setup."

    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["</s>"]
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error generating explanation: {e}"



# 3) Build prompt for classification report
def make_metrics_prompt(accuracy: float, report: Dict) -> str:
    lines = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            lines.append(
                f"{label}: Precision={metrics.get('precision', 0):.2f}, "
                f"Recall={metrics.get('recall', 0):.2f}, F1={metrics.get('f1-score', 0):.2f}"
            )
    metrics_summary = "\n".join(lines)
    return (
        f"I trained a classification model with an accuracy of {accuracy:.2%}.\n"
        f"The per-class metrics are:\n{metrics_summary}\n\n"
        f"Please explain what these numbers mean about the model's performance in plain language."
    )


# 4) Build prompt for SHAP charts
def make_shap_prompt() -> str:
    return (
        "I have generated SHAP plots for model interpretability, including:\n"
        "- A Beeswarm plot\n"
        "- A Bar plot of mean SHAP values\n"
        "- A Waterfall plot for one prediction\n"
        "- A Force plot\n"
        "- A Decision plot\n\n"
        "Explain what each of these SHAP plots shows and how to interpret them in plain language."
    )
