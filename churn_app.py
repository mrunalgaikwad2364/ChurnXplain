"""
churn_app.py

Streamlit UI for the churn model.

Prediction and SHAP-explanation logic live in model_utils.py, shared
with api.py (the FastAPI service) — the UI and the API are guaranteed
to agree on every prediction because they call the same function.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from model_utils import (
    RAW_CUSTOMER_FIELDS,
    describe_feature,
    get_recommendations,
    load_model_and_schema,
    predict_with_explanation,
)

HISTORY_PATH = "prediction_history.csv"

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1100px; }

    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.75rem;
        border: 1px solid rgba(250,250,250,0.10);
    }
    .result-card.high-risk {
        background: linear-gradient(135deg, rgba(220,38,38,0.16), rgba(220,38,38,0.04));
        border-color: rgba(220,38,38,0.35);
    }
    .result-card.low-risk {
        background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(34,197,94,0.04));
        border-color: rgba(34,197,94,0.35);
    }
    .result-top-row { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
    .result-label { font-size: 0.85rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7; }
    .result-value { font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
    .result-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .badge-high { background: rgba(220,38,38,0.22); color: #fca5a5; }
    .badge-low  { background: rgba(34,197,94,0.22); color: #86efac; }
    .result-caption { opacity: 0.65; font-size: 0.85rem; margin-top: 0.5rem; }

    .section-heading {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 2rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(250,250,250,0.10);
    }

    .factor-card {
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(250,250,250,0.08);
        background: rgba(250,250,250,0.02);
        height: 100%;
    }
    .factor-card h4 { margin-top: 0; margin-bottom: 0.75rem; font-size: 1rem; }
    .factor-row { display: flex; justify-content: space-between; gap: 0.75rem; padding: 0.35rem 0; font-size: 0.92rem; }
    .factor-row .impact { opacity: 0.6; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .impact-up { color: #fca5a5; }
    .impact-down { color: #86efac; }

    .reco-card {
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(250,250,250,0.08);
        background: rgba(250,250,250,0.02);
        margin-bottom: 1.5rem;
    }
</style>
"""

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_model_and_schema():
    return load_model_and_schema()


def render_factor_card(title, icon, factors, direction):
    impact_class = "impact-up" if direction == "up" else "impact-down"
    rows_html = ""
    shown = 0
    for name, val, raw in factors:
        desc = describe_feature(name, raw)
        if not desc:
            continue
        sign = "+" if val > 0 else ""
        rows_html += (
            f'<div class="factor-row"><span>{desc}</span>'
            f'<span class="impact {impact_class}">{sign}{val:.2f}</span></div>'
        )
        shown += 1
    if shown == 0:
        rows_html = '<div class="factor-row"><span>None for this customer</span></div>'
    st.markdown(
        f'<div class="factor-card"><h4>{icon} {title}</h4>{rows_html}</div>',
        unsafe_allow_html=True,
    )


model, schema = get_model_and_schema()

# --- Sidebar inputs ---
st.sidebar.header("🧾 Customer Information")
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.checkbox("Has Credit Card", True)
is_active = st.sidebar.checkbox("Active Member", True)
salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 60000.0)
country = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])

raw_fields = {
    "CreditScore": credit_score,
    "Geography": country,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": int(has_card),
    "IsActiveMember": int(is_active),
    "EstimatedSalary": salary,
}

st.title("🏦 Customer Churn Prediction System")
st.markdown("Fill in the customer details on the left panel, then click **Predict Churn**.")

if st.sidebar.button("🚀 Predict Churn"):
    result = predict_with_explanation(raw_fields, model, schema)
    prob = result.probability * 100

    # --- Result card ---
    risk_class = "high-risk" if result.prediction else "low-risk"
    badge_class = "badge-high" if result.prediction else "badge-low"
    badge_text = "HIGH CHURN RISK" if result.prediction else "LOW CHURN RISK"
    st.markdown(
        f"""
        <div class="result-card {risk_class}">
            <div class="result-top-row">
                <div>
                    <div class="result-label">Churn Probability</div>
                    <div class="result-value">{prob:.1f}%</div>
                </div>
                <div class="result-badge {badge_class}">{badge_text}</div>
            </div>
            <div class="result-caption">
                Decision threshold: {result.threshold * 100:.1f}% — chosen to give the best precision
                while still catching at least 75% of actual churners on validation data
                (not a fixed 50/50 cutoff).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Recommendations ---
    st.markdown('<div class="section-heading">💡 Recommendations</div>', unsafe_allow_html=True)
    recos = get_recommendations(result.risk_factors)
    if recos:
        reco_html = "".join(f"<li>{r}</li>" for r in recos)
        st.markdown(f'<div class="reco-card"><ul>{reco_html}</ul></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="reco-card">No specific action needed — continue standard engagement.</div>',
            unsafe_allow_html=True,
        )

    # --- Top risk / protective factors ---
    st.markdown(
        '<div class="section-heading">📌 Top Factors Increasing &amp; Reducing Churn Risk</div>',
        unsafe_allow_html=True,
    )
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        render_factor_card("Top Factors Increasing Churn Risk", "⬆️", result.risk_factors, "up")
    with fcol2:
        render_factor_card("Top Factors Reducing Churn Risk", "⬇️", result.protective_factors, "down")

    # --- SHAP Explainability (waterfall) ---
    st.markdown('<div class="section-heading">🔬 SHAP Explainability</div>', unsafe_allow_html=True)
    st.caption("How each feature pushes this prediction away from the average customer.")
    plt.figure()
    shap.plots.waterfall(result.shap_explanation, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()

    # --- Download report ---
    st.markdown('<div class="section-heading">📅 Download Prediction Report</div>', unsafe_allow_html=True)
    report_data = {
        **raw_fields,
        "Prediction": "Churn" if result.prediction else "Stay",
        "Probability (%)": round(prob, 2),
        "Threshold Used (%)": round(result.threshold * 100, 2),
    }
    report_df = pd.DataFrame([report_data])
    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Report as CSV",
        data=csv,
        file_name="churn_prediction_report.csv",
        mime="text/csv",
    )

    if os.path.exists(HISTORY_PATH):
        existing = pd.read_csv(HISTORY_PATH)
        updated = pd.concat([existing, report_df], ignore_index=True)
    else:
        updated = report_df
    updated.to_csv(HISTORY_PATH, index=False)
    st.success("📄 Prediction saved to history!")

st.sidebar.markdown("---")
if st.sidebar.button("View Prediction History"):
    if os.path.exists(HISTORY_PATH):
        st.subheader("📂 Prediction History")
        st.dataframe(pd.read_csv(HISTORY_PATH).dropna())
    else:
        st.info("No prediction history available yet.")