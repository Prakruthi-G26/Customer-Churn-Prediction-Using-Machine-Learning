import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL & ENCODERS
# =========================
model = joblib.load("model_churn.pkl")
le_dict = joblib.load("label_encoders.pkl")
le_target = joblib.load("target_encoder.pkl")

# =========================
# SHAP EXPLAINER SETUP
# =========================
explainer = shap.TreeExplainer(model)

# =========================
# TITLE
# =========================
st.title("📊 AI Customer Retention Analytics Dashboard")

st.markdown("""
Upload a customer dataset to predict churn risk,
analyze customer behavior,
and generate AI-powered retention recommendations.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📈 Dashboard Information")

st.sidebar.metric("Model ROC-AUC", "81.6%")
st.sidebar.metric("Prediction System", "Batch Analytics")
st.sidebar.metric("Model Type", "XGBoost")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload Customer CSV File",
    type=["csv"]
)

# =========================
# PROCESS FILE
# =========================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Dataset")

    st.dataframe(df.head())

    # =========================
    # ENCODE CATEGORICAL COLUMNS
    # =========================
    for col, le in le_dict.items():

        if col in df.columns:

            classes = list(le.classes_)

            df[col] = df[col].map(
                lambda x: x if x in classes else classes[0]
            )

            df[col] = le.transform(df[col])

    # =========================
    # PREDICTIONS
    # =========================
    probabilities = model.predict_proba(df)[:, 1]

    predictions = model.predict(df)

    churn_labels = le_target.inverse_transform(predictions)

    # =========================
    # ADD RESULTS
    # =========================
    df["Churn Prediction"] = churn_labels

    df["Churn Probability"] = probabilities

    # =========================
    # RISK LEVELS
    # =========================
    def risk_level(prob):

        if prob >= 0.80:
            return "Very High"

        elif prob >= 0.60:
            return "High"

        elif prob >= 0.40:
            return "Medium"

        else:
            return "Low"

    df["Risk Level"] = df["Churn Probability"].apply(risk_level)

    # =========================
    # RETENTION RECOMMENDATIONS
    # =========================
    def recommendation(prob):

        if prob >= 0.80:
            return "Offer premium discount and priority support"

        elif prob >= 0.60:
            return "Send personalized retention offers"

        elif prob >= 0.40:
            return "Monitor customer engagement"

        else:
            return "Maintain regular engagement"

    df["Recommendation"] = df[
        "Churn Probability"
    ].apply(recommendation)

    # =========================
    # KPI SECTION
    # =========================
    st.subheader("📈 Key Business Metrics")

    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df)

    high_risk = len(
        df[df["Risk Level"].isin(["Very High", "High"])]
    )

    avg_risk = df["Churn Probability"].mean()

    predicted_churn = len(
        df[df["Churn Prediction"] == "Yes"]
    )

    with col1:
        st.metric(
            "Total Customers",
            total_customers
        )

    with col2:
        st.metric(
            "Predicted Churn Customers",
            predicted_churn
        )

    with col3:
        st.metric(
            "High Risk Customers",
            high_risk
        )

    with col4:
        st.metric(
            "Average Churn Risk",
            f"{avg_risk:.2%}"
        )
    # =========================
    # 🔍 EXPLAINABLE AI - GLOBAL INSIGHTS
    # =========================
    st.subheader("🧠 Explainable AI Insights (Feature Importance)")

    shap_values = explainer(df.drop(columns=["Churn Prediction", "Churn Probability", "Risk Level", "Recommendation"], errors="ignore"))

    fig_shap = px.bar(
        x=np.abs(shap_values.values).mean(axis=0),
        y=df.drop(columns=["Churn Prediction", "Churn Probability", "Risk Level", "Recommendation"], errors="ignore").columns,
        orientation='h',
        title="Top Factors Influencing Churn"
    )

    st.plotly_chart(fig_shap, use_container_width=True)
    st.subheader("💡 AI Retention Recommendations")

    recommendation_df = df[
        df["Risk Level"].isin(["Very High", "High"])
    ][
        [
            "Churn Prediction",
            "Churn Probability",
            "Risk Level",
            "Recommendation"
        ]
    ]

    st.dataframe(recommendation_df)
    # =========================
    # CHARTS
    # =========================
    st.subheader("📊 Churn Analytics")

    chart_col1, chart_col2 = st.columns(2)

    # Risk Distribution
    with chart_col1:

        risk_counts = df["Risk Level"].value_counts()

        fig1 = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Customer Risk Distribution"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # Churn Prediction Distribution
    with chart_col2:

        churn_counts = df[
            "Churn Prediction"
        ].value_counts()

        fig2 = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            title="Predicted Churn Distribution"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # =========================
    # HIGH RISK CUSTOMERS
    # =========================
    st.subheader("⚠️ High Risk Customers")

    high_risk_df = df[
        df["Risk Level"].isin(["Very High", "High"])
    ]

    st.dataframe(high_risk_df)

    # =========================
    # DOWNLOAD RESULTS
    # =========================
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="customer_churn_predictions.csv",
        mime="text/csv"
    )
    # =========================
    # 👤 CUSTOMER DEEP DIVE (WHY THIS CUSTOMER CHURNS)
    # =========================
    st.subheader("👤 Customer-Level AI Explanation")

    selected_index = st.selectbox(
        "Select Customer Index for Explanation",
        df.index
    )

    customer_data = df.drop(
        columns=["Churn Prediction", "Churn Probability", "Risk Level", "Recommendation"],
        errors="ignore"
    ).iloc[[selected_index]]

    shap_value = explainer(customer_data)

    st.write("### Prediction Breakdown")

    
    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap_value[0],
        show=False
    )

    st.pyplot(fig)
# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("""
### 🚀 Platform Features

✅ Batch churn prediction  
✅ Customer risk analysis  
✅ AI-powered retention recommendations  
✅ Interactive analytics dashboard  
✅ Downloadable prediction reports  
✅ Business intelligence insights
""")