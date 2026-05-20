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
    page_title="AI Customer Retention Platform",
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
# TITLE
# =========================
st.title("🚀 AI-Powered Customer Retention & Churn Analytics Platform")

st.markdown("""
Predict customer churn probability, analyze risk levels,
and generate intelligent retention recommendations.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📈 Model Information")

st.sidebar.metric("Model ROC-AUC", "81.6%")
st.sidebar.metric("Model Type", "XGBoost")
st.sidebar.metric("Prediction System", "Real-Time")

st.sidebar.markdown("---")

st.sidebar.info("""
This platform predicts customer churn risk
and suggests retention actions using AI-driven analytics.
""")

# =========================
# CUSTOMER INPUT SECTION
# =========================
st.subheader("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

# =========================
# BILLING SECTION
# =========================
st.subheader("💳 Billing Information")

col4, col5 = st.columns(2)

with col4:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col5:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=1.0
    )

# Estimated total charges
total_charges = tenure * monthly_charges

st.write(f"### Estimated Total Charges: ${total_charges:.2f}")

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🔍 Predict Customer Churn"):

    # =========================
    # INPUT DATAFRAME
    # =========================
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    X_input = pd.DataFrame([input_dict])

    # =========================
    # ENCODING
    # =========================
    for col, le in le_dict.items():

        if col in X_input.columns:

            classes = list(le.classes_)

            X_input[col] = X_input[col].map(
                lambda x: x if x in classes else classes[0]
            )

            X_input[col] = le.transform(X_input[col])

    # =========================
    # PREDICTION
    # =========================
    probability = model.predict_proba(X_input)[0, 1]

    prediction = model.predict(X_input)[0]

    churn_label = le_target.inverse_transform(
        [prediction]
    )[0]

    # =========================
    # RISK LEVEL
    # =========================
    if probability >= 0.80:
        risk = "🔴 Very High Risk"

    elif probability >= 0.60:
        risk = "🟠 High Risk"

    elif probability >= 0.40:
        risk = "🟡 Medium Risk"

    else:
        risk = "🟢 Low Risk"

    # =========================
    # RETENTION RECOMMENDATION
    # =========================
    if probability >= 0.80:

        recommendation = """
        - Offer premium retention discount
        - Assign dedicated customer support executive
        - Provide loyalty benefits
        - Escalate to retention team immediately
        """

    elif probability >= 0.60:

        recommendation = """
        - Send personalized engagement offers
        - Provide service upgrade options
        - Monitor customer activity closely
        """

    elif probability >= 0.40:

        recommendation = """
        - Send customer satisfaction survey
        - Provide occasional promotional offers
        """

    else:

        recommendation = """
        - Maintain regular engagement
        - Continue standard customer support
        """

    # =========================
    # RESULTS SECTION
    # =========================
    st.markdown("---")

    st.subheader("📊 Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Churn Prediction",
            churn_label
        )

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with result_col3:
        st.metric(
            "Risk Level",
            risk
        )

    # =========================
    # PROGRESS BAR
    # =========================
    st.write("### Churn Risk Score")

    st.progress(float(probability))

    # =========================
    # RECOMMENDATION
    # =========================
    st.subheader("💡 AI Retention Recommendation")

    st.info(recommendation)

    # =========================
    # PIE CHART
    # =========================
    chart_df = pd.DataFrame({
        "Category": ["Churn Risk", "Retention Probability"],
        "Value": [probability, 1 - probability]
    })

    fig = px.pie(
        chart_df,
        names="Category",
        values="Value",
        title="Customer Churn Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # SHAP EXPLAINABILITY
    # =========================
    st.subheader("🧠 Explainable AI Insights")

    try:
        explainer = shap.Explainer(model)

        shap_values = explainer(X_input)

        st.write("""
        The graph below explains which customer features
        contributed most toward churn prediction.
        """)

        fig_shap, ax = plt.subplots()

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        st.pyplot(fig_shap)

    except Exception as e:

        st.warning(
            "SHAP explainability currently unavailable."
        )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("""
### 📌 Platform Features

✅ Real-time churn prediction  
✅ AI-powered retention recommendations  
✅ Risk scoring system  
✅ Interactive analytics dashboard  
✅ Explainable AI insights  
✅ Business-focused customer intelligence
""")