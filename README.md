### 📊 AI-Powered Customer Churn Prediction & Retention Analytics Dashboard

An end-to-end Machine Learning application that predicts customer churn, analyzes risk levels, and generates AI-driven retention strategies using an interactive Streamlit dashboard.

---

### 🚀 Overview

This project leverages a trained XGBoost classification model to predict customer churn probability from uploaded datasets. It transforms raw customer data into actionable business insights through real-time predictions, risk segmentation, and explainable AI.

The dashboard is designed for business teams to understand who will churn, why they will churn, and how to retain them effectively.

---

### ✨ Key Features

- 📂 Upload CSV datasets for batch prediction
- 🤖 ML-based churn prediction using trained model
- 📊 Risk classification (Low, Medium, High, Very High)
- 💡 AI-powered retention recommendations
- 🧠 Explainable AI insights using SHAP
- 📈 Interactive visualizations with Plotly
- 👤 Customer-level prediction analysis
- 📥 Downloadable prediction reports

---

### 🧠 Explainability

The project integrates SHAP (SHapley Additive exPlanations) to provide:

- Global feature importance (what drives churn overall)
- Individual customer explanations (why a specific customer is at risk)

This ensures transparency and trust in model predictions.

---

### 🛠️ Tech Stack

- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Joblib

---

### 📊 Business Value

This system helps businesses:

- Identify at-risk customers early
- Reduce customer churn rate
- Improve customer retention strategies
- Make data-driven marketing decisions
- Understand key churn drivers

---

### 📁 Project Structure

├── app.py # Streamlit dashboard
├── model_churn.pkl # Trained ML model
├── label_encoders.pkl # Feature encoders
├── target_encoder.pkl # Target label encoder
├── Telco-Customer-Churn.csv # Sample dataset
└── README.md

---

### 🚀 How to Run

<Bash/>
pip install -r requirements.txt
streamlit run app.py

---

### 📌 Future Improvements

- Email alerts for high-risk customers
- Real-time API integration
- Advanced LLM-based retention suggestions
- Role-based login dashboard
- Power BI style executive view
