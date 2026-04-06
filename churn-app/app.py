import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load model and columns
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction Dashboard")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", [
    "Month-to-month", "One year", "Two year"
])

internet_service = st.selectbox("Internet Service", [
    "DSL", "Fiber optic", "No"
])

payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])

online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# -----------------------------
# Prepare Input Data
# -----------------------------
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,

    # Contract Encoding
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    # Internet Service
    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,

    # Payment Method
    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card" else 0,

    # Services
    "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
    "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
    "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Align with training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📢 Prediction Result")

    if prob > 0.4:
        st.error(f"⚠️ Customer is likely to CHURN ({prob:.2f})")
    else:
        st.success(f"✅ Customer will STAY ({prob:.2f})")

    # -----------------------------
    # Probability Chart
    # -----------------------------
    st.subheader("📊 Churn Probability")

    fig1, ax1 = plt.subplots()
    ax1.bar(["Stay", "Churn"], [1 - prob, prob])
    ax1.set_ylabel("Probability")
    ax1.set_title("Prediction Confidence")

    st.pyplot(fig1)

    # -----------------------------
    # Feature Importance Chart
    # -----------------------------
    st.subheader("🔥 Top Feature Importance")

    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    ax2.barh(feat_df["Feature"], feat_df["Importance"])
    ax2.invert_yaxis()
    ax2.set_title("Top 10 Important Features")

    st.pyplot(fig2)