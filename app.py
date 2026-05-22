import streamlit as st
import numpy as np
import pickle

# Load Model
with open('retention_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎯 Customer Retention Engine")
st.markdown("A tunnel-optimized interface for live machine learning predictions.")

st.divider()

# Dropdowns and text boxes instead of glitchy sliders
st.header("📋 Step 1: Customer Profile")
tenure = st.number_input("Tenure (Months with Company)", min_value=1, max_value=72, value=12)

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract = contract_map[st.selectbox("Contract Type", list(contract_map.keys()))]

st.header("🛠️ Step 2: Actionable Levers")
monthly_bill = st.number_input("Monthly Bill ($)", min_value=15.0, max_value=120.0, value=75.0)

tech_support_map = {"No": 0, "Yes": 2} 
tech_support = tech_support_map[st.radio("Offer Free Tech Support?", list(tech_support_map.keys()))]

# Prediction Math
input_data = np.zeros((1, 19))
input_data[0, 2] = tenure         
input_data[0, 14] = contract       
input_data[0, 17] = monthly_bill   
input_data[0, 11] = tech_support   

churn_probability = model.predict_proba(input_data)[0][1]
risk_percentage = churn_probability * 100

st.divider()
st.header("📊 Step 3: Real-Time Risk Assessment")
if risk_percentage > 50:
    st.error(f"🔴 HIGH RISK: This customer has a {risk_percentage:.1f}% chance of leaving!")
else:
    st.success(f"🟢 LOW RISK: This customer has a {risk_percentage:.1f}% chance of staying loyal.")
