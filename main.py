import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/model.pkl")

st.title("Shield Insurance Premium Estimator")

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(3)

with row1[0]:
    age = st.number_input("Age", 1, 100, 30,step=1)
with row1[1]:
    gender = st.selectbox("Gender", ["Male", "Female"])
with row1[2]:
    region = st.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

with row2[0]:
    marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])
with row2[1]:
    physical_activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
with row2[2]:
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])

with row3[0]:
    dependants = st.number_input("Number of Dependants", 0, 10, 2)
with row3[1]:
    bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obesity", "Underweight"])
with row3[2]:
    smoking = st.selectbox("Smoking Status", ["No Smoking", "Occasional", "Regular"])

with row4[0]:
    employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
with row4[1]:
    income_level = st.number_input("Income Level", 0, 10, 2)
with row4[2]:
    income_lakhs = st.number_input("Income (Lakhs)", 0.0, 100.0, 12.0)

with row5[0]:
    medical_history = st.text_input("Medical History", "no disease")

with row5[1]:
    insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])

with st.form("premium_form"):







    submitted = st.form_submit_button("Predict Premium")

if submitted:
    try:
        # Load scaler
        scaler = joblib.load("model/scaler.pkl")
        
        # Insurance plan encoding
        insurance_plan_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
        encoded_insurance_plan = insurance_plan_map[insurance_plan]
        
        # Calculate normalized risk score
        risk_score_map = {
            'high blood pressure': 6,
            'no disease': 0,
            'heart disease': 8,
            'thyroid': 5,
            'diabetes': 6,
            'none': 0
        }
        medical_history_clean = medical_history.lower().strip()
        diseases = [d.strip() for d in medical_history_clean.split('&')]
        disease1 = diseases[0] if len(diseases) > 0 and diseases[0] else 'none'
        disease2 = diseases[1] if len(diseases) > 1 and diseases[1] else 'none'
        total_risk_score = risk_score_map.get(disease1, 0) + risk_score_map.get(disease2, 0)
        normalized_risk_score = total_risk_score / 14.0
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'age': [age],
            'number_of_dependants': [dependants],
            'income_level': [income_level],
            'income_lakhs': [income_lakhs],
            'insurance_plan': [encoded_insurance_plan],
            'normalized_risk_score': [normalized_risk_score],
            'gender_Male': [1 if gender == "Male" else 0],
            'region_Northwest': [1 if region == "Northwest" else 0],
            'region_Southeast': [1 if region == "Southeast" else 0],
            'region_Southwest': [1 if region == "Southwest" else 0],
            'marital_status_Unmarried': [1 if marital_status == "Unmarried" else 0],
            'physical_activity_Low': [1 if physical_activity == "Low" else 0],
            'physical_activity_Medium': [1 if physical_activity == "Medium" else 0],
            'stress_level_Low': [1 if stress_level == "Low" else 0],
            'stress_level_Medium': [1 if stress_level == "Medium" else 0],
            'bmi_category_Obesity': [1 if bmi == "Obesity" else 0],
            'bmi_category_Overweight': [1 if bmi == "Overweight" else 0],
            'bmi_category_Underweight': [1 if bmi == "Underweight" else 0],
            'smoking_status_Occasional': [1 if smoking == "Occasional" else 0],
            'smoking_status_Regular': [1 if smoking == "Regular" else 0],
            'employment_status_Salaried': [1 if employment == "Salaried" else 0],
            'employment_status_Self-Employed': [1 if employment == "Self-Employed" else 0]
        })
        
        # Scale numerical columns
        cols_to_scale = ['age', 'number_of_dependants', 'income_lakhs', 'income_level', 'insurance_plan']
        input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
        
        # Predict
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Annual Premium: ₹{prediction:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")