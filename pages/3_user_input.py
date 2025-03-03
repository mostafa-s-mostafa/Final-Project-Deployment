import streamlit as st
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")

model = load_model()


# Diagnosis range options (reused for diag 1-3)
diagnosis_ranges = [
    '240-279 (Endocrine/Metabolic)', '630-679 (Pregnancy)',
    '001-139 (Infectious/Parasitic)', '140-239 (Neoplasms)',
    '390-459 (Circulatory)', '460-519 (Respiratory)',
    '800-999 (Injury/Poisoning)', '680-709 (Skin)', 
    '710-739 (Musculoskeletal)', '520-579 (Digestive)',
    'V01-V91 (Health Status Factors)', '780-799 (Symptoms)',
    '580-629 (Genitourinary)', '290-319 (Mental Disorders)',
    '320-389 (Nervous System)', '280-289 (Blood Disorders)',
    'Other', '740-759 (Congenital)', 'E800-E999 (External Causes)'
]

medication_options = ['No', 'Steady', 'Up', 'Down']

def create_diagnosis_section(diagnosis_number):
    col = st.columns(1)[0]
    with col:
        st.subheader(f"Diagnosis {diagnosis_number}")
        numeric_value = st.number_input(f"diag_{diagnosis_number}_numeric_value", min_value=0, max_value=999, value=0)
        range_val = st.selectbox(f"diag_{diagnosis_number}_range", diagnosis_ranges)
        diag_type = st.radio(f"Diagnosis {diagnosis_number} Type", ['V', 'E', 'Numeric'])
        
    return {
        f'diag_{diagnosis_number}_numeric_value': numeric_value,
        f'diag_{diagnosis_number}_range': range_val,
        f'diag_{diagnosis_number}_is_V': 1 if diag_type == 'V' else 0,
        f'diag_{diagnosis_number}_is_E': 1 if diag_type == 'E' else 0,
        f'diag_{diagnosis_number}_is_numeric': 1 if diag_type == 'Numeric' else 0
    }

def main():
    st.title("Medical Data Entry Form")
    
        # CSV Upload for Batch Predictions
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        predictions = model.predict(df)
        prediction_probs = model.predict_proba(df)
        df['Prediction'] = ['Readmitted' if p == 1 else 'Not Readmitted' for p in predictions]
        df['Probability_Readmission'] = prediction_probs[:, 1]
        st.write("### Batch Prediction Results")
        st.dataframe(df)

    with st.form("medical_form"):
        # ========== Personal Information ==========
        st.header("Patient Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other', np.nan])
        with col2:
            gender = st.selectbox("Gender", ['Female', 'Male'])
        with col3:
            age_midpoint = st.slider("Age Midpoint", 5, 95, 65)
        
        # ========== Medical History ==========
        st.header("Medical History")
        
        # Medications
        st.subheader("Medications")
        med_col1, med_col2 = st.columns(2)
        with med_col1:
            metformin = st.selectbox("Metformin", medication_options)
            repaglinide = st.selectbox("Repaglinide", medication_options)
            nateglinide = st.selectbox("Nateglinide", medication_options)
            chlorpropamide = st.selectbox("Chlorpropamide", medication_options)
            glimepiride = st.selectbox("Glimepiride", medication_options)
            acetohexamide = st.selectbox("Acetohexamide", ['No', 'Steady'])
            
        with med_col2:
            glipizide = st.selectbox("Glipizide", medication_options)
            glyburide = st.selectbox("Glyburide", medication_options)
            tolbutamide = st.selectbox("Tolbutamide", ['No', 'Steady'])
            pioglitazone = st.selectbox("Pioglitazone", medication_options)
            rosiglitazone = st.selectbox("Rosiglitazone", medication_options)
            acarbose = st.selectbox("Acarbose", medication_options)
        
        # More Medications
        st.subheader("Additional Medications")
        med_col3, med_col4 = st.columns(2)
        with med_col3:
            miglitol = st.selectbox("Miglitol", medication_options)
            troglitazone = st.selectbox("Troglitazone", ['No', 'Steady'])
            tolazamide = st.selectbox("Tolazamide", ['No', 'Steady', 'Up'])
            insulin = st.selectbox("Insulin", medication_options)
            
        with med_col4:
            glyburide_metformin = st.selectbox("Glyburide-Metformin", medication_options)
            glipizide_metformin = st.selectbox("Glipizide-Metformin", ['No', 'Steady'])
            glimepiride_pioglitazone = st.selectbox("Glimepiride-Pioglitazone", ['No', 'Steady'])
            metformin_rosiglitazone = st.selectbox("Metformin-Rosiglitazone", ['No', 'Steady'])
            metformin_pioglitazone = st.selectbox("Metformin-Pioglitazone", ['No', 'Steady'])

        # ========== Lab & Procedures ==========
        st.header("Lab Results & Procedures")
        lab_col1, lab_col2, lab_col3 = st.columns(3)
        with lab_col1:
            time_in_hospital = st.number_input("Time in Hospital (days)", 1, 14, 4)
            num_lab_procedures = st.number_input("Lab Procedures Count", 1, 99, 43)
            num_procedures = st.number_input("Procedures Count", 0, 6, 1)
            
        with lab_col2:
            num_medications = st.number_input("Medications Count", 1, 60, 16)
            number_diagnoses = st.number_input("Diagnoses Count", 1, 16, 7)
            max_glu_serum = st.radio("Max Glucose Serum", [0, 1])
            
        with lab_col3:
            numchange = st.number_input("Number of Changes", 0, 4, 0)
            service_utilization = st.number_input("Service Utilization", 0, 31, 1)
            A1Cresult = st.radio("A1C Result", [0, 1])


        # ========== Diagnosis Information ==========
        st.header("Diagnosis Details")
        diag_data = {}
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        with diag_col1:
            diag_data.update(create_diagnosis_section(1))
        with diag_col2:
            diag_data.update(create_diagnosis_section(2))
        with diag_col3:
            diag_data.update(create_diagnosis_section(3))

        # ========== Admission Details ==========
        st.header("Admission Information")
        adm_col1, adm_col2, adm_col3 = st.columns(3)
        with adm_col1:
            admission_category = st.selectbox("Admission Category", 
                ['other', 'emergency', 'urgent', 'elective', 'newborn', 'trauma'])
            
        with adm_col2:
            discharge_care_level = st.selectbox("Discharge Care Level", 
                ['25', 'home', 'rehab', 'transfer', 'short_hospital_stay', 'AMA', '4',
                 'long_hospital_stay', 'death', '13', '12', '16', '17', 'hospice', '9', 
                 '20', '15', '24', '28', '19', '27'])
            
        with adm_col3:
            referral_source = st.selectbox("Referral Source", 
                ['physician_referral', 'ER', 'clinic_referral', 'transfer_hospital',
                 'transfer_healthcare_facility', '20', 'HMO_referral', '17', 'court_law',
                 'other', '14', '10', '22', '11', '25', '13'])

        # ========== Service Utilization ==========
        st.header("Service Utilization")
        util_col1, util_col2, util_col3 = st.columns(3)
        with util_col1:
            number_outpatient = st.number_input("Outpatient Visits", 0, 10, 0)
        with util_col2:
            number_emergency = st.number_input("Emergency Visits", 0, 19, 0)
        with util_col3:
            number_inpatient = st.number_input("Inpatient Visits", 0, 14, 0)

        # ========== Additional Fields ==========
        st.header("Additional Information")
        add_col1, add_col2 = st.columns(2)
        with add_col1:
            change = st.radio("Change", ['No', 'Ch'])
            diabetesMed = st.radio("Diabetes Medication", ['No', 'Yes'])
        with add_col2:
            is_emergency_admission = st.radio("Emergency Admission", [0, 1])
            discharge_to_home = st.radio("Discharge to Home", [0, 1])
            admitted_from_emergency = st.radio("Admitted from Emergency", [0, 1])
        
        Cluster = st.selectbox("Cluster", [0, 1, 2, 3])

        # Submit button
        submitted = st.form_submit_button("Submit Full Record")
        
        if submitted:
            # Compile all data
            patient_data = {
                'race': race,
                'gender': gender,
                'age_midpoint': age_midpoint,
                'metformin': metformin,
                'repaglinide': repaglinide,
                'nateglinide': nateglinide,
                'chlorpropamide': chlorpropamide,
                'glimepiride': glimepiride,
                'acetohexamide': acetohexamide,
                'glipizide': glipizide,
                'glyburide': glyburide,
                'tolbutamide': tolbutamide,
                'pioglitazone': pioglitazone,
                'rosiglitazone': rosiglitazone,
                'acarbose': acarbose,
                'miglitol': miglitol,
                'troglitazone': troglitazone,
                'tolazamide': tolazamide,
                'insulin': insulin,
                'glyburide-metformin': glyburide_metformin,
                'glipizide-metformin': glipizide_metformin,
                'glimepiride-pioglitazone': glimepiride_pioglitazone,
                'metformin-rosiglitazone': metformin_rosiglitazone,
                'metformin-pioglitazone': metformin_pioglitazone,
                'change': change,
                'diabetesMed': diabetesMed,
                **diag_data,
                'admission_category': admission_category,
                'discharge_care_level': discharge_care_level,
                'referral_source': referral_source,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'max_glu_serum': max_glu_serum,
                'A1Cresult': A1Cresult,
                'numchange': numchange,
                'service_utilization': service_utilization,
                'is_emergency_admission': is_emergency_admission,
                'discharge_to_home': discharge_to_home,
                'admitted_from_emergency': admitted_from_emergency,
                'Cluster': Cluster
            }
            
            input_df = pd.DataFrame([patient_data])
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            if prediction == 1:
                message = (
                    f"🔮 Prediction: **Readmitted**\n\n"
                    f"⚠️ **High readmission risk!** Probability: {prediction_proba[1]:.2%}.\n\n"
                    "- Schedule follow-up care 📅\n"
                    "- Monitor glucose and A1C levels 🩸\n"
                    "- Provide educational resources 📖"
                )
            else:
                message = (
                    f"🔮 Prediction: **Not Readmitted**\n\n"
                    f"✅ **Low readmission risk!** Probability: {prediction_proba[0]:.2%}.\n\n"
                    "🎉 Patient has a low risk of readmission. Maintain current care practices!"
                )

            st.success(message)


            st.write(f"🧪 Probability: {prediction_proba[1]:.2%} chance of readmission")
            st.dataframe(input_df)

if __name__ == "__main__":
    main()