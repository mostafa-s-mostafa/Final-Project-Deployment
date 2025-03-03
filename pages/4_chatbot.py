import streamlit as st
import pandas as pd
import imblearn
import joblib  

st.title("🤖 Churn Chatbot")

model = joblib.load("model_pipeline.pkl")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! I'll analyze readmission risk. Let’s start!"}
    ]
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}
if "batch_data" not in st.session_state:
    st.session_state.batch_data = None

required_inputs = [
    'race', 'gender', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'numchange', 'service_utilization', 'is_emergency_admission', 'admission_category',
    'discharge_to_home', 'discharge_care_level', 'admitted_from_emergency', 'age_midpoint',
    'diag_1_range', 'diag_2_range', 'diag_3_range', 'A1Cresult', 'max_glu_serum',
    'diabetesMed', 'Cluster'
]

if st.session_state.batch_data is not None:
    st.write("📊 Batch file detected! Processing all entries...")
    batch_df = st.session_state.batch_data.copy()
    batch_df = batch_df.astype({
        'time_in_hospital': 'int',
        'num_lab_procedures': 'int',
        'num_procedures': 'int',
        'num_medications': 'int',
        'number_outpatient': 'int',
        'number_emergency': 'int',
        'number_inpatient': 'int',
        'number_diagnoses': 'int',
        'age_midpoint': 'float'
    })
    batch_df["Readmission Prediction"] = model.predict(batch_df)
    batch_df["Readmission Probability"] = model.predict_proba(batch_df)[:, 1]
    st.dataframe(batch_df[["Readmission Prediction", "Readmission Probability"]])
    st.session_state.messages.append({"role": "bot", "content": "Batch predictions complete! ✅"})
    st.session_state.batch_data = None

else:
    missing_inputs = [col for col in required_inputs if col not in st.session_state.user_inputs]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if missing_inputs:
        user_input = st.chat_input(f"Please provide **{missing_inputs[0]}**:")
        if user_input:
            st.session_state.user_inputs[missing_inputs[0]] = user_input
            st.session_state.messages.append({"role": "user", "content": user_input})
            bot_response = f"Got it! Now, please provide **{missing_inputs[1] if len(missing_inputs)>1 else 'all remaining inputs'}**."
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            with st.chat_message("bot"):
                st.markdown(bot_response)
    else:
        input_data = pd.DataFrame([st.session_state.user_inputs])
        input_data = input_data.astype({
            'time_in_hospital': 'int',
            'num_lab_procedures': 'int',
            'num_procedures': 'int',
            'num_medications': 'int',
            'number_outpatient': 'int',
            'number_emergency': 'int',
            'number_inpatient': 'int',
            'numchange': 'int',
            'age_midpoint': 'float'
        })
        prediction = model.predict(input_data)
        readmission_probability = model.predict_proba(input_data)[0][1]
        
        if prediction[1] == 1:
            bot_response = (
                f"⚠️ **High readmission risk!** Probability: {readmission_probability:.2%}.\n\n"
                "**Recommended Actions:**\n"
                "- Schedule follow-up care 📅\n"
                "- Monitor glucose and A1C levels 🩸\n"
                "- Provide educational resources 📖"
            )
        else:
            bot_response = (
                f"✅ **Low readmission risk.** Probability: {readmission_probability:.2%}.\n\n"
                "🎉 Patient has a low risk of readmission. Maintain current care practices!"
            )
        
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        
        # Display all chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
