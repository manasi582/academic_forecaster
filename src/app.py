import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, preprocess_data

# Set page config
st.set_page_config(page_title="Academic Forecaster", layout="wide")

@st.cache_data
def load_and_prep_data(subject='mat'):
    df = load_data(subject)
    X, y = preprocess_data(df)
    return df, X.columns

def main():
    st.title("🎓 Academic Forecaster using Random Forest")
    st.markdown("Predict student final grade (G3) and pass/fail status based on various features.")
    
    # Sidebar for subject selection
    subject = st.sidebar.selectbox("Select Subject", ['Math (mat)', 'Portuguese (por)'])
    subject_code = 'mat' if 'Math' in subject else 'por'
    
    # Load data and model
    try:
        df, feature_columns = load_and_prep_data(subject_code)
        regressor = joblib.load(f'models/rf_regressor_{subject_code}.joblib')
        classifier = joblib.load(f'models/rf_classifier_{subject_code}.joblib')
    except FileNotFoundError:
        st.error("Models not found. Please run model training first.")
        return

    st.sidebar.header("Student Features")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Demographics & Social")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sex = st.selectbox("Sex", ['F', 'M'])
            age = st.slider("Age", 15, 22, 16)
            address = st.selectbox("Address", ['Urban', 'Rural'])
            famsize = st.selectbox("Family Size", ['<=3', '>3'])
            
        with col2:
            Pstatus = st.selectbox("Parent's Cohabitation", ['Together', 'Apart'])
            Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], format_func=lambda x: ["None", "Primary (4th)", "5th-9th", "Secondary", "Higher"][x])
            Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], format_func=lambda x: ["None", "Primary (4th)", "5th-9th", "Secondary", "Higher"][x])
            Mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
            
        with col3:
            Fjob = st.selectbox("Father's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
            reason = st.selectbox("Reason for School", ['home', 'reputation', 'course', 'other'])
            guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
        
        st.subheader("Academic & Habits")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            traveltime = st.selectbox("Travel Time", [1, 2, 3, 4], format_func=lambda x: ["<15 min", "15-30 min", "30 min-1h", ">1h"][x-1])
            studytime = st.selectbox("Study Time", [1, 2, 3, 4], format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x-1])
            failures = st.slider("Past Failures", 0, 4, 0)
            schoolsup = st.selectbox("Extra Educational Support", ['yes', 'no'])
            
        with col5:
            famsup = st.selectbox("Family Educational Support", ['yes', 'no'])
            paid = st.selectbox("Extra Paid Classes", ['yes', 'no'])
            activities = st.selectbox("Extra-curricular Activities", ['yes', 'no'])
            nursery = st.selectbox("Attended Nursery", ['yes', 'no'])
            
        with col6:
            higher = st.selectbox("Wants Higher Education", ['yes', 'no'])
            internet = st.selectbox("Internet Access", ['yes', 'no'])
            romantic = st.selectbox("Romantic Relationship", ['yes', 'no'])
            
        st.subheader("Health & Others")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            famrel = st.slider("Family Relationship Quality", 1, 5, 4)
            freetime = st.slider("Free Time", 1, 5, 3)
            
        with col8:
            goout = st.slider("Going Out with Friends", 1, 5, 3)
            Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
            
        with col9:
            Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
            health = st.slider("Current Health Status", 1, 5, 5)
            absences = st.number_input("Absences", 0, 93, 0)

        st.subheader("Previous Grades")
        g1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
        g2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
        
        submit = st.form_submit_button("Predict Performance")
        
    if submit:
        # Create dataframe from input
        input_data = {
            'school': 'GP', # Defaulting to GP as it's not in input but required? Or maybe I should add it.
            'sex': sex,
            'age': age,
            'address': 'U' if address == 'Urban' else 'R',
            'famsize': 'LE3' if famsize == '<=3' else 'GT3',
            'Pstatus': 'T' if Pstatus == 'Together' else 'A',
            'Medu': Medu,
            'Fedu': Fedu,
            'Mjob': Mjob,
            'Fjob': Fjob,
            'reason': reason,
            'guardian': guardian,
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': schoolsup,
            'famsup': famsup,
            'paid': paid,
            'activities': activities,
            'nursery': nursery,
            'higher': higher,
            'internet': internet,
            'romantic': romantic,
            'famrel': famrel,
            'freetime': freetime,
            'goout': goout,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': health,
            'absences': absences,
            'G1': g1,
            'G2': g2
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Preprocess input
        # We need to ensure columns match training data
        # We can reuse preprocess_data but we need to handle the dummy variables correctly
        # A better way is to append to original data (dummy) and then drop, or manually create columns
        
        # Let's use pd.get_dummies on the input_df, but we need to ensure all categories are present
        # Or we can align with feature_columns
        
        # Simplest way: Concatenate with a dummy row from original df to ensure structure, then drop it
        # But we have feature_columns.
        
        # Let's encode manually or use the same logic
        # Re-using preprocess_data might be tricky if it drops columns not present
        
        # Alternative: Use the same get_dummies but reindex
        input_df_encoded = pd.get_dummies(input_df)
        
        # Align columns
        # Create a dataframe with all 0s for missing columns
        for col in feature_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0
        
        # Reorder and select only feature columns
        input_df_final = input_df_encoded[feature_columns]
        
        # Predict
        prediction_reg = regressor.predict(input_df_final)[0]
        prediction_clf = classifier.predict(input_df_final)[0]
        
        st.divider()
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Predicted Final Grade (G3)", f"{prediction_reg:.2f} / 20")
            
        with col_res2:
            status = "PASS" if prediction_clf == 1 else "FAIL"
            color = "green" if status == "PASS" else "red"
            st.markdown(f"### Status: :{color}[{status}]")
            
        # Feature Importance Plot (Static from file or dynamic?)
        # Let's show the static one for the subject
        st.subheader("Feature Importance (Global)")
        st.image(f"results/plots/feature_importance_{subject_code}.png")

if __name__ == "__main__":
    main()
