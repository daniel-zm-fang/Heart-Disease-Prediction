import streamlit as st
import numpy as np
import dill


xgb_clf = dill.load(open('models/xgb_model.pkl', 'rb'))
std_scaler = dill.load(open('models/std_scaler.pkl', 'rb'))

def show_prediction_page():
    st.write('**Please fill in the following information and click the "Predict" button to get the prediction.**')
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        bmi = st.slider('Body Mass Index (BMI, kg/m**2)', 10, 40, 20)
        gen_health = st.select_slider('How would you rate your general health?', ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'], 'Good')
        gen_health_mappings = {'Poor':4, 'Fair':3, 'Good':2, 'Very Good':1, 'Excellent':0}
        gen_health = gen_health_mappings[gen_health]
        age_cat = st.selectbox('What is your age category?', ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])
        age_mappings = {'18-24':1, '25-29':2, '30-34':3, '35-39':4, '40-44':5, '45-49':6, '50-54':7, '55-59':8, '60-64':9, '65-69':10, '70-74':11, '75-79':12, '80 or older':13}
        age_cat = age_mappings[age_cat]
        race_cat = ['American Indian/Alaskan Native', 'Asian', 'Black', 'Hispanic', 'Other', 'White']
        race = st.selectbox('What is your ethnicity?', race_cat, index=5)
        races = [1 if race_cat[i] == race else 0 for i in range(len(race_cat))]
        smoking = st.selectbox('Do you smoke? If yes, please specify.', ['now smokes every day', 'now smokes some days', 'Former smoker', 'Never smoked'], index=3)
        smoking_mappings = {'now smokes every day':4, 'now smokes some days':3, 'Former smoker':2, 'Never smoked':1}
        smoking = smoking_mappings[smoking]
        diabetes = st.radio('Have you ever been told by a doctor that you have diabetes?', ['Yes', 'No, borderline diabetes', 'Yes (only during pregnancy)', 'No'], index=3)
        diabetes_mappings = {'Yes':2, 'No, borderline diabetes':1, 'Yes (only during pregnancy)':0, 'No':0}
        diabetes = diabetes_mappings[diabetes]

    with col2:
        sex = st.radio('What is your sex', ['Male', 'Female'], index=0)
        sex = 1 if sex == 'Male' else 0
        cholesterol = st.radio('Did you check your cholesterol level in the past 5 years?', ['Yes', 'No, but I have checked it before', 'Never'], index=2)
        cholesterol_mappings = {'Yes':1, 'No, but I have checked it before':2, 'Never':3}
        cholesterol = cholesterol_mappings[cholesterol]
        alcohol = st.checkbox('Have you had at least one drink of alcohol in the past 30 days?', value=False)
        bronchitis = st.checkbox('Have you been diagnosed with C.O.P.D. (chronic obstructive pulmonary disease), emphysema or chronic bronchitis?', value=False)
        phys_active = st.checkbox('During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?', value=False)
        asthma = st.checkbox('Have you ever been told by a doctor that you have asthma?', value=False)
        kidney_disease = st.checkbox('Have you ever been told by a doctor that you have kidney disease? (not including kidney stones, bladder infection or incontinence)', value=False)
        skin_cancer = st.checkbox('Have you ever been told by a doctor that you have skin cancer?', value=False)
        depression = st.checkbox('Have you ever been told by a doctor that you have depressive disorder (including depression, major depression, dysthymia, or minor depression)?', value=False)
        heart_attack = st.checkbox('Have you ever had a heart attack?', value=False)
        stroke = st.checkbox('Have you ever had a stroke?', value=False)
        told_chol_high = st.checkbox('Have you ever been told by a doctor that your cholesterol level is high?', value=False)
        arthritis = st.checkbox('Have you ever been told by a doctor that you have arthritis?', value=False)

    ok = st.button('Predict', type='primary')
    if ok:
        args_lst = [gen_health, asthma, phys_active, skin_cancer, bmi, smoking, alcohol, age_cat] + races + [kidney_disease, bronchitis, depression, diabetes, sex, heart_attack, cholesterol, stroke, told_chol_high, arthritis]
        print(len(args_lst))
        X = np.array(args_lst).reshape(1, -1)
        X = std_scaler.transform(X)
        y_pred = round(100 * xgb_clf.predict_proba(X)[0][1], 2)
        if y_pred > 90:
            st.error(f'Probability of heart diease: {y_pred}%.\nYou are very likely at risk of developing heart disease. Take care of yourself!')
        elif y_pred > 50:
            st.warning(f'Probability of heart diease: {y_pred}%.\nYou have some risk of developing heart disease. Take care of yourself!')
        elif y_pred > 30:
            st.warning(f'Probability of heart diease: {y_pred}%.\nYou have a healthy heart, but there your health can be improved a bit.')
        else:
            st.balloons()
            st.success(f'Probability of heart diease: {y_pred}%.\nYou have a very healthy heart, well done!')
