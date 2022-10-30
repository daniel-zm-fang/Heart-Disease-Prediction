import streamlit as st
import pandas as pd

def show_about_page():
    st.subheader('What is the purpose of this app?')
    st.write('In the US, every 1 in 5 deaths is related to heart disease. Thus this app aims to help anyone predict whether they have heart disease or not for free with a single click. You can also explore the dataset to gain insights on the heart disease.')
    st.warning('This app is not meant to replace a doctor, but to help you gain insights on your heart health.')
    st.subheader('Where does the dataset come from?')
    st.write('The dataset is downloaded from the [CDC\'s BRFSS (Behavioral Risk Factor Surveillance System)](https://www.cdc.gov/brfss/annual_data/annual_data.htm). ' + \
                'Specifically, I downloaded the datasets from 2017 to 2021 and combined them into one dataset. After some data cleaning, the final dataset has 2127204 rows and 23 columns.')
    st.subheader('What is the model used and how accurate is it?')
    st.write('The model is the XGBClassifier model from the XGBoost library. The model is trained on the `LLCP_agg_cleaned.csv`' + \
                ' dataset on this project GitHub. The model is trained on 80% of the data and tested on 20% of the data. Here is its classification report on the test set (can also be found in `exploration_5.ipynb`):')
    st.table(
        pd.DataFrame([
            ['0.99', '0.81', '0.89', '401255'],
            ['0.21', '0.82', '0.33', '24186'],
            ['', '', '0.81', '425441'],
            ['0.60', '0.81', '0.61', '425441'],
            ['0.94', '0.81', '0.86', '425441']],
            columns=['Precision', 'Recall', 'F1-Score', 'Support'],
            index=['No Heart Disease', 'Heart Disease', 'Accuracy', 'Macro Avg', 'Weighted Avg']))
    st.write('* The model\'s overall accuracy is 81%. The model\'s accuracy is low for the positive class (only 21%). This is however a **worthy trade-off** because the model\'s recall for the positive class is high (82%)' + \
                ' the most important goal of the model is to identify as many people with heart disease as possible. Since heart diease is a serious matter, we don\'t want any false negatives' + \
                ' (people with heart disease but the model predicts they don\'t have heart disease). This is why the model\'s recall for the positive class is more important than its accuracy' + \
                ' for the positive class.')
    st.write('* Also, the model outputs a probability that a person might have heart disease instead of a binary Yes/No answer. Even if the model predicts a ' + \
                'false positive, that means that the person have some risk factors for heart disease. This serves as a warning for the person to take action to prevent heart disease in the future.')
    st.subheader('What do I need to know about the prediction results?')
    st.write('* The model is trained on a dataset from the US. The model might not be as accurate for people outside the US. Also, all the data is gathered from phone calls. There might be ' + \
                'some bias in the participants\' answers. For example, some people might not be comfortable sharing their health information over the phone, so they might not answer the questions ' + \
                'truthfully or they might not answer at all. Also, some people might not be able to answer the questions truthfully because they don\'t know the answers.')
    st.write('* 75% of the participants are white. If your race is Hispanic, American Indian/Alaska Native, this model might not be as accurate for you because the model is not trained on enough data from these groups.')
    st.write('But other than that, let\'s break down the 2 possible cases:')
    st.write('1. **Case 1: The model predicts that a person doesn\'t have heart disease.** The model is 99% accurate for this case. This means that the person has a 99% chance of not having heart disease. ' + \
                'So I would feel confident in saying that the person don\'t have heart disease.')
    st.write('2. **Case 2: The model predicts that a person has heart disease.** The model is 21% accurate for this case. This means that the person has a 21% chance of having heart disease. ' + \
                'Recall that the accuracy for the positive class is low because the model is being conservative in predicting heart disease. Even if there are little evidence, the model will still ' + \
                'predict that the person has heart disease. So I would be cautious in this case. I would recommend the person to go to a doctor to get a more accurate diagnosis.')            
