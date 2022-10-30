import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
    

def show_exploration_page():
    st.write('**This page describes the main features of the dataset that is used to train the model. Since the dataset is' + \
                'very large, we will only use a sample of 10\% of the data. The following result is calculated in real-time' + \
                'using the `LLCP_agg_cleaned_10_percent.csv` dataset on this project GitHub. The following plots are interactive. ' + \
                'You can hover over the points to see the exact values.**')
    
    data = pd.read_csv('data/LLCP_agg_cleaned_10_percent.csv', low_memory=False)
    st.subheader('Dataset Preview')
    st.dataframe(data.head(10))
    st.write('The shape of the dataset is:', data.shape)

    st.subheader('Dataset Description')
    meanings = ['General Health', 'Asthma', 'Exercise in Past 30 Days',
                'Skin Cancer', 'BMI', 'Smoking Habit', 'Drinking Habit', 'Age Category', 'Heart Disease (target)', 'White', 'Black', 'Asian',
                'American Indian/Alaskan Native', 'Hispanic' , 'Other Race', 'Kidney Diease', 'C.O.P.D. emphysema or chronic bronchitis', 
                'Depressive Disorder', 'Diabetes', 'Sex at birth', 'Ever had Heart Attack', 'Cholesterol Test in Last 5 Years', 
                'Ever had Stroke', 'Ever Told Cholesterol Level is High', 'Combinaton of Race_1.0 to Race_6.0 columns']
    table = pd.DataFrame([data.columns, meanings, data.dtypes, data.nunique(), data.isna().sum() / len(data) * 100]).T
    table.columns = ['Column', 'Meaning', 'Data Type', '# Unique Values', '% of Missing Values']
    st.table(table)

    st.subheader('Notable  Categorical Features Distribution')
    st.markdown(
        '''
        * The dataset\'s target variable is very imbalanced. There are much more participants with no heart disease than participants with heart disease.
        * There are slightly more male participants than female participants.
        * The majority of participants are white.
        * Most age groups are represented equally.
        '''
    )
    col1, col2= st.columns(2)
    age_mappings = {1: '18 - 24', 2: '25 - 29', 3: '30 - 34', 4: '35 - 39', 5: '40 - 44', 6: '45 - 49', 7: '50 - 54', 8: '55 - 59', 9: '60 - 64', 10: '65 - 69', 11: '70 - 74', 12: '75 - 79', 13: '80+'}
    with col1:
        labels = ['No Heart Disease', 'Heart Disease']
        fig = px.pie(data, values=data['CVDCRHD4'].value_counts(), names=labels, title='Heart Disease Distribution')
        st.plotly_chart(fig)
        labels = [age_mappings[i] for i in range(1, 14)]
        fig = px.pie(data, values=data['_AGEG5YR'].value_counts(), names=labels, title='Age Distribution')
        st.plotly_chart(fig)
        General_Health = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
        fig = px.pie(data, values=data['GenHealth'].value_counts(), names=General_Health, title='General Health Distribution')
        st.plotly_chart(fig)
    with col2:
        labels = ['Male', 'Female']
        fig = px.pie(data, values=data['Sex'].value_counts(), names=labels, title='Sex distribution')
        st.plotly_chart(fig)
        labels = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other Race']
        fig = px.pie(data, values=data['Race'].value_counts(), names=labels, title='Race distribution')
        st.plotly_chart(fig)
        labels = ['Yes', 'No']
        fig = px.pie(data, values=data['EXERANY2'].value_counts(), names=labels, title='Exercise distribution')
        st.plotly_chart(fig)

    st.subheader('Features Correlation')
    col3, col4 = st.columns(2)
    with col3:
        vals = data.corr()['CVDCRHD4'].sort_values(ascending=False)
        fig = go.Figure(data=go.Heatmap(z=vals.values.reshape(1, -1), x=vals.index), layout=go.Layout(title='Correlation with Features and Heart Diease'))
        st.plotly_chart(fig)
        st.write('Not many features have strong features with the target variable. The top 4 features are heart attack, general health, age, and bronchitis.' + \
                    'Interestingly, these features are also the most important features ranked by the XGBoost model.')

    with col4:
        fig = go.Figure(data=go.Heatmap(z=data.corr().values, x=data.corr().columns, y=data.corr().columns), layout=go.Layout(title='Correlation Matrix'))
        st.plotly_chart(fig)
        st.write('A poor general health is somewhat correlated with other health complications like diabetes, stroke, and heart attack. We should ignore the ' + \
                    'one-hot encoded Race_x.0 columns. Being told that your cholesterol level is high is very strongly correlated with getting a cholesterol ' + \
                    'test in the last 5 years. Drinking alcohol has a negative correlation with general health.')
    
    st.subheader('Relationship Between the Target Variable and Other Notable Features')
    col5, col6 = st.columns(2)
    with col5:
        y_vals = {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'}
        fig = px.box(data, x='CVDCRHD4', y='GenHealth', color='CVDCRHD4', labels={'CVDCRHD4': 'Heart Disease', 'GenHealth': 'General Health'}, title='General Health vs. Heart Disease')
        fig.update_yaxes(tickvals=[1, 2, 3, 4, 5], ticktext=[y_vals[1], y_vals[2], y_vals[3], y_vals[4], y_vals[5]])
        st.plotly_chart(fig)
        st.write('There is a trend that people with heart disease tend to have worse general health. Interestingly, people with heart disease report their general health as good or fair and not many report it as poor.')
        temp_data = data.groupby(['GenHealth', '_AGEG5YR'])['CVDCRHD4'].sum().reset_index()
        temp_data = temp_data.rename(columns={'CVDCRHD4': 'Heart Disease Count', '_AGEG5YR': 'Age Group'})
        temp_data['Age Group'] = temp_data['Age Group'].map(age_mappings)
        temp_data['GenHealth'] = temp_data['GenHealth'].map(y_vals)
        fig = px.bar(temp_data, x='Age Group', y='Heart Disease Count', color='GenHealth', barmode='group', labels={'GenHealth': 'General Health', 'Heart Disease Count': 'Heart Disease Count'}, title='Heart Disease Count by Age Group and General Health')
        st.plotly_chart(fig)
        st.write('The distribution of general health doesn\'t change much with age. However, the number of people with heart disease increases with age. Again, we observe that people with heart disease is lesslikely to report their general health as poor.')

    with col6:
        fig = px.box(data, x='CVDCRHD4', y='_AGEG5YR', color='CVDCRHD4', labels={'CVDCRHD4': 'Heart Disease', '_AGEG5YR': 'Age'}, title='Age Group vs. Heart Disease')
        fig.update_yaxes(tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ticktext=[age_mappings[1], age_mappings[2], age_mappings[3], age_mappings[4], age_mappings[5], age_mappings[6], age_mappings[7], age_mappings[8], age_mappings[9], age_mappings[10], age_mappings[11], age_mappings[12], age_mappings[13]])
        st.plotly_chart(fig)
        st.write('There is a trend that people with heart disease tend to be older. Most people with heart disease are between 60 and 80 years old. There are also a few outliers that are young and have heart disease.')
        temp_data = data.groupby(['Sex', '_AGEG5YR'])['CVDCRHD4'].sum().reset_index()
        temp_data = temp_data.rename(columns={'CVDCRHD4': 'Heart Disease Count', '_AGEG5YR': 'Age Group'})
        temp_data['Sex'] = temp_data['Sex'].map({1: 'Male', 0: 'Female'})
        fig = px.bar(temp_data, x='Age Group', y='Heart Disease Count', color='Sex', title='Heart Disease Count by Age Group and Sex')
        fig.update_xaxes(title='Age Group', tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ticktext=[age_mappings[1], age_mappings[2], age_mappings[3], age_mappings[4], age_mappings[5], age_mappings[6], age_mappings[7], age_mappings[8], age_mappings[9], age_mappings[10], age_mappings[11], age_mappings[12], age_mappings[13]])
        st.plotly_chart(fig)
        st.write('There seem to be more elderly male participants with heart disease than elderly female participants with heart disease. But the age and sex combo does not seem shed much new insight on heart disease.')
