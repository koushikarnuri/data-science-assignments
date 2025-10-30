#!/usr/bin/env python
# coding: utf-8

# In[36]:


import streamlit as st
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



# In[37]:


model=pickle.load(open('logi.pkl','rb'))


# In[43]:


def user_input_varibles():
    Pregnancies=st.sidebar.number_input('Enter the number of Pregnancies')
    Glucose=st.sidebar.number_input('Enter the Level of Glucose')
    BloodPressure=st.sidebar.number_input('Enter the Value of BloodPressure')
    Insulin=st.sidebar.number_input('Enter the Value of Insulin')
    BMI=st.sidebar.number_input('Enter the Value of BMI')
    DiabetesPedigreeFunction=st.sidebar.number_input('Enter the Value of DiabetesPedigreeFunction')
    Age=st.sidebar.number_input('Enter the Age')
    data= {'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,'Insulin':Insulin,'BMI':BMI,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'Age':Age}
    features=pd.DataFrame(data,index=[0])
    return features
df= user_input_varibles()
pred_prob=model.predict_proba(df)
pred=model.predict(df)
button= st.button('pred')
if button==True:
    st.subheader('Prediction')
    st.write('Positive' if pred_prob[0][1]>=0.5 else 'Negative')
    st.subheader('Prediction_Probability')
    st.write(pred_prob)


# In[ ]:




