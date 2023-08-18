# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 08:52:45 2023

@author: kunde
"""

import pickle
import streamlit as st

#loading the saved model

spam_model=pickle.load(open('Spam_mail_prediction.sav','rb'))

# Load the feature extraction object
feature_extraction = pickle.load(open('feature_extraction.sav', 'rb'))
#creating side bar for navigation

st.title('Spam mail Prediction Using ML')
    
    # getting the input data from the user
    
Message = st.text_input('Paste the Mail Here:')
    
    
    # code for Prediction
result = ''
    
    # creating a button for Prediction
    
if st.button('SPAM Prediction Result'):
        # Preprocess the input message
        input_mail = [Message]
        input_mail_features = feature_extraction.transform(input_mail)
        
        # Make the prediction using the processed message
        spam_prediction = spam_model.predict(input_mail_features)
        
        if (spam_prediction[0] == 1):
          result = 'The Mail is NOT SPAM'
        else:
          result = 'The Message is SPAM'
        
st.success(result)

