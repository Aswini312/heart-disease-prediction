import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and label encoder
model_filename = 'heart_disease_type_model.pkl'
with open(model_filename, 'rb') as file:
    model, label_encoder = pickle.load(file)

st.title('Heart Disease Type Prediction')

# Collect user input
age = st.number_input('Age', min_value=1, max_value=120, value=30)
chest_pain_type = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
resting_bp = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar', [0, 1])
resting_ecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
num_major_vessels = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, value=0)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[age, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, num_major_vessels]],
                          columns=['Age', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar',
                                   'Resting Electrocardiographic Results', 'Maximum Heart Rate Achieved',
                                   'Number of Major Vessels Colored by Fluoroscopy'])

# Display the input data for verification
st.subheader('Input Data')
st.write(input_data)

# Make a prediction when the user clicks the button
if st.button('Predict'):
    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)
    st.subheader(f'The predicted heart disease type is: **{predicted_label[0]}**')
