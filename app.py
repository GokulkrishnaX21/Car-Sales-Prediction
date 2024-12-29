import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title ("Sales Prediction App")

st.divider()

st.write("This app is for getting sale estimator")


age = st.number_input("Enter the age", min_value=18, max_value=90, value=40, step=1)
salary = st.number_input("Enter the Salary", min_value=1000, max_value=99999999, step = 5000, value =30000)
networth = st.number_input("Enter the net worth", min_value=0, max_value=99999999, step=20000)

x = [age,salary,networth]

calculatebutton = st.button("Calculate")

st.divider()

if calculatebutton:
    
    st.balloons()
    
    x_2 = np.array(x)
    
    x_array = scaler.transform([x_2])
    
    prediction = model.predict(x_array)
    
    st.write(f"Prediction is {prediction[0]}")
    st.write("Advice cars in the similar values")
    
else:
    st.write("Please enter the values and press the calculate button")    