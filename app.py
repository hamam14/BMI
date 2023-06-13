import pandas as pd
import pickle
import streamlit as st
import numpy as np

# load the model from disk
model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
model1 = pickle.load(open('minmaxxx.pkl', 'rb'))

# Define the prediction function
def predict(Height, Weight, BMI):
#     Membuat DataFrame dengan data input
    input_data = pd.DataFrame([[Height, Weight, BMI]], columns=['Height', 'Weight', 'BMI'])
    
    # Melakukan prediksi dengan model yang telah diload
    preprocess = model1.transform(input_data)
    prediction = model.predict(preprocess)
    if prediction == 0:
        return "Underweight (Kurus BMI kurang dari 18,5)"
    elif prediction == 1:
        return "Normal weight (Berat badan Normal BMI antara 18,5 hingga 24,9)"
    elif prediction == 2:
        return "Pre-Obesity (Berat badan BMI antara 24 hingga 27)"
    elif prediction == 3:
        return "Obese Class I (Obesitas Kelas I BMI antara 30 hingga 34,9)"
    elif prediction == 4:
        return "Obese Class II (Obesitas Kelas II BMI antara 35 hingga 39,9)"
    elif prediction == 5:
        return "Obese Class III (Obesitas Kelas III BMI 40 atau lebih)"
      
    return prediction


st.set_page_config(
    page_title="Prediksi BMI",
    page_icon="👋",
)

st.header('Prediksi BMI/Indeks Massa Tubuh')

tab1, tab2, tab3, tab4 = st.tabs(["Deskripsi Data", "Tab Pre - Processing", "Tab Modeling", "Tab Implementasi"])

Height = st.number_input("Masukkan Tinggi Badan Anda", min_value=0, max_value=300, value=10)
Weight = st.number_input("Masukkan Berat Badan Anda", min_value=0, max_value=300, value=10)
BMI = round(Weight / ((Height / 100) ** 2), 2)
st.write("BMI:", BMI)

if st.button('Prediksi'):
    prediksi = predict(Height, Weight, BMI)
    st.success(f'Anda termasuk {prediksi}')
