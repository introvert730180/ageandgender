import streamlit as st
import cv2
import numpy as np

# Function to preprocess and predict age and gender
def predict_age_gender(image):
    # Preprocess image
    # Perform prediction using age and gender models

st.title('Age and Gender Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    age_pred, gender_pred = predict_age_gender(image)

    if age_pred and gender_pred:
        st.write(f'Predicted Age: {age_pred}')
        st.write(f'Predicted Gender: {gender_pred}')
    else:
        st.write("Prediction failed. Please try another image.")
