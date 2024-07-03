import streamlit as st
import cv2
import numpy as np
import os

# Load pre-trained models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize OpenCV DNN models
def initialize_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('./data/deploy_age.prototxt', './data/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('./data/deploy_gender.prototxt', './data/gender_net.caffemodel')
    return age_net, gender_net

# Function to predict age and gender from an image
def predict_age_gender(image, age_net, gender_net):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Get Face 
        face_img = image[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Display prediction
        overlay_text = f'Gender: {gender}, Age: {age}'
        cv2.putText(image, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def main():
    st.title('Age and Gender Prediction using Caffe Models')

    # Sidebar file upload widget
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    stframe = st.empty()
    if uploaded_file is not None:
        # Display the uploaded image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Perform prediction if button is clicked
        if st.button('Predict'):
            # Initialize models
            age_net, gender_net = initialize_caffe_models()

            # Save the uploaded image locally
            image_path = 'temp_image.jpg'
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Predict age and gender
            frame = predict_age_gender(image, age_net, gender_net)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels='RGB', use_column_width=True)

            # Display prediction

            # Remove temporary image file
            os.remove(image_path)

if __name__ == "__main__":
    main()
