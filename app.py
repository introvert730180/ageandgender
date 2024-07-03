import streamlit as st
import cv2
import numpy as np

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
def predict_age_gender(image_path, age_net, gender_net):
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    return gender, age

def main():
    st.title('Age and Gender Prediction using Caffe Models')

    # Sidebar file upload widget
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
            gender, age = predict_age_gender(image_path, age_net, gender_net)

            # Display prediction
            st.success(f'Predicted Gender: {gender}, Predicted Age Range: {age}')

            # Remove temporary image file
            os.remove(image_path)

if __name__ == "__main__":
    main()
