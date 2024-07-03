import streamlit as st
import cv2
import numpy as np

# Set up OpenCV VideoCapture object
cap = cv2.VideoCapture(0)
cap.set(3, 480) # set width
cap.set(4, 640) # set height

# Define model and lists for age and gender prediction
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

# Initialize Caffe models for age and gender prediction
def initialize_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('data/deploy_age.prototxt', 'data/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('data/deploy_gender.prototxt', 'data/gender_net.caffemodel')
    return age_net, gender_net

# Function to predict age and gender from the webcam feed
def predict_age_gender(image, age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Get face region
        face_img = image[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Overlay text with gender and age prediction
        overlay_text = f'Gender: {gender}, Age: {age}'
        cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image

def main():
    st.title('Live Age and Gender Prediction')

    # Initialize Caffe models
    age_net, gender_net = initialize_caffe_models()

    # Main loop for capturing frames and displaying predictions
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error('Error: Unable to capture frame.')
            break

        # Predict age and gender on the frame
        frame = predict_age_gender(frame, age_net, gender_net)

        # Convert the frame to RGB (Streamlit requires RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        st.image(frame_rgb, channels='RGB', use_column_width=True)

        # Check if the 'q' key is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()

if __name__ == '__main__':
    main()
