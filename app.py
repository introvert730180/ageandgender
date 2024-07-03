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

# Function to capture image from webcam and return OpenCV frame
def capture_frame(cap):
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return None

def main():
    st.title('Live Age and Gender Prediction')

    age_net, gender_net = initialize_caffe_models()

    # Create OpenCV VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened correctly
    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return

    # Display the live camera feed
    stframe = st.empty()
    button_capture = st.button("Capture Image")

    while True:
        ret, frame = cap.read()

        # Display the live camera feed
        stframe.image(frame, channels='BGR', use_column_width=True)

        # Capture image and predict on button press
        if button_capture:
            # Predict age and gender on the captured frame
            frame = predict_age_gender(frame, age_net, gender_net)

            # Convert the frame to RGB from BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the resulting frame in Streamlit
            stframe.image(frame_rgb, channels='RGB', use_column_width=True)

            # Show reupload button after prediction
     # Continue capturing frames

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
            cap.release()

if __name__ == "__main__":
    main()
