import streamlit as st
import cv2 as cv
from PIL import Image
import numpy as np
import os



features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

features1 = np.load('features1.npy', allow_pickle=True)
labels1 = np.load('labels1.npy', allow_pickle=True)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
haar_cascade1=cv.CascadeClassifier('haar_eye.xml')
DIR = r"Faces\train"
people = []

for i in os.listdir(DIR):
        people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_train.yml")
def detect_faces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Detect faces
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    name='Unknown'
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv.rectangle(img , (x, y), (x + w, y + h), (255, 255, 0), 2)
        label, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])
        print(id,confidence)
        print('face name = ', people[label], 'with a accuracy of', confidence)
        cv.putText(img , str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    return img
def main():
    """Face Recognition App"""
    st.markdown("<h1 style='text-align: center; color: black;'>WELCOME TO FACE RECOGNIZER !</h1>", unsafe_allow_html=True)
    #st.title("FACE RECOGNITION APP")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:#00a4fe ;padding:10px">
    <h2 style="color:black;text-align:center;font-size:30px">UPLOAD YOUR IMAGE AND LET US IDENTIFY YOU</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        result_img= detect_faces(our_image)
        st.image(result_img)


if __name__ == '__main__':
    main()
