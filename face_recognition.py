import numpy as np
import cv2 as cv
import os



features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

features1 = np.load('features1.npy', allow_pickle=True)
labels1 = np.load('labels1.npy', allow_pickle=True)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
haar_cascade1=cv.CascadeClassifier('haar_eye.xml')
DIR = r"C:\Users\Asmita\PycharmProjects\face 3\Faces\train"
people = []

for i in os.listdir(DIR):
        people.append(i)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_train.yml')
img=cv.imread(r'C:\Users\Asmita\PycharmProjects\face 3\Faces\test\IMG_20210830_112627.jpg')

#img = cv.imread(filename)
resize =cv.resize(img, (300, 300), interpolation=cv.INTER_CUBIC)
gray =cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
cv.imshow('person', gray)

    # detect the face in the image

face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
eye_rect= haar_cascade1.detectMultiScale(gray,1.1,4)


for (x, y, w, h) in face_rect:
        faces_roi = gray[y:y + h, x:x + h]
        label, confidence = face_recognizer.predict(faces_roi)
        print('face name = ', people[label], 'with a accuracy of', confidence)
        cv.putText(resize, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=2)
        cv.rectangle(resize, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

eye_c=-60
for (x, y, w, h) in eye_rect:
        eye_c+=60
        cv.putText(resize,"EYE", (x, y-5), cv.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), thickness=1)
        cv.rectangle(resize, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
        cv.imshow('detected', resize)

cv.waitKey(0)



