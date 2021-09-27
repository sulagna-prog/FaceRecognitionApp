import numpy as np
import cv2 as cv
import os
import pandas as pd

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

features1 = np.load('features1.npy', allow_pickle=True)
labels1 = np.load('labels1.npy', allow_pickle=True)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
haar_cascade1 = cv.CascadeClassifier('haar_eye.xml')
DIR = r"C:\Users\Asmita\PycharmProjects\PythonProject2\Faces\train"
DIR1 = r"C:\Users\Asmita\PycharmProjects\PythonProject2\Faces\test"
people = []
traverse = []
data1=[]
data2=[]
data3=[]
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_train.yml')

for i in os.listdir(DIR):
 people.append(i)

#for i in os.listdir(DIR1):
    #traverse.append(i)
#print(traverse)
for j in os.listdir(DIR1):
    img_path = os.path.join(DIR1,j)
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in face_rect:
     faces_roi = gray[y:y + h, x:x + h]
     label, confidence = face_recognizer.predict(faces_roi)
     data1.append(j)
     data2.append(people[label])
     data3.append(confidence)
dict={'file name':data1,'image of':data2,'accuracy':data3}
df = pd.DataFrame(dict)
df.to_csv('file.csv')
print(df)
cv.waitKey(0)



