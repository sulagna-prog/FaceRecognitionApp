import os
import numpy as np
import cv2 as cv
DIR=r"Faces\train"
haar_cascade=cv.CascadeClassifier('haar_face.xml')
haar_cascade1=cv.CascadeClassifier('haar_eye.xml')
people=[]
for i in os.listdir(DIR):
    people.append(i)
print('people',people)
features=[]
labels=[]
features1=[]
labels1=[]
def create_train():
 for person in people:
     path=os.path.join(DIR,person)
     label=people.index(person)

     for img in os.listdir(path):
      img_path=os.path.join(path,img)
      img_array=cv.imread(img_path)
      #resize = cv.resize(img_array, (300, 300), interpolation=cv.INTER_CUBIC)
      gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

      faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)
      eye_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)

      for(x,y,w,h) in faces_rect:
        faces_roi=gray[y:y+h,x:x+w]
        features.append(faces_roi)
        labels.append(label)

      for (x, y, w, h) in eye_rect:
          eyes_roi = gray[y:y + h, x:x + w]
          features1.append(eyes_roi)
          labels1.append(label)


create_train()
print('training done----------')


features=np.array(features,dtype='object')
labels=np.array(labels)

features1=np.array(features,dtype='object')
labels1=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

eye_recognizer=cv.face.LBPHFaceRecognizer_create()



# train the recognizer on thr features list and the label list

face_recognizer.train(features,labels)
face_recognizer.save('face_train.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)

eye_recognizer.train(features1,labels1)
eye_recognizer.save('eye_train.yml')
np.save('features1.npy',features)
np.save('labels1.npy',labels)

