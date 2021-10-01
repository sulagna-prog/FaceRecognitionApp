# Face Recognition
*This is the consolidated github repository for the IT Project completion by* "Group III - 25" *on the topic*<br>"Face Recognition in Picture."<br><br>

## Contents of the Repository
[Faces](https://github.com/sulagna-prog/FaceRecognitionApp/tree/master/Faces) :   Contains the training and test dataset folders<br>
[app.py](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/app.py) :  The final web app<br>
[data.py](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/data.py) :  Creates a file that stores accuracy rates<br>
[eye_train.yml](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/eye_train.yml) :  Stores information of Detected eyes from image<br>
[face_train.yml](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/face_train.yml) :  Stores information of Detected faces from image<br>
[faces_train.py](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/faces_train.py) :  Trains the model<br>
[features.npy](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/features.npy) :  Features of face stored in numpy array<br>
[features1.npy](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/features1.npy) :  Features of eye stored in numpy array<br>
[file.csv](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/file.csv) :  File created to store accuracy rates<br>
[haar_eye.xml](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/haar_eye.xml) :  Haar Cascade Classifier for eye detection<br>
[haar_face.xml](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/haar_face.xml) :  Haar Cascade Classifier for face detection<br>
[labels.npy](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/labels.npy) :  Store names of detected faces<br>
[label1.npy](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/labels1.npy) :  Store label for detected eye<br>
[requirements.txt](https://github.com/sulagna-prog/FaceRecognitionApp/blob/master/requirements.txt) :  Required packages<br><br>

## Requirements
<ul>
  <li>Python 3.3+</li>
  <li>Windows (prefered and tested on)</li>
</ul>
<br>

## Libraries to be installed
Commands to be followed to install the required Python module.
### Numpy:
<pre>pip install numpy</pre>
### Pandas:
<pre>pip install pandas</pre>
### OpenCV:
<pre>pip install opencv-contrib-python</pre>
### Pillow:
<pre>pip install pillow</pre>
### Streamlit:
<pre>pip install streamlit</pre>

<br><br>
## How to run the code
<ol>
  <li> Download the zip file from GitHub</li><br>
  <li> Extract the .zip file</li><br>
  <li> Open the extracted folder in any python environment</li><br>
  <li> Run <code>faces_train.py</code> using command:</li>
      <br><pre>python faces_train.py</pre><br>
  <li> The training process begins and will take a few seconds.</li><br>
  <li> Run <code>app.py</code> using streamlit command:</li>
      <br><pre>streamlit run app.py</pre>
      <ul>
        <li>Model is deployed on the local server.</li></ul><br>
  <li> Click on the Local url</li><br>
      <ul>
        <li>This will open your project in the default browser.</li>
      </ul><br>
  <li> Test the model</li><br>
      <ul>
        <li>Upload images to test the model.</li>
      </ul><br>
  <li> <code>data.py</code> creates a <code>.csv</code> file to store the accuracy rates.</li><br>
  <li> <code>file.csv</code> is successfully created.</li><br>
</ol>

## Demo
[Click here to watch the tutorial](https://www.youtube.com/watch?v=wQkhJoUCK8g)

<!---## References
[[1] Face Recognition System Face Detection.pdf](http://www.pace.ac.in/documents/ece/FACE%20RECOGNITION%20SYSTEM%20WITH%20FACE%20DETECTION.pdf)<br>
[Real-Time Secure System for Detection and Recognition the Face of Criminals.pdf](https://drive.google.com/drive/u/0/folders/1T0cJKKVl-Nrecp5g9SE6nkXkGd0jMiqL)<br>







