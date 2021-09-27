import subprocess

subprocess.run("faces_train.py & face_recognition.py", shell=True)
#exec(open("faces_train.py").read())
#exec(open("face_recognition.py").read())