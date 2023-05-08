import face_recognition
import sys
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_input = cv2.VideoCapture(0)                         #Creating an object of VideoCapture method , camera module will take continous video input


Sashank = face_recognition.load_image_file(r"D:\Digital Image Processing\Lab\PROJECT\venv\STUDENTS\sashank.jpeg")
Sashank_train = face_recognition.face_encodings(Sashank)[0]

JaiShanker = face_recognition.load_image_file(r"D:\Digital Image Processing\Lab\PROJECT\venv\STUDENTS\jaishanker.jpeg")
JaiShanker_train = face_recognition.face_encodings(JaiShanker)[0]

Modi = face_recognition.load_image_file(r"D:\Digital Image Processing\Lab\PROJECT\venv\STUDENTS\modi.jpeg")
Modi_train = face_recognition.face_encodings(Modi)[0]

Nirmala = face_recognition.load_image_file(r"D:\Digital Image Processing\Lab\PROJECT\venv\STUDENTS\tagore.jpeg")
Nirmala_train = face_recognition.face_encodings(Nirmala)[0]

# In the code above, we first load the image file using load_image_file() function. We then extract the information
# out of an image using face_encodings() method, this method returns an 128-dimensional face encodings



students_of_class =[
	Sashank_train,
	JaiShanker_train,
	Modi_train,
	Nirmala_train
]

# /\ We are appending all the encodings onto an list


students_names=[
	"Sashank ",
	"Jai Shanker",
	"Narendra Modi",
	"Nirmala Sitharaman"
]

# /\ We are appending  the NAMES according to the order of face_encodings list, onto an list

students = students_names.copy()

face_location=[]
face_encodings=[]
face_names=[]
flag=True

# /\ The above lines of code is necessary for storing and computation

current_time= datetime.now()
current_date= current_time.strftime("%d-%m-%Y")

# /\ Storing current date in current_date variable
wr=open(current_date+'.csv','w+',newline='')
lnwr = csv.writer(wr)

# /\ Opening a linewriter to write the data to the csv file


while True:                                                                              # infinite loop
	_,frame = video_input.read()                                                         # Extracting the video data
	read_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)                                 # Decreasing the size for easy recognition
	rgb_read_frame = read_frame[:,:,::-1]                                                # Converting for BGR to RGB,because the face_recognition uses RGB format
	if flag:
		face_location = face_recognition.face_locations(rgb_read_frame)                  # to detect a face in the frame
		face_encodings = face_recognition.face_encodings(rgb_read_frame,face_location)   # Store the data from the detected face in frame
		face_names=[]
		for face_seen in face_encodings:                                                 #
			found= face_recognition.compare_faces(students_of_class,face_seen)           # Mathematical model for comparing the face in frame with students_of_class
			name=""
			face_feature = face_recognition.face_distance(students_of_class,face_seen)   # returns  a euclidean distance for each comparison face.
			face_match=np.argmin(face_feature)                                           # argmin gives the best fit ( best probability of known faces)
			if found[face_match]:
				name=students_names[face_match]                                          # Stores the name of the best fit face

			face_names.append(name)                                                      # Entering the name to a list
			if name in students_names:
				if name in students:
					students.remove(name)                                                # Removing the name from database, to reduce discrepancy
					print(name +" is present.")                                          # If a same face appears again , the data will get overwritten.
					tn= current_time.strftime("%H-%M-%S")                                # Stores the current time
					lnwr.writerow([name,tn])                                             # Write to the CSV file

	cv2.imshow("Attendence Moniter",frame)
	if cv2.waitKey(1) & 0xFF == ord('z'):                                                # Press the key 'z' to save and exit.
		break


video_input.release()
cv2.destroyAllWindows()
wr.close()
