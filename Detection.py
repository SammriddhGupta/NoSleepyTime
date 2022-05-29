#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install cmake')
get_ipython().system('pip install dlib')
get_ipython().system('pip install imutils')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scipy')
get_ipython().system('pip install pygame')
get_ipython().system('pip install datetime')


# In[2]:


# Importing OpenCV Library for basic image processing functions
import cv2

# Numpy for array related functions
import numpy as np

# SciPy provides us with the module scipy.spatial, which has functions for working with spatial data.
from scipy.spatial import distance

# Dlib for deep learning based Modules and face landmark detection
import dlib

''' imtuils consists of a series of convenience functions to make basic image processing functions such as 
    translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, 
    and much more easier with OpenCV, we use the face_utils module for basic operations of these conversions'''
from imutils import face_utils

'''Pygame is a set of Python modules designed to create games and multimedia programs. 
   Pygame comes with a module called mixer, with which we will play an audio file.'''
import pygame

# The datetime module supplies classes for manipulating dates and times.
import datetime

'''the time module provides various time-related functions. 
   We use ctime([secs]) to convert a time expressed in seconds since the epoch to a string of a form'''
import time

#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio_alert.wav')

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calcualte the distance between the upper and lower lip to be considered a yawn
def cal_yawn(shape): 
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
  
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
  
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
  
    distance_yawn = distance.euclidean(top_mean,low_mean)
    return distance_yawn

yawn_thresh = 25
ptime = 0

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

#calculating the EAR ratio 
def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    # Checking if it is blinked
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0

while True:
    # Read each frame and flip it, and convert to grayscale
    ret, frame = cap.read()
    if(ret == False):
        break;

    # finding the frames per second (fps) and displaying it on screen
    ctime = time.time() 
    fps= int(1/(ctime-ptime))
    ptime = ctime
    cv2.putText(frame,f'FPS:{fps}',(frame.shape[1]-120,frame.shape[0]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # grab the current timestamp and draw it on the frame
    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Detect facial points through detector function
    faces = detector(gray)
    face_frame = frame.copy()

    # detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
      
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmark = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmark)
        
        # Detecting/Marking the lower and upper lip #
        lip = landmarks[48:60]
        
        # we can draw contour lines around the mouth, and similarly for the eyes, but I have commented it out for cleanliness
        # cv2.drawContours(frame,[lip],-1,(0, 165, 255),thickness=1)
  
        # Calculating the lip distance #
        lip_dist = cal_yawn(landmarks)
        
        # checking if the distance between the lips is above the threshold and playing the alert
        if lip_dist > yawn_thresh : 
            cv2.putText(frame, 'Yawning :( ',(335,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            pygame.mixer.music.play(-1)
        else: 
            pygame.mixer.music.stop()
            
        
        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        # Now we decide what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            if(sleep>6):
                status="SLEEPING !!!"
                color = (0,0,255)
                pygame.mixer.music.play(-1)
            else: 
                pygame.mixer.music.stop()


        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status="Drowsy !"
                color = (255,0,0)
                pygame.mixer.music.play(-1)
            else: 
                pygame.mixer.music.stop()

        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status="Active :)"
                color = (0,255,0)
        
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)
        
        # adding an exit cue to close the application
        cv2.putText(frame, "press q to exit", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255),1)

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # for displaying the output windows 
    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()        
cv2.destroyAllWindows()

