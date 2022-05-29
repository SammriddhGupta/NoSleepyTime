# [_No Sleepy Time_](https://github.com/SammriddhGupta/NoSleepyTime/edit/main/README.md)

## A Drowsiness Detection System
### Alerts you by tracking your eye and mouth movement

<img src = "Images/No Sleepy Time.png" width=2000 height=500>

## Motivation

- Many people face long nights at work. Especially truck drivers, security guards and healthcare professionals. These are jobs that are fundamental to society and to the health, wellbeing and comfort of the general population. 

- Getting behind the wheel while feeling tired is not uncommon. It’s something most drivers have probably done. It can be incredibly dangerous, unless we find a way to warn drivers when their tiredness has become too severe and is impairing their driving.

Therefore I decided to work on a detection system using facial recognition to detect drowsiness and trigger an alert before it is too late

## Technology Stack

- OpenCV
OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. 

- Dlib 
Dlib is an open source suite of applications and libraries written in C++ under a permissive Boost license. Dlib offers a wide range of functionality across a number of machine learning sectors, including classification and regression, numerical algorithms such as quadratic program solvers, an array of image processing tools, and diverse networking functionality, among many other facets.

- Face_Recognition
A python package that wraps Dlib’s face recognition functions into a simple, easy to use API.


## Features
The detector plays an alarm when the status is detected as: 
- drowsy - eyes slightly closed
- sleeping - eyes fully closed
- yawning - mouth wide open


## Working 

Facial landmarks is a technique for localizing and representing salient regions or facial parts of the person’s face. It has various applications like face alignment, head pose estimation, face swapping, blink detection, drowsiness detection and so on. 
In our use case, the aim is to detect facial structures on the person’s face using a method called shape prediction.

There are 2 important steps in facial landmark detection: 
- Detecting or Localizing the face.
- Predicting the landmarks of key facial regions in the detected face.

The pre-trained facial landmark detector inside the dlib library is an implementation of the paper 

[One Millisecond Face Alignment with an Ensemble of Regression Trees (by Kazemi and Sullivan (2014)](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) 

The facial landmark detector which is pre-trained inside the Dlib library of python for detecting landmarks, is used to estimate the location of 68 points or (x, y) coordinates which map to the facial structures. 
These 68-(x,y) coordinates represent the important regions of the face like mouth, left eyebrow, right eyebrow, left eye, right eye, nose, and jaw.

The indices of the 68 coordinates or points can be easily visualized in the image below:

<img src = "Images/68-facial-landmarks.png">

The Locations of the Facial Features are as follows:
- The left eye is accessed with points [42, 47].
- The right eye is accessed using points [36, 41].
- The mouth is accessed through points [48, 67].

Once the landmarks are predicted, we use only the eye landmarks and the mouth landmarks to determine the Eye Aspect Ratio(EAR) and Mouth Aspect Ratio(MAR) to check if a person is drowsy.

Based on the paper, [Real-Time Eye Blink Detection using Facial Landmarks](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

We can derive an equation that reflects this relation called the eye aspect ratio (EAR):
<img src = "Images/EAR_formula.png">

The idea is to design a system which rings a constant alarm whenever the status is detected as drowsy or sleepy based on the EAR ratio. 
An additional feature is the detection of yawning as well. 

### A little bit on the Dlib library: 

The dlib library provides two functions that can be used for face detection:
- A HOG + Linear SVM face detector that is accurate and computationally efficient.
- A Max-Margin (MMOD) CNN face detector that is both highly accurate and very robust, capable of detecting faces from varying viewing angles, lighting conditions, and occlusion.

The _get_frontal_face_detector_ function does not accept any parameters. 
A call to it returns the pre-trained HOG + Linear SVM face detector included in the dlib library.
Dlib’s HOG + Linear SVM face detector is fast and efficient. By nature of how the Histogram of Oriented Gradients (HOG) descriptor works, it is not invariant to changes in rotation and viewing angle.

We’re going to use OpenCV for computer vision, the Dlib library for facial recognition and the imutils package to use some functions that will help us convert the landmarks to NumPy array and make it easy for us to use. 

## Screenshots

