import cv2

#creating face detection variable
faceDetection = cv2.CascadeClassifier('/home/andre/Projects/Face_Recognition/cascades/haarcascade_frontalface_default.xml')

#creating face recognition variable and aplying LBPH method inside it
faceRecognition = cv2.face.LBPHFaceRecognizer_create()

#reading the classifier inside  variable 
faceRecognition.read('/home/andre/Projects/Face_Recognition/face_detector/Classifier.yml')

#variables of dimensions to draw a rectangle into image 
width, height = 200, 200


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#if you have more than one webcam on your computer, verify the correct number to put inside variable.The number 0 is for default camera
camera = cv2.VideoCapture(0)


#loop to: 
# 1 - call the video from your webcam
# 2 - extract the number id from the name of the image previously saved on your computer 
# 3 - switch the id for your respective names
while True:
    nome = ''
    connected, image = camera.read()
    Gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceDetected = faceDetection.detectMultiScale(Gray_Image, minSize=(100, 100))
    for (x, y, w, h) in faceDetected:
        faceimage = cv2.resize(Gray_Image[y:y + h, x:x + w], (width, height))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, confidence = faceRecognition.predict(faceimage)

        if id == 1:
            nome = 'Andre Luiz'
        elif id == 2:
            nome = 'Gabriel Alcantara'
        else:
            nome = 'Desconhecido'

        cv2.putText(image, nome, (x, y + (h+30)), font, 2, (0, 0, 255))

    cv2.imshow('Face', image)
    if cv2.waitKey(1) == ord('q'):
        break


# closing applications
camera.release()

cv2.destroyAllWindows()

'''

# To use your ip camera or even your cell phone like a webcam, use the code bellow

# Remenber to update the paths to avoid errors 

# IP CAM

import urllib.request
import cv2
import numpy as np
import time

faceDetection = cv2.CascadeClassifier('/home/andre/Projects/Face_Recognition/cascades/haarcascade_frontalface_default.xml')

faceRecognition = cv2.face.LBPHFaceRecognizer_create()

faceRecognition.read('/home/andre/Projects/Face_Recognition/face_detector/Classifier.yml')

width, height = 200, 200

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://your-cam-ip:8080/shot.jpg'

while True:

    # Use urllib to get the image and convert into a cv2 usable format
    imgResp= urllib.request.urlopen(url)

    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)

    nome = ''
    #connected, image = img.read()
    Gray_Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceDetected = faceDetection.detectMultiScale(Gray_Image, minSize=(100, 100))
    for (x, y, w, h) in faceDetected:
        faceimage = cv2.resize(Gray_Image[y:y + h, x:x + w], (width, height))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, confidence = faceRecognition.predict(faceimage)

        if id == 1:
            nome = 'Andre Luiz'
        elif id == 2:
            nome = 'Gabriel Alcantara'
        else:
            nome = 'Desconhecido'

        cv2.putText(img, nome, (x, y + (h+30)), font, 2, (0, 0, 255))

    cv2.imshow('Face', img)
    if cv2.waitKey(1) == ord('q'):
        break

img.release()

cv2.destroyAllWindows()'''
