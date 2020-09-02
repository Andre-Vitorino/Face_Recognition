import cv2
import numpy as np

#if you have more than one webcam on your computer, verify the correct number to put inside variable.The number 0 is for default camera
video = cv2.VideoCapture(0)

# Using a haarcascade model to detect your face in the image
face_classifier = cv2.CascadeClassifier('/home/andre/Projects/Face-Recognition/cascades/haarcascade_frontalface_default.xml')

#Using a haarcascade model to detect your eyes in the image
eye_classifier = cv2.CascadeClassifier('/home/andre/Projects/Face-Recognition/cascades/haarcascade_eye.xml')

sample = 1

totalSamples = 25

name = input('Type your name: ')

id = input('Type your ID: ')

width, height = 200, 200

print('Capturing Faces...')

# Loop make to:
# 1 - Get 25 pictures 
# 2 - Saving these pictures by using this pattern: Name - ID - Number of the actual picture

while True:
    connected, image = video.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detection = face_classifier.detectMultiScale(gray_image, minSize=(200, 200))

    for (x, y, w, h) in face_detection:
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        eyes_area = img[y:y + h, x:x + w]
        eyes_area_gray = cv2.cvtColor(eyes_area, cv2.COLOR_BGR2GRAY)
        eye_detection = eye_classifier.detectMultiScale(eyes_area_gray, minSize=(5, 5), minNeighbors=10)
        for (xe, ye, we, he) in eye_detection:
            cv2.rectangle(eyes_area, (xe, ye), (xe + we, ye + he), (0, 0, 255), 2)

            if cv2.waitKey(1) & 0xFF == ord('t'):
                FaceImage = cv2.resize(gray_image[y:y + h, x:x + w], (width, height))

                if sample < 10:
                    cv2.imwrite(f'pictures/{str(name)}_{str(id)}_0{str(sample)}.jpg', FaceImage)
                    print(f'picture {str(sample)} sucefully captured')
                    sample += 1
                else:
                    cv2.imwrite(f'pictures/{str(name)}_{str(id)}_{str(sample)}.jpg', FaceImage)
                    print(f'picture {str(sample)} sucefully captured')
                    sample += 1

    cv2.imshow('Face', image)
    cv2.waitKey(1)
    if sample >= totalSamples + 1:
        break

    cv2.waitKey(1)

print('All pictures sucefully captured')

# Commands to kill the application

video.release()

cv2.destroyAllWindows()
