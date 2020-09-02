import cv2
import os
import numpy as np

#eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)

#fisherface = cv2.face.FisherFaceRecognizer_create()

# There are these 2 other classificators, but im not will use in this code because of the low accuracy of them 

lbph = cv2.face.LBPHFaceRecognizer_create()

# DEF to read the names of this pictures, taken previously, and classify these persons by his IDs

def GetImageWithId():
    faces = []
    Ids = []
    paths = [os.path.join('/home/andre/Projects/Face_Recognition/pictures', p) for p in os.listdir('/home/andre/Projects/Face_Recognition/pictures')]
    paths.sort()
    for t in paths:
        Gray_Image = cv2.cvtColor(cv2.imread(t), cv2.COLOR_BGR2GRAY)
        Ids.append(int(t[-8:-7]))
        faces.append(Gray_Image)

    return np.array(Ids), faces


ids, faceimage = GetImageWithId()



print('Training Faces...')

'''eigenface.train(faceimage, ids)
eigenface.write('EigenClassifier.yml')

fisherface.train(faceimage, ids)
eigenface.write('FisherClassifier.yml')'''

lbph.train(faceimage, ids)
lbph.write('EigenClassifier.yml')

print('Trained finished')

GetImageWithId()








