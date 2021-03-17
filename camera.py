import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the resources
model = keras.models.load_model('emotions_detection_2')
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def process_frame(frame):
    # Detect faces with haarcascade
    faces = haarcascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=7)
    # If a face is detected predict the emotion
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            # Crop the image on the detected face
            image_predict = frame[y:y+h,x:x+w]
            # Switch image to Gray and resize it to 48x48 
            image_predict = cv2.cvtColor(image_predict, cv2.COLOR_BGR2GRAY)
            image_predict = cv2.resize(image_predict, (48, 48))
            image_predict = image_predict.reshape(-1, 48, 48, 1).astype('float')/255.0
            # Make the prediction using the model
            pred = model.predict_classes(image_predict)
            #Display a square around the face and add the predicted emotion text above
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 
            frame = cv2.rectangle(frame, (x, y), (x+w, y - 20), (0,255,0), -1)
            frame = cv2.putText(frame, labels[pred[0]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    return frame

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if ret == True:
        # Process the image
        image = process_frame(frame)  
    cv2.imshow("test", image)

    k = cv2.waitKey(1)
    if k%256 == 27:
        break

cam.release()
cv2.destroyAllWindows()