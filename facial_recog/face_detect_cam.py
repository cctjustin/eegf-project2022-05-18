import time
import cv2
import numpy as np
import sys
import playsound

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
face_width = []
print("Calibration required. Please keep your face at the position you would like to keep.")

c = 0
while c <= 3:
    # Read the video capture
    ret, frame = cap.read()
    # Turn it gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Add rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_width.append(w)
    cv2.imshow("Video", frame)
    time.sleep(1)
    c += 1

cap.release()
cv2.destroyAllWindows()
print(face_width)
cal_val = sum(face_width)/len(face_width)
print(cal_val)
face_width = []
time.sleep(3)

cap = cv2.VideoCapture(0)
while True:
    # Read the video capture
    ret, frame = cap.read()

    # Turn it gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Add rectangle
    for (x, y, w, h) in faces:
        if w > cal_val:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Too close!", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,cv2.LINE_4,)
            playsound.playsound('853_960.mp3')
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Press q to quit
cap.release()
cv2.destroyAllWindows()

