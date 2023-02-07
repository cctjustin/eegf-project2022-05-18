import cv2
import numpy as np
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

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

    face_width = []
    print(faces)
    # Add rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Video", frame)
    face_width.append(w)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Press q to quit
cap.release()
cv2.destroyAllWindows()

print(face_width)
print(sum(face_width)/len(face_width))