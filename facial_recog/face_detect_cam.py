import time
import cv2
import numpy as np
import sys
import playsound

# making 0 to prevent error
with open("distance.txt", "w") as f:
    f.write(str(0))

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
face_width = []
cal_length = float(input("Please insert the distance you would like to keep from the screen (in cm): "))
real_width = float(input("Please insert your face width (in cm): "))
rest_length = float(input("Please insert the length of time for 1 period (in mins): "))
print("Calibration required. Please keep your face at the position you would like to keep.")

c = 0
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_width.append(w)
        print(face_width)
    cv2.imshow("Video", frame)
    if len(face_width) >= 30:
        break

cap.release()
cv2.destroyAllWindows()
print(face_width)
cal_val = sum(face_width)/len(face_width)
print(cal_val)
virtual_distance = cal_val / real_width * cal_length
constant = virtual_distance * real_width
face_width = []
time.sleep(1)

cap = cv2.VideoCapture(0)

start = time.time()
disappear_time = 0
disappear = False
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
            cv2.putText(frame, str(round(constant / w, 1)) + "cm", (x + w - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4, )
            # print(time.time() - start)
            with open("distance.txt", "w") as f:
                f.write(str(round(constant / w,1)))
            playsound.playsound('853_960.mp3')

        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(round(constant / w, 1)) + "cm", (x + w - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_4, )
            # print(time.time() - start)
            with open("distance.txt", "w") as f:
                f.write(str(round(constant / w,1)))
        if time.time() - start > rest_length:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Go get some rest!", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_4, )
            playsound.playsound('ring.mp3')
    cv2.imshow("Video", frame)

    # deciding whether a face exists
    if len(faces) > 0:
        disappear_time = 0
        if disappear:
            disappear = False
            start = time.time()
    else:
        # define disappear
        if disappear_time == 0:
            disappear_time = time.time()
            # print(time.time() - disappear_time)
    if 600 > time.time() - disappear_time > 10:
        disappear = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Press q to quit
cap.release()
cv2.destroyAllWindows()

