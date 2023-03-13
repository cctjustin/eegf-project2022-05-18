import time
import cv2
import numpy as np
import sys
import playsound
from imutils import paths
import face_recognition
import pickle
import os

# making 0 to prevent error
with open("distance.txt", "w") as f:
    f.write(str(0))

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

photo_count = 0
photo_number = 20
cap = cv2.VideoCapture(0)
face_width = []
rest = {}
key = 0
time_storage = []
cal_length = float(input("Please insert the distance you would like to keep from the screen (in cm): "))
real_width = float(input("Please insert your face width (in cm): "))
rest_length = float(input("Please insert the length of time for 1 period (in mins): ")) * 60
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
        if photo_count < photo_number:
            roi = frame[y:y + h, x:x + w]
            image_item = "User/" + str(photo_count) + ".png"
            print(image_item)
            cv2.imwrite(image_item, roi)
            photo_count += 1
            time.sleep(0.2)
    cv2.imshow("Video", frame)
    if len(face_width) >= 30:
        break

cap.release()
cv2.destroyAllWindows()

# Extract features of the photos taken
imagePaths = list(paths.list_images('User'))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()

# Compute Distance
print(face_width)
cal_val = sum(face_width)/len(face_width)
print(cal_val)
virtual_distance = cal_val / real_width * cal_length
constant = virtual_distance * real_width
face_width = []
time.sleep(1)

# Read pickle file
data = pickle.loads(open('face_enc', "rb").read())
print("Streaming started")

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
    # Identify the Face

    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values and True for the embeddings it matches closely
        # and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        # set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                # Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            # set name which has highest count
            name = max(counts, key=counts.get)
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
    for (x, y, w, h) in faces:
        if w > cal_val:
            cv2.putText(frame, "Too close!", (x- 200,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,cv2.LINE_4,)
            cv2.putText(frame, str(round(constant / w, 1)) + "cm", (x + w - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4, )
            # print(time.time() - start)
            with open("distance.txt", "w") as f:
                f.write(str(round(constant / w,1)))
            playsound.playsound('853_960.mp3')

        else:
            cv2.putText(frame, str(round(constant / w, 1)) + "cm", (x + w - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_4, )
            # print(time.time() - start)
            with open("distance.txt", "w") as f:
                f.write(str(round(constant / w,1)))
        if time.time() - start > rest_length:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Go get some rest!", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_4, )
            playsound.playsound('ring.mp3')

    # deciding whether a face exists
    if len(faces) > 0:
        disappear_time = 0
        if disappear:
            disappear = False
            start = time.time()
            rest[key]['Length of rest'] = str(round(time.time() - time_storage[key - 1],1)) + ' seconds'
    else:
        # define disappear
        if disappear_time == 0:
            disappear_time = time.time()
            # print(time.time() - disappear_time)
    if (15 > time.time() - disappear_time > 10) and not disappear:
        disappear = True
        key += 1
        rest[key] = {'Start time':time.ctime(), 'Length of rest': 0}
        time_storage.append(time.time())


    cv2.imshow("Video", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Press q to quit
cap.release()
cv2.destroyAllWindows()
print(rest)

