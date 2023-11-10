import cv2
import time
import datetime

cap = cv2.VideoCapture(0) # will capture the camera, 1 signifies its my macbook cam

# sets up a cascade classifier using a pre-existing one called haarcascade
# haarcascade is prebuilt into opencv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalFace_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

recording = True
#This gets the width and the hight of the frame size of the video
frame_size = (int(cap.get(3)), int(cap.get(4)))
#This records the video in mp4v
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#Output Steam, the 20 is the frame rate,
out = cv2.VideoWriter("video.mp4", fourcc, 20.0, frame_size)


while True:
    _, frame = cap.read() # read
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY) # give us a grayscale img

    # DISCLAIMER: this is a premade algo:
    #   -gray is the img, 1.3 is scale factor (accuracy of this pre-existing algo)
    #   - dont worry about the num 5: its using the min neighbors algo
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 2)

    #it will record when the number of faces and bodies increases
    if len(faces) + len(bodies) > 0:
        recording = True

    #code to record the video
    out.write(frame)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)


    for x, y, width, height in bodies:
        pad_w, pad_h = int(0.15*width), int(0.05*height)
        cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + width - pad_w, y + height - pad_h), (0,255,0), 1)

    cv2.imshow("Camera", frame) # show the camera

    if cv2.waitKey(1) == ord('q'): # q is the key to exit the cam
        break

#this will save the video
out.release()
cap.release()
cv2.destroyAllWindows() # when while loop breaks, destroy all "windows" aka the window