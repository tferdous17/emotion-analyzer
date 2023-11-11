import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils # utility function for drawing
mp_holistic = mp.solutions.holistic # premade thing for landmarks (hand, body, pose etc)

# drawing spec that defines color, thickness of lines, and radius for the vertices
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

# author: Tasnim Ferdous
cap = cv2.VideoCapture(0) # cap means "capture", so the video we're capturing from the webcam
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened(): # while image is still being captured
        ret, frame = cap.read()
        # Define color scheme for feed
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # results stores processing results on img
        results = holistic.process(image)


        # draw face "landmarks": print the contour of the analyzed face thru webcam
        # facemesh contours is the lines and circles that pop up around your face
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        # draw landmarks for right hand
        # again draws lines and vertexes and accurately tracks your hand movements
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # author: Gagan Sapkota
        # draw landmarks for left hand
        #draws purple line for differentiation
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # drew print pink landmarks
        # draw landmarks for pose (full body)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        #This line of code shows the feed
        cv2.imshow('Webcam Feed', image)

        #Key q exists the webcam
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

#This line of code closes the window
cap.release()
cv2.destroyAllWindows()