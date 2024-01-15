import cv2
import numpy as np
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture(r"C:\Users\hvendas\Desktop\GIT\app\opticalFlow\steel_ball.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(varThreshold=200)
kernel = np.ones((15, 15), np.uint8)

# Dictionary to store object trajectories
trajectories = {}

while True:
    ret, frame = cap.read()
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 7)
            detections.append([x,y,w,h])


    boxes_ids= tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        center_x = int(x + (w / 2))
        center_y = int(y + (h / 2))
        cv2.putText(frame,str(id),(x,y-15), cv2.FONT_HERSHEY_PLAIN, 5,(255,0,0),7)

                # Update object trajectories
        if id in trajectories:
            trajectories[id].append((center_x, center_y))

        else:
            trajectories[id] = [(center_x, center_y)]

        # Draw the trajectory
        if len(trajectories[id]) > 1:
            for i in range(1, len(trajectories[id])):
                cv2.line(frame, trajectories[id][i - 1], trajectories[id][i], (0, 0, 255), 7)



    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Video", frame)

    mask = cv2.resize(mask, (640, 480))
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
