import cv2 as cv
import picar as pc
import numpy as np

import laneDetection
import control

cap = cv.VideoCapture(0)

pc.setup()
fw = pc.front_wheels.Front_Wheels()
bw = pc.back_wheels.Back_Wheels()

# choose constant velocity for backwheels
bw.backward()
bw.speed = 20

while True:
    ret, img = cap.read()

    # Get trajectory and image
    backwarped, trajectory = laneDetection.pipeline(img) 
    
    # Compute new steering angle
    cp, ci, cd = (1.0, 0.0, 0.5)*0.0005
    alpha = control.steeringAngleAlpha(trajectory, img.shape[1], img.shape[0])

    # Apply steering angle
    fw.turn(alpha / np.pi * 180) # expects angle in degrees

    # Stop program if user wants it
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    cv.imshow("Real-Time", backwarped)


cap.release()
cv.destroyAllWindows()