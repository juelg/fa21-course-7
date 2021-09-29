import cv2 as cv
import picar as pc
import numpy as np
import time

from laneDetection.classic_cv import pipeline
from control.pidSteeringAngle import steeringAngleAlpha

cap = cv.VideoCapture(0)

pc.setup()
fw = pc.front_wheels.Front_Wheels()
bw = pc.back_wheels.Back_Wheels()

# choose constant velocity for backwheels
bw.backward()

bw.speed = 30

try:
    while True:
        start = time.time()
        ret, img = cap.read()

        # Get trajectory and image
        backwarped, trajectory = pipeline(img) 
        print(trajectory)
        
        # Compute new steering angle
        cp, ci, cd = np.array([1.0, 0.1, 0.0])*0.0005
        alpha = steeringAngleAlpha(trajectory, img.shape[1], img.shape[0], cp, ci, cd)

        # Apply steering angle
        fw.turn(alpha / np.pi * 180) # expects angle in degrees

        # Stop program if user wants it
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()
        print("Duration: {:.2f}s".format(end-start))

        #cv.imshow("Real-Time", backwarped)
except KeyboardInterrupt as e:
    bw.speed = 0
    fw.turn(90)
    cap.release()
    cv.destroyAllWindows()