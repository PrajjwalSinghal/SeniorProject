from cv2 import cv2 
import keras
from string import ascii_uppercase
import numpy as np
from time import time

def predToChar(pred):
    idx = np.argmax(pred)
    return ascii_uppercase[idx], pred[idx]


if __name__ == "__main__":
    
    grayModel = keras.models.load_model("../Trained Models/an3sources_Adam.hdf5")
    
    ##################################
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2
    # get camera access
    camera = cv2.VideoCapture(0)
    # ROI coordinates
    top, right, bottom, left = 200, 850, 500, 1150

    while(True):
        # get current frames
        (grabbed, frame) = camera.read()

        # flipping bsc otherwise its mirror
        frame = cv2.flip(frame, 1)
        # get roi
        roi = frame[top:bottom, right:left]
        # draw a rectangle around ROI
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        ################################## CODE FOR PREDICTING CLASSES ######################################################
        # colorROI = roi.reshape(1,300,300,3)
        grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grayROI = grayROI.reshape(1,300,300,1)

        grayPrediction = grayModel.predict(grayROI)

        grayPred, grayConf = predToChar(grayPrediction[0])

        cv2.putText(frame, "GRAY: " + str(grayPred) + ", " + str(grayConf)[0:4], (75,75), font, fontScale, color, thickness, cv2.LINE_AA)
        ################################### CODE FOR PREDICTING CLASSES ######################################################        

        cv2.imshow("Video Feed", frame)
        cv2.imshow("Image", grayROI[0])
        keypress = cv2.waitKey(1) & 0xFF
        if  keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
