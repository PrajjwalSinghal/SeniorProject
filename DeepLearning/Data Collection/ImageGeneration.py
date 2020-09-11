from cv2 import cv2 
from time import time

uniqueString = "Prajjwal"

# lower case
char = "y"
trainPath = "data/train/" + char + "/"
valPath = "data/val/" + char + "/"

if __name__ == "__main__":

    count = 0
    startCapturing = False
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

        ################################### CODE FOR STORING IMAGES #########################################################
        if startCapturing:
            if count < 500:
                path = trainPath
                text = "train"
            elif count < 600:
                path = valPath
                text = "val"
            else:
                startCapturing = False
        else:
            cv2.putText(frame, "Press " + char, (70,70), font, fontScale, color, thickness, cv2.LINE_AA)
       
        if startCapturing:
            cv2.putText(frame, text + " " + str(count), (70,70), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite(path + char.upper() + uniqueString + str(count) + ".jpg",roi)
            count += 1
        ################################### CODE FOR STORING IMAGES #########################################################
        

        cv2.imshow("Video Feed", frame)
        
        keypress = cv2.waitKey(1) & 0xFF
        ################################### CODE FOR STORING IMAGES #########################################################
        if keypress == ord(char):
            startCapturing = True
        ################################### CODE FOR STORING IMAGES #########################################################
        if  keypress == ord("a"):
            break   # Exit the loop

# free up memory
camera.release()
cv2.destroyAllWindows()
