import os
import plaidml
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from cv2 import cv2
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from string import ascii_uppercase
import numpy as np
from time import time

def predToChar(pred):
    idx = np.argmax(pred)
    return ascii_uppercase[idx], pred[idx]

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=96, input_shape=(300,300,1), kernel_size=(11,11),strides=(4,4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(300*300*1,), activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(14, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
    return model

if __name__ == "__main__":
    
    grayModel1 = create_model()
    grayModel1.load_weights("../Trained Models/an3sources_Adam.hdf5")
    
    grayModel2 = create_model()
    grayModel2.load_weights("../Trained Models/an_everybody.hdf5")
    
    grayModel3 = create_model()
    grayModel3.load_weights("../Trained Models/new_test_data.hdf5")
    
#    grayModel1 = keras.models.load_model("../Trained Models/an3sources_Adam.hdf5")
#    grayModel2 = keras.models.load_model("../Trained Models/an_everybody.hdf5")
#    grayModel3 = keras.models.load_model("../Trained Models/new_test_data.hdf5")
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

        grayPrediction1 = grayModel1.predict(grayROI)
        grayPrediction2 = grayModel2.predict(grayROI)
        grayPrediction3 = grayModel3.predict(grayROI)

        grayPred1, grayConf1 = predToChar(grayPrediction1[0])
        grayPred2, grayConf2 = predToChar(grayPrediction2[0])
        grayPred3, grayConf3 = predToChar(grayPrediction3[0])
        

        cv2.putText(frame, "Old: " + str(grayPred1) + ", " + str(grayConf1)[0:4], (75,75), font, fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.putText(frame, "Mid: " + str(grayPred2) + ", " + str(grayConf2)[0:4], (75,105), font, fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.putText(frame, "New: " + str(grayPred3) + ", " + str(grayConf3)[0:4], (75,135), font, fontScale, color, thickness, cv2.LINE_AA)
        ################################### CODE FOR PREDICTING CLASSES ######################################################        

        cv2.imshow("Video Feed", frame)
        cv2.imshow("Image", grayROI[0])
        keypress = cv2.waitKey(1) & 0xFF
        if  keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
