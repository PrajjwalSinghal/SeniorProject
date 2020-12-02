# Inception


import os
import plaidml
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from cv2 import cv2
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from string import ascii_uppercase
import numpy as np
from keras import applications
from time import time
from spellchecker import SpellChecker


d = {0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
 10: 'K',
 11: 'L',
 12: 'M',
 13: 'N',
 14: 'O',
 15: 'P',
 16: 'Q',
 17: 'R',
 18: 'S',
 19: 'T',
 20: 'U',
 21: 'V',
 22: 'W',
 23: 'X',
 24: 'Y',
 25: 'Z',
 26: 'del',
 27: 'nothing',
 28: 'space'}
# d = {0: 'O', 1: 'P', 2: 'R', 3: 'S', 4: 'T', 5: 'V', 6: 'W', 7: 'X', 8: 'Y'}
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]
no_of_classes = 25

def createKaggleModel():
    model = tf.keras.models.load_model("../Trained Models/Kaggle_28_28_1.h5")
    return model
    my_model = Sequential()
    my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
    my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
    my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
    my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Flatten())
    my_model.add(Dropout(0.5))
    my_model.add(Dense(512, activation='relu'))
    my_model.add(Dense(n_classes, activation='softmax'))
    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    my_model.load_weights("../Trained Models/kaggleModel.h5")
    return my_model

def createResNetModel():
    adam = Adam(lr=0.0001)
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape = (300, 300, 1))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(no_of_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights("../Trained Models/FinalModel.h5")
    return model
def create_model_new():
    sgd = SGD(lr=0.002)
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

    model.add(Dense(9, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights("../Trained Models/AlexNet_without_QandU.hdf5")
    return model

def create_model_old():
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

def predToChar(pred, jumpstart = 0):
    idx = np.argmax(pred)
    return d[idx+jumpstart], pred[idx]
    return ascii_uppercase[idx+ jumpstart], pred[idx]

if __name__ == "__main__":
    
    firstHalfAlphabets = create_model_old()
    firstHalfAlphabets.load_weights("../Trained Models/new_test_data.hdf5")

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
    # Word creation variables
    currentChar = ""
    currentWord = ""
    sentence = []
    lastPredTime = time()
    # Autocorrect API
    spell = SpellChecker()
    
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
        grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grayROI = grayROI.reshape(1,300,300,1)

        firstHalfAlphabets_prediction = firstHalfAlphabets.predict(grayROI)

        firstHalfAlphabets_pred, firstHalfAlphabets_conf = predToChar(firstHalfAlphabets_prediction[0])

        cv2.putText(frame, "A-N Model: " + str(firstHalfAlphabets_pred) + ", " + str(firstHalfAlphabets_conf)[0:4], (75,75), font, fontScale, color, thickness, cv2.LINE_AA)
        ################################### CODE FOR PREDICTING CLASSES ######################################################        

        ################################### CODE FOR CREATING A WORD ######################################################
        
        if not currentChar or currentChar != str(firstHalfAlphabets_pred):
            currentChar = str(firstHalfAlphabets_pred)
            lastPredTime = time()
        elif time() - lastPredTime > 1.5 and (not currentWord or currentChar != currentWord[-1]):
            currentWord += currentChar
            currentChar = ""

        cv2.putText(frame, "Current Char: " + currentChar , (75,110), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Current Word: " + currentWord , (75,140), font, fontScale, color, thickness, cv2.LINE_AA)
        
        x = 300
        y = 600
        cv2.putText(frame, "Input", (x-250,y), font, fontScale/1.1, (0, 0, 0), thickness, cv2.LINE_AA)
        for word in sentence:
            cv2.putText(frame, word[0], (x,y), font, fontScale/1.1, (0, 0, 255), thickness, cv2.LINE_AA)
            x += 200

        x = 300
        y = 550
        cv2.putText(frame, "AutoCorrect", (x-250,y), font, fontScale/1.1, (0, 0, 0), thickness, cv2.LINE_AA)
        for word in sentence:
            cv2.putText(frame, word[1], (x,y), font, fontScale/1.1, (0, 255, 0), thickness, cv2.LINE_AA)
            x += 200

        cv2.imshow("Video Feed", frame)
        cv2.imshow("Image", grayROI[0])
        keypress = cv2.waitKey(1) & 0xFF
        if  keypress == ord("q"):
            break
        if keypress == ord('b') and currentWord:
            sentence.append([currentWord, spell.correction(currentWord[1:]).upper()])
            currentWord = ""
            currentChar = ""
            lastPredTime = time()
            if len(sentence) > 3:
                sentence.pop(0)
        if keypress == ord('x') and currentWord:
            currentWord = currentWord[:-1]       

        

# free up memory
camera.release()
cv2.destroyAllWindows()
