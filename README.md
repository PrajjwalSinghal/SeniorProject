Kaggle link to the trained model: www.kaggle.com/dataset/655c4b441eebaa8246a4b50380ded3f39101ce80d57e2948bc110aba2995b811


# Web-Application

## Project setup

```
npm install
```

### To run Server Side

```
run: node server.js

```

### Compiles and hot-reloads for development

```
npm run start
```


# Deep-Learning

This directory contains all the data and files required to train the deep learning model to detect ASL fingerspelling.
Also, this directory contains a program to do a live demo of ASL recognition, using the webcam.

To start with model training or demo, you will need to setup a python environment.

## Python Environment Setup Steps

Naviage inside "DeepLearning" directory and run the following commands on the terminal

1. python3 -m venv localPythonEnvironment

This will create a new python environment inside the DeppLearning directory.
We will install all the required dependencies in this environment

2. source localPythonEnvironment/bin/activate

This will activate the local python environment and you should be able to see "(localPythonEnvironment)" in your terminal

If you have activate your environment your terminal should have something like this:
(localPythonEnvironment) Prajjwals-MacBook-Pro:DeepLearning prajjwalsinghal$ 

If the environment hasn't been activated, your terminal will have something like this:
Prajjwals-MacBook-Pro:DeepLearning prajjwalsinghal$ 

3. pip install -r requirements.txt

This will install all the required dependencies to the localPythonEnvironment

4. pip install -U plaidml-keras
5. plaidml-setup

These steps are necessary for running the model on AMD GPU's, if you want to run on CPU or NVIDIA GPU's skip step 4 and 5 and comment the first 4 lines of LiveDemo.py

## Steps to run the live demo program.

Make sure you have completed the setup for python environment and you have activate the environment.
Download the trained model from www.kaggle.com/dataset/655c4b441eebaa8246a4b50380ded3f39101ce80d57e2948bc110aba2995b811 and move it to DeepLearning/Model Testing/Trained Models/

Navigate to the DeepLearning/ModelTesting/src/ folder and in your terminal type "python LiveDemo.py".

This will turn on your webcam and will launch a new window, in the window place your hand inside the green bounding box, and you will be able to see a prediction for the ASL sign the you make.

Note: For the LiveDemo.py program to run, you will have to give camera access permission to terminal
