# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:03:38 2019

@author: jackc
"""

import tensorflow as tf
import cv2 
import os
import glob
from tensorflow.keras import layers
from keras.applications import ResNet50
from keras.applications import vgg16
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import SGD
import keras
from keras.layers.normalization import BatchNormalization
from keras.applications import imagenet_utils


def openImages(limit = -1, train = True, test = True, size = (25,25)):
    trainDirs = [x[0] for x in os.walk("Training")]
    trainL = os.listdir('Training')
    trainDirs = trainDirs[1:]
    testDirs = [x[0] for x in os.walk("Test")]
    testDirs = testDirs[1:]
    testL = os.listdir('Test')
#    for direc in trainDirs:
#        print(direc)
    
    preprocess = imagenet_utils.preprocess_input
    training = []
    testing = []
#    for d in trainDirs:
#        training.append(os.listdir(d))
#    for d in testDirs:
#        testing.append(os.listdir(d))
    
    training = []
    testing = []
    trainLabels = []
    testLabels = []
    
    if train:
        for i in trainL:
            ctr = 0
            for file in glob.glob('Training/'+ i + '/*.jpg'):
                image = load_img(file, target_size=size)
                image = img_to_array(image)
                image = preprocess(image)
                training.append(image)
                ctr += 1
                trainLabels.append(i)
                if ctr == limit:
                    break
    
    
        
    if test:
        for i in testL:
            ctr = 0
            for file in glob.glob('Test/'+ i + '/*.jpg'):
                image = load_img(file, target_size=size)
                image = img_to_array(image)
                image = preprocess(image)
                testing.append(image)
                ctr += 1
                testLabels.append(i)
                if ctr == limit:
                    break
    
    return trainLabels, training, testLabels, testing

def preProcess(img, size = (90,90)): 
    
    return cv2.resize(img, size)


def snn():
    dv = .04*v^2 + 5*v + 140 - u + I
    du = a*(b*v-u)
    v += dv*dt
    u += du*dt
    

def main():
    imgSize = (224,224)
    
    trainLabels, training, testLabels, testing = openImages(32, train = True, test = False, size = imgSize)
    numFruits = len(trainLabels)
    
    base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(imgSize[0], imgSize[1], 3))
    model = Sequential()
    for layer in base.layers[:22]:
        model.add(layer)
    
    for layer in model.layers[:20]:
        layer.trainable = False
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numFruits,activation='sigmoid'))
    
    model.summary()
    
    sgd = SGD(lr=0.001)
    model.compile(sgd, loss='mse',metrics=['acc'])
    
    print(np.shape(training))
#    print(np.shape(trainLabels))
#    print(trainLabels)
    model.fit(training, trainLabels, epochs=10, batch_size=32)

    
    
    
    
#    
#    img = training[1][2]
#    cv2.namedWindow(trainLabels[1], cv2.WINDOW_NORMAL)
#    cv2.imshow(trainLabels[1],img)
#    a = np.reshape(img, (2500,3))
#    print(np.shape(img))
#    print(np.shape(a))
#    
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
    
#    tau = 10*ms
#    eqs = '''
#    dv/dt = (1-v)/tau : 1
#    '''
#    
#    start_scope()
#
#    G = NeuronGroup(1, eqs, method='exact')
#    print('Before v = %s' % G.v[0])
#    run(100*ms)
#    print('After v = %s' % G.v[0])
#    
#    
#    group = NeuronGroup(N, 'dv/dt = -v / tau : volt', threshold=-50*mV, reset=-70*mV)
    







if __name__ == '__main__': 
    main()



