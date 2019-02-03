# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:03:38 2019

@author: jackc
"""

import tensorflow as tf
import cv2 
import os
import numpy as np
import glob

def openImages(limit = -1, train = True, test = True):
    trainDirs = [x[0] for x in os.walk("Training")]
    trainLabels = os.listdir('Training')
    trainDirs = trainDirs[1:]
    testDirs = [x[0] for x in os.walk("Test")]
    testDirs = testDirs[1:]
    testLabels = os.listdir('Test')
#    for direc in trainDirs:
#        print(direc)
        
    training = []
    testing = []
#    for d in trainDirs:
#        training.append(os.listdir(d))
#    for d in testDirs:
#        testing.append(os.listdir(d))

    if train:
        for i in trainLabels:
            temp = []
            for file in glob.glob('Training/'+ i + '/*.jpg'):
                temp.append(cv2.imread(file))
                if len(temp) == limit:
                    break
            training.append(temp)
    
    if test:
        for i in testLabels:
            temp = []
            for file in glob.glob('Test/'+ i + '/*.jpg'):
                temp.append(cv2.imread(file))
                if len(temp) == limit:
                    break
            testing.append(temp)
    
    return trainLabels, training, testLabels, testing


def main():
    
    trainLabels, training, testLabels, testing = openImages(5, train = True, test = False)

    img = training[1][2]
    cv2.namedWindow(trainLabels[1], cv2.WINDOW_NORMAL)
    cv2.imshow(trainLabels[1],img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    







if __name__ == '__main__': 
    main()



