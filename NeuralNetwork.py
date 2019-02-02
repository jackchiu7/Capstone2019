# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:03:38 2019

@author: jackc
"""

import tensorflow as tf
import cv2 as cv
import os




def main():
    trainDirs = [x[0] for x in os.walk("Training")]
    trainLabels = os.listdir('Training')
    trainDirs = trainDirs[1:]
    testDirs = [x[0] for x in os.walk("Test")]
    testDirs = testDirs[1:]
    testLabels = os.listdir('Test')
#    for direc in trainDirs:
#        print(direc)
        
    training = {}
    testing = {}
    for d in trainDirs:
        training[d] = os.listdir(d)
    for d in testDirs:
        testing[d] = os.listdir(d)
    









if __name__ == '__main__': 
    main()



