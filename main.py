#! /usr/bin/python

import cv2
import sys

from ann.train import train
from ann.test import test

def main():

    if len(sys.argv) < 2:

        print('Please specify your option [test/train]')

        return

    emotions = ['NEUTRAL', 'HAPPY', 'SAD', 'SURPRISED']

    if sys.argv[1] == 'train':

        train(emotions)

    elif sys.argv[1] == 'test':

        cam = cv2.VideoCapture(0)

        cv2.namedWindow('How are you?')

        while True:

            ret, frame = cam.read()

            if not ret:

                break
            
            predicted = test(frame, emotions)

            cv2.imshow('How are you?', predicted if predicted is not None else frame)
            
            k = cv2.waitKey(1)
            
            if k % 256 == 27:
                
                print('Closing...')

                break

    else:

        print('Wrong option')

        return

main()
