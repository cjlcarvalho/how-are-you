#! /usr/bin/python

import cv2
import sys

from train import train
from test import test

def main():

    if len(sys.argv) < 2:

        print('Please specify your option [test/train]')

        return 1

    emotions = ['NEUTRAL', 'HAPPY', 'SAD', 'SURPRISED']

    if sys.argv[1] == 'train':

        train(emotions)

    elif sys.argv[1] == 'test':

        cam = cv2.VideoCapture(0)

        cv2.namedWindow('How are you?')

        count = 0

        while True:

            ret, frame = cam.read()

            cv2.imshow('How are you?', frame)

            if not ret:

                break

            k = cv2.waitKey(1)

            if k % 256 == 27:

                print('Closing...')

                break

            elif k % 256 == 32:

                img_name = 'frame_%d.jpg' % count

                cv2.imwrite(img_name, frame)

                test(img_name, emotions)

                count += 1

    else:

        print('Wrong option')

        return 1

    return 0

main()
