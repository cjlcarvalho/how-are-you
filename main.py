#! /usr/bin/python

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

        if len(sys.argv) == 3:

            test(sys.argv[2], emotions)

        else:

            print('No input was given')

            return 1

    else:

        print('Wrong option')

        return 1

    return 0

main()
