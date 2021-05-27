from __future__ import print_function
from __future__ import division

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os


# BLACK & WHITE IMAGING
def black_white(filename):  
    print(filename)
    image = cv.imread('./static/media/122677.png')
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite("./static/media/filename2.jpg", image_gray)
