
import cv2
import numpy as np


def pre_processing(image, width, height):
    #cvtColor will convert to one channale here its gray instead of rgb
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    #applies binary thereshold to grayscale image
    #convert image to binary image where image is either 0(black) or 255(white) based on thereshold 1
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    #here 1 dimentional is added in the begining 
    #so the dinmenison is now (1, width, height)
    return image[None, :, :].astype(np.float32)
