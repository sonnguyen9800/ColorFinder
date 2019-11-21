#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------
import sys

import matplotlib.pyplot as plt
import numpy as np
import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os.path

from skimage.color import rgb2lab, deltaE_cie76


cap = cv2.VideoCapture(0)
(ret, frame) = cap.read()
prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'


BOUNDERIES = {
        "RED": ([17, 15, 100], [50, 56, 200]),
        "BLUE": ([86, 31, 4], [220, 88, 50]),
        "YELLOW": ([25, 146, 190], [62, 174, 250]),
        "GRAY": ([103, 86, 65], [145, 133, 128])
    }

COLORS = {
        'RED': [255, 0, 0],
        'YELLOW': [255, 165, 0],
        'BLUE': [75, 0, 130],
        'VIOLET': [127, 0, 255],
        'GREEN': [0, 128, 0],
        # 'YELLOW': [255, 255, 0],
        # 'BLUE': [0,0,255]
    }

def comparison(colorsRGB, Colors: dict):
    selected_color = rgb2lab(np.uint8(np.asarray([[colorsRGB]])))
    min = sys.maxsize
    color_picked = []
    for color in Colors:
        curr_color = rgb2lab(np.uint8(np.asarray([[Colors[color]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if diff < min:
            min = diff
            color_picked = color
    return color_picked


if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')


upper_left = (50, 50)
bottom_right = (300, 300)

color = 'rgb'
bins = 16

while True:
    # Capture frame-by-frame
    (ret, frame) = cap.read()

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


    # Rectangle marker
    r = cv2.rectangle(final, upper_left, bottom_right, (100, 50, 200), 1)
    rect_img = final[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    # cv2.imshow("Original", frame)

    cv2.putText(
        final,
        'Prediction: ' + prediction,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        200,
        )

    cv2.imshow("original", frame)
    cv2.imshow('final', final)

    # color_histogram_feature_extraction.color_histogram_of_test_image(final)
    color_extracted = color_histogram_feature_extraction.getRGBvalues(rect_img)

    prediction = comparison(color_extracted, COLORS)

    # prediction = knn_classifier.main('training.data', 'test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

