'''
Script for  stitching together two images.
Intended for stitching images together in a mosaic, but can also theoretically 
be used for a panorama.

Depends on opencv 3, and imutils packages

Usage: python -f <first image path> -s <second image path> -o <output image path>

Partial code credit goes to the tutorial at this url
https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

The basic logic and library calls come from this tutorial, but we have also
made significant changes to get performance to work as we expect it to.
'''
# 

# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 

# import the necessary packages
import imgstitcher
import argparse
import imutils
import cv2
import numpy as np

import sys

if len(sys.argv) != 4:
	print("usage: python3 stitch.py <input 1> <input 2> <output>")
	sys.exit(1)

input = sys.argv[1]
input2 = sys.argv[2]
output = sys.argv[3]

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(input)
imageB = cv2.imread(input2)

imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
result = imgstitcher.stitch([imageA, imageB])

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Result", result)

#cv2.imwrite("matches.png", vis)
cv2.imwrite(output, result)
cv2.waitKey(0)
