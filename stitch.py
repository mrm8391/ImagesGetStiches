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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first image")
ap.add_argument("-s", "--second", required=True,
	help="second image")
ap.add_argument("-o", "--output", required=True,
	help="output image")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
output = args["output"]

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
