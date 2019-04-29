
import cv2
import numpy as np


stitchedImg = cv2.imread('stitch4.png', cv2.IMREAD_UNCHANGED)

gauss = cv2.GaussianBlur(stitchedImg, (9, 9), 0)

median = cv2.medianBlur(stitchedImg, 9)

cv2.imshow('og', stitchedImg)

cv2.imshow('gauss', gauss)

cv2.imshow('median', median)

cv2.waitKey(0)



