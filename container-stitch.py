# Stitch images quickly using container idea - one large image over which transformed images are overlaid
# Problems: makes batch approaches more difficult
import cv2
import numpy as np
import imutils
from glob import glob
from sys import exit

MIN_MATCH_COUNT = 4

def addImage(image, container):
	# threshold both images, find non-overlapping sections, add to container
	gimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gcont = cv2.cvtColor(container,cv2.COLOR_BGR2GRAY)
	ret,threshimg = cv2.threshold(gimg,10,255,cv2.THRESH_BINARY)
	ret,threshcont = cv2.threshold(gcont,10,255,cv2.THRESH_BINARY)
	intersect = cv2.bitwise_and(threshimg, threshcont) # find intersection between container and new image
	mask = cv2.subtract(threshimg,intersect) # subtract the intersection, leaving just the new part to union
	kernel = np.ones((3,3),'uint8') # for dilation below
	mask = cv2.dilate(mask,kernel,iterations=1) # make the mask slightly larger so we don't get blank lines on the edges
	maskedImage = cv2.bitwise_and(image, image, mask=mask) # apply mask
	con = cv2.add(container, maskedImage) # add the new pixels
	return con

# can change extension
imagenames = glob('*.jpg')
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

if len(imagenames) < 2:
    print('not enough images in current directory!')
    exit()

print(imagenames)
images = [cv2.imread(img) for img in imagenames]
images = [imutils.resize(img, width=800) for img in images]

width, height, _ = images[0].shape
# used to determine position on the final canvas
windowShift = [width*2, height*2]
# used to determine dimensions of final canvas
windowSize = (width * 5, height * 5)

# create base canvas image
base = np.zeros((windowSize[1],windowSize[0],3), np.uint8)
img1 = np.array(base)
# add first image image to the canvas
base[windowShift[1]:images[0].shape[0]+windowShift[1], windowShift[0]:images[0].shape[1]+windowShift[0]] = images[0] 
img1 = addImage(base, img1)

while len(images) > 0:
    img2 = images.pop(0)

    #greyscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Detect keypoints
    kpts1, descs1 = orb.detectAndCompute(gray1,None)
    kpts2, descs2 = orb.detectAndCompute(gray2,None)

    #deal with matches
    matches = matcher.match(descs1, descs2)
    matches = sorted(matches, key = lambda x:x.distance)
    good = [m for m in matches if m.distance < 100]
    canvas = img2.copy()

    # Try to find a good homography matrix
    if len(good)>MIN_MATCH_COUNT:
        # swap dst and src to calculate correct homography for canvas approach
        dst_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # find homography matrix in cv2.RANSAC using good match points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) 
    else:
        print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))
        # put image back in queue to re-evaluate after more images are added to the mosaic
        images.append(nextimg)
        continue
    # use calculated homography to perform transforms and stitching
    res = cv2.warpPerspective(img2, M, windowSize)
    img1 = addImage(res, img1)
    print('processed image')


## save and display
cv2.imwrite("found.png", img1)
cv2.imshow("found", img1)
cv2.waitKey()
cv2.destroyAllWindows()
