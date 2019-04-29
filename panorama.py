# import the necessary packages
import numpy as np
import imutils
import cv2

import imgutils

def detectAndDescribe(image):
	# convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect and extract features from the image
	descriptor = cv2.xfeatures2d.SIFT_create()
	(kps, features) = descriptor.detectAndCompute(image, None)

	# convert the keypoints from KeyPoint objects to NumPy
	# arrays
	kps = np.float32([kp.pt for kp in kps])

	# return a tuple of keypoints and features
	return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
	ratio, reprojThresh):
	# compute the raw matches and initialize the list of actual
	# matches
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
	matches = []

	# loop over the raw matches
	for m in rawMatches:
		# ensure the distance is within a certain ratio of each
		# other (i.e. Lowe's ratio test)
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			matches.append((m[0].trainIdx, m[0].queryIdx))

	# computing a homography requires at least 4 matches
	if len(matches) > 4:
		# construct the two sets of points
		ptsA = np.float32([kpsA[i] for (_, i) in matches])
		ptsB = np.float32([kpsB[i] for (i, _) in matches])

		# compute the homography between the two sets of points
		(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
			reprojThresh)

		# return the matches along with the homograpy matrix
		# and status of each matched point
		return (matches, H, status)

	# otherwise, no homograpy could be computed
	return None


def stitch(images, ratio=0.75, reprojThresh=4.0):
	# unpack the images, then detect keypoints and extract
	# local invariant descriptors from them
	(imageB, imageA) = images
	imageB = imgutils.padImage(imageB, max(imageA.shape[0],imageA.shape[1]))
	imageA = imgutils.padImage(imageA, max(imageB.shape[0],imageB.shape[1]))
	
	(kpsA, featuresA) = detectAndDescribe(imageA)
	(kpsB, featuresB) = detectAndDescribe(imageB)

	# match features between the two images
	M = matchKeypoints(kpsA, kpsB,
		featuresA, featuresB, ratio, reprojThresh)

	# if the match is None, then there aren't enough matched
	# keypoints to create a panorama
	if M is None:
		return None

	# otherwise, apply a perspective warp to stitch the images
	# together
	(matches, H, status) = M
	result = cv2.warpPerspective(imageA, H,
		(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))	
	
	for y in range(0,imageB.shape[0]):
		for x in range(0, imageB.shape[1]):
			if list(imageB[y][x]) != [0,0,0]:
				result[y][x] = imageB[y][x]
	
	result = imgutils.trimImage(result)
	#result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

	# return the stitched image
	return result

























