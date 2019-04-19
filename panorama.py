# import the necessary packages
import numpy as np
import imutils
import cv2

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3(or_better=True)

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		imageB = self.padImage(imageB, imageB.shape[0])
		imageA = self.padImage(imageA, imageA.shape[0])
		
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)

		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
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
		
		cv2.imshow("imgAwarped",result)
		cv2.imshow("imga", imageA)
		cv2.imshow("imgb", imageB)
		cv2.waitKey();cv2.destroyAllWindows()		
		
		
		for y in range(0,imageB.shape[0]):
			for x in range(0, imageB.shape[1]):
				if list(imageB[y][x]) != [0,0,0]:
					result[y][x] = imageB[y][x]
		
		result = self.trimImage(result)
		#result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)

		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
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

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis
	
	'''
	Pad image with invisible pixels
	'''
	def padImage(self, img, amount):
		
		return cv2.copyMakeBorder(img, amount, amount, amount, amount, cv2.BORDER_CONSTANT, value=[0,0,0])
	
	'''
	Remove invisible pixels from borders around image
	'''
	def trimImage(self, img):
		leftBorder = 0
		rightBorder=img.shape[1]
		topBorder = 0
		bottomBorder=img.shape[0]
		
		# doing this in a bonehead dumb way because I cannot for the life of me
		# figure out how to do it intelligently with cv2 masks and such
		
		# go through each row/column and find the closest rows/columns that
		# are a full line of transparent pixels ([0,0,0]) and can be trimmed from border
		
		# Loop each column from left to right
		for x in range(0,img.shape[1]):
			allTrans = True
		
			for y in range(0,img.shape[0]):
				pix = img[y,x]
				
				if list(pix) != [0,0,0]:
					allTrans = False
					break
				
			if allTrans:
				leftBorder = x
			else:
				break
		
		# Loop each column from right to left
		for x in range(img.shape[1]-1, 0, -1):
			allTrans = True
		
			for y in range(0,img.shape[0]):
				pix = img[y,x]
				
				if list(pix) != [0,0,0]:
					allTrans = False
					break
				
			if allTrans:
				rightBorder = x + 1
			else:
				break
		
		# Loop each row from top to bottom
		for y in range(0,img.shape[0]):
			allTrans = True
		
			for x in range(0,img.shape[1]):
				pix = img[y,x]
				
				if list(pix) != [0,0,0]:
					allTrans = False
					break
				
			if allTrans:
				topBorder = y
			else:
				break
		
		# Loop each row from bottom to top
		for y in range(img.shape[0]-1, 0, -1):
			allTrans = True
		
			for x in range(0,img.shape[1]):
				pix = img[y,x]
				
				if list(pix) != [0,0,0]:
					allTrans = False
					break
				
			if allTrans:
				bottomBorder = y + 1
			else:
				break
	
		return img[topBorder:bottomBorder, leftBorder:rightBorder]









































