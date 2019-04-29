'''
Utility functions for modifying images
'''

import cv2

'''
Pad image with invisible pixels
'''
def padImage(img, amount):
	
	return cv2.copyMakeBorder(img, amount, amount, amount, amount, cv2.BORDER_CONSTANT, value=[0,0,0])

'''
Remove invisible pixels from borders around image.

NOTE: Done horribly, horribly inefficiently. Had very poor luck trying to get the
optimal approach for this to work, so this is basically a brute force method.
'''
def trimImage(img):
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









































