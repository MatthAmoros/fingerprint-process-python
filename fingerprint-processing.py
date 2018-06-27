import cv2
import numpy as np

def extract_skeleton(img):
	skeleton = np.zeros(img.shape, np.uint8)
	eroded = np.zeros(img.shape, np.uint8)
	temp = np.zeros(img.shape, np.uint8)

	thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

	iters = 0
	while(True):
		cv2.erode(thresh, kernel, eroded)
		cv2.dilate(eroded, kernel, temp)
		cv2.subtract(thresh, temp, temp)
		cv2.bitwise_or(skeleton, temp, skeleton)
		thresh, eroded = eroded, thresh # Swap instead of copy

		iters += 1
		if cv2.countNonZero(thresh) == 0:
			return (skeleton,iters)

def get_countour_orientation(img):
	im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	"""
	for i in range(len(contours)):
		cnt = contours[4]
		cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
	"""
	cv2.drawContours(img, contours, -1, (0,255,0), 3)
	return im2

def pre_process(img):
	"""
	Resize and binarize image
	@param : Grayscale image inferior to 300 x 300
	@return : 300x300 binarized image
	"""
	## Resize to squared image
	img = cv2.resize(img, (300, 300))
	## Compute binary image with an adaptative threshold
	thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

	return thresh

if __name__ == "__main__":

	img = cv2.imread('./samples/01.png', 0)
	img = pre_process(img)
	cv2.imwrite('./fingerprint_extract/0_pre_proc.png', img)

	skelet, iter = extract_skeleton(img)
	skelet = cv2.bitwise_not(skelet)

	cv2.imwrite('./1_skelet.png', skelet)
