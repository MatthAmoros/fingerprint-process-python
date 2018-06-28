import cv2
import numpy as np

def extract_skeleton(img):
	"""
	Extract skeleton image (repeat opening and substract)
	@param img = binarized image
	@return : skeleton image and iterations count
	"""
	skeleton = np.zeros(img.shape, np.uint8)
	eroded = np.zeros(img.shape, np.uint8)
	temp = np.zeros(img.shape, np.uint8)
	thresh = img.copy()

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

def get_contours_orientation(img):
	"""
	Get contours orientation
	"""
	im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	"""
	for i in range(len(contours)):
		cnt = contours[4]
		cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
	"""
	cv2.drawContours(img, contours, -1, (0,255,0), 3)
	return im2

def crop_image_square(img, crop_size=5):
	"""
	Crop image into smaller windows
	@param crop_size = window size
	@return an array of cropped windows
	"""
	crop_collection = []

	for r in range(0, img.shape[0] - crop_size, crop_size):
		for c in range(0, img.shape[1] - crop_size, crop_size):
			window = img[r:r+crop_size, c:c+crop_size]
			crop_collection.append(window)

	return crop_collection

def pre_process(img):
	"""
	Resize and binarize image
	@param img = Grayscale image inferior to 300 x 300
	@return : 300x300 binarized image
	"""
	## Resize to squared image
	img = cv2.resize(img, (400, 400))
	## Compute binary image with an adaptative threshold
	thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

	return thresh

if __name__ == "__main__":
	img = cv2.imread('./samples/001_01_0055.png', 0)
	binarized = pre_process(img)

	cropped_images = crop_image_square(img)

	sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)

	cv2.imwrite('./fingerprint_extract/0_pre_proc.png', binarized)

	cv2.imwrite('./fingerprint_extract/2_sobely.png', sobely)
