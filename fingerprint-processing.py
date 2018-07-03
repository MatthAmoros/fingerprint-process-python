import cv2
import numpy as np
import math

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

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

	iters = 0
	while(True):
		cv2.erode(thresh, kernel, eroded)
		cv2.dilate(eroded, kernel, temp)
		cv2.subtract(thresh, temp, temp)
		cv2.bitwise_or(skeleton, temp, skeleton)
		thresh, eroded = eroded, thresh # Swap instead of copy

		iters += 1
		if cv2.countNonZero(thresh) == 0:
			return skeleton

def crop_image_square(img, crop_size=3):
	"""
	Crop image into smaller windows
	@param crop_size = window size
	@return an array of cropped windows
	"""
	crop_collection = []

	for r in range(0, img.shape[0]):
		for c in range(0, img.shape[1]):
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
	img = cv2.resize(img, (300, 300))

	## Threshold
	_, thrsh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return thrsh

def get_crossnumber_map(img):
	"""
	Extract pattern from binarized image
	@param img : binarized image
	@return matrix of weight
	"""
	image_x = img.shape[0]
	image_y = img.shape[1]

	print("Input image : " + str(image_x) + "x" + str(image_y))

	""" Structure filters """
	structure_line_end_01 = np.array([[-1, -1,-1],\
								 	  [-1,1,1],\
								 	  [-1,-1,-1]])

	structure_line_end_02 = np.array([[-1, -1,-1],\
									  [1,1, -1],\
									  [-1,-1,-1]])

	strcuture_cross = np.array([[-1, 1,-1],\
							  [1,1, 1],\
							  [-1,1,-1]])

	""" Ouput binary (0 - 1) image """
	_, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

	linear_weight_01 = cv2.filter2D(thresh, -1, structure_line_end_01)
	linear_weight_02 = cv2.filter2D(thresh, -1, structure_line_end_02)
	cross_weight = cv2.filter2D(thresh, -1, strcuture_cross)

	""" Intialize CN map and histogram """
	cn_map = np.zeros(thresh.shape, np.uint8)
	cn_hist = np.zeros([5])

	""" Using cross number to detect minutiae """
	for r in range(cn_map.shape[0]):
		for c in range(cn_map.shape[1]):
			if (r > 1 and c > 1) and (r < 299 and c < 299):
				"""
				From NIST Special Publication 500-245
				"""
				p = thresh[r][c]

				p1 = thresh[r][c+1]
				p2 = thresh[r-1][c+1]
				p3 = thresh[r-1][c]

				p4 = thresh[r-1][c-1]
				p5 = thresh[r][c-1]
				p6 = thresh[r+1][c-1]

				p7 = thresh[r+1][c]
				p8 = thresh[r+1][c+1]
				p9 = p1

				sum = 0

				sum += abs(p1 - p2)
				sum += abs(p2 - p3)
				sum += abs(p3 - p4)
				sum += abs(p4 - p5)
				sum += abs(p5 - p6)
				sum += abs(p6 - p7)
				sum += abs(p7 - p8)
				sum += abs(p8 - p9)
				cn = 0.5 * sum

				##NOTE : OUTPUT is > 128, why ?
				cn = math.ceil(cn / 128.0)

				"""
				1 == Ridge ending point
				2 == Continuing ridge point
				3 == bifurcation point
				4 == crossing point
				"""
				""" Using structure filter to validate minutiae """
				validated = False
				sum_linear_weight = linear_weight_01[r][c] + linear_weight_02[r][c]

				if cn == 1:
					validated = sum_linear_weight == 1
				elif cn == 2:
					validated = sum_linear_weight == 3
				elif cn == 3:
					validated = cross_weight[r][c] >= 3
				elif cn == 4:
					validated = True

				if validated:
					cn_hist[cn] += 1
					cn_map[r][c] = cn

			else:
				cn_map[r][c] = 0


	print("Cross number hist : " + str(cn_hist))
	return cn_hist, cn_map

if __name__ == "__main__":
	img = cv2.imread('./samples/001_01_0055.png', 0)

	colorized_output = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	colorized_output = cv2.resize(colorized_output, (300, 300))

	binarized = pre_process(img)

	cv2.imwrite('./fingerprint_extract/0_pre_proc.png', binarized)

	thinned = cv2.ximgproc.thinning(binarized)

	cv2.imwrite('./fingerprint_extract/1_thinned.png', thinned)

	cn_hist, cn_map = get_crossnumber_map(thinned)

	cv2.imwrite('./fingerprint_extract/2_detections.png', cn_map)

	circle_color = (0,0,0)
	previous_feature_pos = (0,0)
	first_feature_pos = (0,0)
	for r in range(cn_map.shape[0]):
		for c in range(cn_map.shape[1]):
			minutiae_x = c
			minutiae_y = r
			circle_color = (0,0,0)
			detected_feature = False
			"""
			According to crossing number algorithm :

			1 == Ridge ending point
			2 == Continuing ridge point
			3 == bifurcation point
			4 == crossing point
			"""

			if cn_map[r][c] > 0:
				if cn_map[r][c] == 1:
					circle_color = (0,255,0)
					detected_feature = False ##DEBUG : Disabled
				if cn_map[r][c] == 2:
					circle_color = (255,0,0)
					detected_feature = False ##DEBUG : Disabled
				if cn_map[r][c] == 3:
					circle_color = (0,0,255)
					detected_feature = True
				elif cn_map[r][c] == 4:
					circle_color = (255,255,0)
					detected_feature = True

				if detected_feature:
					if previous_feature_pos[0] > 0:
						cv2.line(colorized_output, previous_feature_pos, (minutiae_x, minutiae_y), [255,0,0], 1)
					else:
						first_feature_pos = (minutiae_x, minutiae_y)
					cv2.circle(colorized_output, (minutiae_x, minutiae_y), 2, circle_color, -1)
					previous_feature_pos = (minutiae_x, minutiae_y)

	cv2.line(colorized_output, previous_feature_pos, first_feature_pos, [255,0,0], 1)

	cv2.imwrite('./fingerprint_extract/3_detected_minutiae.png', colorized_output)
