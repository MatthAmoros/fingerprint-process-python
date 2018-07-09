import cv2
import numpy as np
import math
import json


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
	_, thrsh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return thrsh

def filter_out_small_elements(img, size=1):
	"""
	Filter out small elements
	@param img = Thinned image
	@param size = Element size
	@return filtered image
	"""
	_, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	""" Get contours surfaces """
	areas = [cv2.contourArea(c) for c in contours]

	for cnt in range(len(areas)):
		if(areas[cnt] < size):
			cv2.drawContours(img, [contours[cnt]], 0, (0,255,0), -1)

	return img

def get_external_contour(img, margin=1):
	"""
	Get external object borders
	@param img = Thinned image
	@param margin = Applied margin
	@return bounding box
	"""
	""" Finding bounding box """
	""" Blur to smooth contours """
	blurred = cv2.blur(img,(20,20))
	_, contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	""" Get biggest contours """
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]

	""" x, y, w, h of bounding rectangle """
	bounding_box = cv2.minAreaRect(cnt)
	bounding_box = cv2.boxPoints(bounding_box)
	bounding_box = np.int0(bounding_box)

	""" Apply margin """
	bounding_box = bounding_box * margin
	bounding_box = np.int0(bounding_box)


	out = img.copy()
	cv2.drawContours(out, [bounding_box], 0, (255,0,255), 2)

	cv2.imwrite('./fingerprint_extract/0-5_bounds.png', out)

	return bounding_box

def is_point_in_rectangle(point, rectangle):
	"""
	Return True if point is contained in rectangle
	@param point = (x,y)
	@param rectangle = aray[4][2]
	@return boolean
	"""
	x = point[0]
	y = point[1]

	rect_1 = rectangle[1]
	rect_2 = rectangle[3]

	return (x > rect_1[0] and x < rect_2[0] and y > rect_1[1] and y < rect_2[1])


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

	strcuture_cross_02 = np.array([[1, 1, 1],\
									  [-1, -1, 1],\
									  [1, 1, 1]])

	""" Filter out small surface element to avoid nosie """
	img = filter_out_small_elements(img)

	""" Ouput binary (0 - 1) image """
	_, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

	linear_weight_01 = cv2.filter2D(thresh, -1, structure_line_end_01)
	linear_weight_02 = cv2.filter2D(thresh, -1, structure_line_end_02)
	cross_weight = cv2.filter2D(thresh, -1, strcuture_cross)

	""" Get external contour to verify found minutiae """
	external_contour = get_external_contour(img)

	center_x = math.ceil(external_contour[3][0] / 2)
	center_y = math.ceil(external_contour[3][1] / 2)
	radius = math.ceil((external_contour[3][0] - external_contour[1][0]) / 2)

	print("Box center : " + str((center_x, center_y)))
	print("Peripherical radius : " + str(radius))
	print("Bounding box : " + str(external_contour))

	""" Intialize CN map and histogram """
	cn_map = np.zeros(thresh.shape, np.uint8)
	cn_hist = np.zeros([5])

	""" Implicit cast to float to avoid buffer overflow """
	thresh = 1.0 * thresh

	""" Using cross number to detect minutiae """
	for r in range(cn_map.shape[0]):
		for c in range(cn_map.shape[1]):
			if (r > 1 and c > 1) and (r < 299 and c < 299):
				"""
				From NIST Special Publication 500-245
				Cross number algorithm
				"""

				p = thresh[r][c]

				""" To avoid noise, check that current point is included in fingerprint contours """
				if is_point_in_rectangle((r, c), external_contour) == False:
					continue

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

				sum += np.fabs(p1 - p2)
				sum += np.fabs(p2 - p3)
				sum += np.fabs(p3 - p4)
				sum += np.fabs(p4 - p5)
				sum += np.fabs(p5 - p6)
				sum += np.fabs(p6 - p7)
				sum += np.fabs(p7 - p8)
				sum += np.fabs(p8 - p9)
				cn = 0.5 * sum

				""" Cast back to int, to use it as an array index """
				cn = int(cn)

				"""
				0 == Nothing
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
					""" Check for peripherical points """
					""" Used to avoid storing ridge ending on fingerprint endings """
					#DEBUG Disabled
					is_external = False#(calculate_eucl_dist((r, c), (center_x, center_y)) > radius)
					validated = validated and not is_external
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

def draw_minitiae(img, minutiae_pos):
	for i in range(len(minutiae_pos)):
		#NOTE : Red filled circle of 2px at minutiae center
		cv2.circle(img, (minutiae_pos[i][0], minutiae_pos[i][1]), 2, minutiae_pos[i][2], -1)

	return img

def calculate_eucl_dist(p1, p2):
	x1 = p1[0]
	y1 = p1[1]
	x2 = p2[0]
	y2 = p2[1]

	eucl_dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
	eucl_dist = math.trunc(eucl_dist)

	return eucl_dist

def build_template(minutiae_pos):
	"""
	Compute distance between each minutiae and return it as an array
	"""
	template = np.zeros(len(minutiae_pos), dtype='int, int')

	""" Compute euclidean distance for each minutiae """
	for i in range(len(minutiae_pos) - 1):
		eucl_dist = calculate_eucl_dist(minutiae_pos[i], minutiae_pos[i+1])
		template[i] = (eucl_dist, minutiae_pos[i][3])

	""" Compute last one """
	eucl_dist = calculate_eucl_dist(minutiae_pos[len(minutiae_pos) - 1], minutiae_pos[0])
	template[len(minutiae_pos) - 1] = (eucl_dist, minutiae_pos[len(minutiae_pos) - 1][3])

	""" Sort """
	template.sort()

	""" Distinct """
	template = np.unique(template)

	return template

def save_to_json_file(template_name, template):
	"""
	Save template to JSON file for later use
	"""
	with open('./templates/' + template_name + '.tmplt', 'w') as outfile:
		""" Numpy arrays need to be converted to list to be dumped to JSON """
		listify = template.tolist()
		json.dump(listify, outfile)

def build_template_from_image(image_path, subject_name):
	"""
	Build a template of minutiae from specified grayscale image
	@param image_path = Path to fingerprint image file
	@param subject_name = Subject name, ussed to name template
	"""
	subject = subject_name
	img = cv2.imread(image_path, 0)

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
	features_pos = []
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
				type = cn_map[r][c]
				if type == 1:
					circle_color = (0,255,0)
					detected_feature = True ##DEBUG : Disabled
				if type == 2:
					circle_color = (255,0,0)
					detected_feature = False ##DEBUG : Disabled
				if type == 3:
					circle_color = (0,0,255)
					detected_feature = True
				elif type == 4:
					circle_color = (255,255,0)
					detected_feature = True

				if detected_feature:
					features_pos.append((minutiae_x, minutiae_y, circle_color, type))

	detection = draw_minitiae(colorized_output, features_pos)

	cv2.imwrite('./fingerprint_extract/3_detected_minutiae.png', detection)

	template = build_template(features_pos)

	print("Template : ")
	print(template)

	save_to_json_file(subject, template)

def compare_templates(template_1, tamplate_2):
	"""
	Compare two templates (Numpy arrays) and returns matching score
	@param template_1 = Minutiae template
	@param template_2 = Minutiae template
	@return Matching score ( 0 - 100% ), Matching minutiae
	"""

	smallest_template = ()
	score = 0

	if(len(template_1) <= len(tamplate_2)):
		smallest_template = template_1
	else:
		smallest_template = tamplate_2

	intersect_result = np.zeros(len(smallest_template), dtype='int, int')

	for mn in template_1:
		for mn_dest in tamplate_2:
			if (mn == mn_dest).all():
				intersect_result[score] = (mn[0], mn[1])
				score += 1

	score = score / len(template_1) * 100

	print("Found : " + str(score) + "% matching")
	return score, intersect_result

def load_template_from_file(file_path):
	with open(file_path) as f:
		template = np.array(json.load(f))

	return template

if __name__ == "__main__":
	"""
	Test
	"""
	print("Loading previously acquiered templates")
	temp1 = load_template_from_file('./templates/001_01.tmplt')
	temp2 = load_template_from_file('./templates/001_01 (copy).tmplt')


	score, intersect = compare_templates(temp1, temp2)

	print("Compute new template from image")
	build_template_from_image('./samples/001_01_0063.png', '001_01_02')
	temp3 = load_template_from_file('./templates/001_01_02.tmplt')

	print(str(type(intersect)))

	score, intersect = compare_templates(intersect, temp3)
	print("Matching score " + str(score))

	print(intersect)
