import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


image = cv2.imread("./img.jpg",0)


image_thr = cv2.adaptiveThreshold(image, 255, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 51, 0)

# Apply morphological opening with vertical line kernel
kernel = np.ones((image.shape[0], 1), dtype=np.uint8) * 255
image_mop = cv2.morphologyEx(image_thr, cv2.MORPH_OPEN, kernel)

# Canny edge detection
image_canny = cv2.Canny(image_mop, 1, 3)

# Get pixel values from the input image (force RGB/BGR on given input) within stripes
image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
pixels = image_bgr[image_mop > 0, :]
print(pixels)



cv2.imshow("ori image",image_thr )


cv2.waitKey(0)