import cv2
import numpy as np

#images
img = cv2.imread('lowfreq.png', 0)
#img = cv2.imread('highfreq.png',0)

cv2.imshow('OG image TASK 4', img)

#matrix of zeros based on image size;filling it with gaussian noise
rows, cols = img.shape[:2]
gaus_noise = np.zeros((rows, cols), dtype=np.uint8)
mean = 128
stand_dev = 156
# standard_deviation = 56
cv2.randn(gaus_noise, mean, stand_dev)

#adding noise to the original image
gaus_noise = (gaus_noise * 0.5).astype(np.uint8)
noisy_img = cv2.add(img, gaus_noise)

cv2.imshow('Img with gaussian', noisy_img)

#apply sobel filter
dx = 1
dy = 1

dst = cv2.Sobel(img, -1, dx, dy)

cv2.imshow('Img with sobel', dst)
cv2.waitKey()
