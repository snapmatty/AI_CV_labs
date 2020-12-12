import cv2
import numpy as np
import random

#reading the images
#image = cv2.imread('lowfreq.png',0)
image = cv2.imread('highfreq.png',0)

cv2.imshow('Original image TASK 1', image)

#########################TASK 1#####################################################

#create a matrix of zeroes from image size, then fill it with gaussian noise
rows, cols = image.shape[:2]
gaus_noise = np.zeros((rows, cols), dtype=np.uint8)
mean = 128
standard_deviation = 156
# standard_deviation = 56
cv2.randn(gaus_noise, mean, standard_deviation)

#add noise to OG image
gaus_noise = (gaus_noise * 0.5).astype(np.uint8)
noise_img = cv2.add(image, gaus_noise)

cv2.imshow('Img with gaussian noise', noise_img)

#apply gaussian blur on OG image
filt_img = cv2.GaussianBlur(noise_img, (3, 3), cv2.BORDER_DEFAULT)
cv2.imshow("Img with gaussian noise - filtered", filt_img)

#sharpen image (also sharpens the noise)
sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharp_img = cv2.filter2D(image, -1, sharp)
cv2.imshow("Img with gaussian noise - sharpened", sharp_img)
cv2.waitKey()

#########################TASK 2#######################################

#reading the images
#image = cv2.imread('lowfreq.png',0)
image = cv2.imread('highfreq.png',0)

cv2.imshow('Original image TASK 2', image)

#add salt and pepper noise to og img
output = np.zeros(image.shape,np.uint8)
#probability of noise
prob = 0.02
thres = 1 - prob

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        rdn = random.random()
        if rdn < prob:
            output[i][j] = 0
        elif rdn > thres:
            output[i][j] = 255
        else:
            output[i][j] = image[i][j]


cv2.imshow('salt and pepper noise',output)

#apply  blur on OG image
filtered_image = cv2.medianBlur(output,3)
cv2.imshow("salt and pepper noise - filtered", filtered_image)

#sharpen the image and noise
sharp_filt = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharp_img = cv2.filter2D(filtered_image, -1, sharp_filt)
cv2.imshow("salt and pepper noise - sharpened", sharp_img)
cv2.waitKey()
cv2.destroyAllWindows()


#####################TASK 3##########################################

cv2.imshow('OG image TASK 3', image)
dx = 1
dy = 1
dst = cv2.Sobel(image,-1,dx,dy)

cv2.imshow('Img with sobel', dst)
cv2.waitKey()
