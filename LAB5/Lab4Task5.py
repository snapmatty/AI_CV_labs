import cv2
import numpy as np

#loading images
image = cv2.imread('highfreq.png',0)

cv2.imshow('OG img', image)

dist = cv2.Laplacian(image,-1,1)
cv2.imshow("Img filtered;built in laplace filter", dist)

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

filt_img = cv2.filter2D(image, -1, kernel)
cv2.imshow("Img filtered;custom laplace filter", filt_img)
cv2.waitKey()

#custom filter has worse quality, all noise and edges are very bold. On the other hand in-built function
#output looks very smooth and all edges are thin, we see only contours