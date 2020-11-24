#249109 Mateusz Drygiel AI and CV lab 3rd
import numpy as np
import cv2
import matplotlib.pyplot as plt

#TASK 1 AND 2

#defining the quantization of the image
def Quantizing_img(img, intense):

    adjust = 256 / intense
    adjust = int(adjust)
    img_quant = img.copy()
    rows, cols = img.shape[:2]
    contrast = img.std()
    for row in range(0,rows):
        for col in range(0,cols):
            index = img_quant[row,col] / adjust
            index_after_quant = int(index) * adjust
            img_quant[row, col] = int(index_after_quant)

    return img_quant

#defining the image stretching process
def Stretching_img(img):

   cv2.imshow('Before Stretching', img)

   img_stretch = cv2.resize(img, (700,700))
   cv2.imshow('After Stretching', img_stretch)
   return img_stretch

#histogram equalization of the image
def Histogram_eq(img):

    equl = cv2.equalizeHist(img)        #equalization histogram
    eqresults = np.hstack((img, equl))  #making it so that the images appear next to eachother
    cv2.imshow('Before and after Eq', eqresults)
    return eqresults

#defining the resizing of the image
def Resize_img(img):

    sorc = img.copy()
    scale_percent = 70  # the percentage level of the OG size
    wid = int(sorc.shape[1] * scale_percent / 100)   #the width of image
    hei = int(sorc.shape[0] * scale_percent / 100)   #the height of image
    dimens = (wid, hei)
    resized = cv2.resize(sorc, dimens, interpolation=cv2.INTER_AREA)
    return resized

#Reading the images, both low and high freq
low_freq_img = cv2.imread('AppleOld.png', 0)
high_freq_img = cv2.imread('db2.jpg', 0)
#Depending on which image we would like to use we can choose either low or hi frequency one
img = Resize_img(low_freq_img)
#img2 = Resize_img(high_freq_img)
#depending on the level of intensity we have the following images being passed along with intensity lvl
#first they are quantized
img_lv32 = Quantizing_img(img, 32)
im_hist = Histogram_eq(img)         #and then after quant one can ask the program to show the comparison before and after hist eq
img_lv64 = Quantizing_img(img, 64)
img_lv128 = Quantizing_img(img, 128)
#im_hist128 = Histogram_eq(img)
#then depending on the task, one can do Stretching:
img_lv32str = Stretching_img(img_lv32)
img_lv64str = Stretching_img(img_lv64)
img_lv128str = Stretching_img(img_lv128)


#calculating the histograms depending on the intensity level and if the image was stretched or not
higr = cv2.calcHist([img],[0],None,[256],[0,256])
higr_32 = cv2.calcHist([img_lv32],[0],None,[256],[0,256])
higr_64 = cv2.calcHist([img_lv64], [0], None, [256], [0, 256])
higr_128 = cv2.calcHist([img_lv128], [0], None, [256], [0, 256])
#for the stretched images its the same procedure as above just hcanging the 'img_lvlX' to 'img_lvlXstr'
higr_32str = cv2.calcHist([img_lv32str],[0],None,[256],[0,256])
#plotting each histogram
plt.plot(higr),plt.show()
plt.plot(higr_32str),plt.show()
#representing the images (in here its from the very first image before quantization and after with different intensity
#and also the ones with stretch applied to them
cv2.imshow('original',img)
cv2.imshow('quantized 128',img_lv128)
cv2.imshow('quantized 64',img_lv64)
cv2.imshow('stretched 64',img_lv64str)
cv2.imshow('quantized 32',img_lv32)
cv2.imshow('stretched 32', img_lv32str)

cv2.waitKey(0)

#TASK 4

#Lena DFT creation
DFT = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
DFT_shi = np.fft.fftshift(DFT)
mag_sp = 20 * np.log(cv2.magnitude(DFT_shi[:, :, 0], DFT_shi[:, :, 1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('The image')
plt.subplot(122),plt.imshow(mag_sp, cmap ='gray')
plt.title('The image spectrum')
plt.show()

#Creating inverse DFT
rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)
#Creating mask with square center 1 and rest all 0
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

#Lena before and after Fourier
fshift = DFT_shi * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Before Fourier')
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('After Fourier')
plt.show()
