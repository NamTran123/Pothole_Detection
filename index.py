import cv2
import numpy as np
import helpers as hp
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# for gaussian blur
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150


input_file = '/home/sink-all/Desktop/ML Source/DOAN/infonet_11.jpg'
input_file1 = '/home/sink-all/Desktop/ML Source/DOAN/index4.jpg'

img = [input_file,input_file1]

for  image  in img:
    image = mpimg.imread(input_file)
    
    image3 = hp.gaussian_blur(image, kernel_size)
    plt.title('Gausian Image'), plt.xticks([]), plt.yticks([])
    plt.imshow(image3, interpolation='nearest')
    plt.axis('off')
    plt.show()
    cv2.imwrite('gaussian_blur.jpg',image3)
    cv2.waitKey(0)

    image4 = hp.contrast_adjustments(image3)

    plt.title('contrast_adjustments'), plt.xticks([]), plt.yticks([])
    plt.imshow(image4, interpolation='nearest')
    plt.axis('off')
    plt.show()
    cv2.imwrite('contrast_adjustments.jpg',image4)
    cv2.waitKey(0)


    img_grey = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
    # Threshold

    th3 = hp.get_threshold(img_grey)

    # Erosion
    erosion = hp.get_erosion(th3)

    # Dilation 1

    dilate = hp.get_dilation(erosion)

    # Dilation 1
    dilate = hp.get_dilation(dilate)

    # canny
    canny = hp.get_canny(image, dilate, low_threshold, high_threshold)

    plt.show()
    cv2.waitKey(0)

    # find contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 500 and area < 10000:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)
            # cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
    cv2.imshow("output", image)
    cv2.imwrite('output.jpg',image)
    cv2.waitKey(0)
