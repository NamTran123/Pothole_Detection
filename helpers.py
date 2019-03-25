
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height



# Helper functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def  contrast_adjustments(img):
    image = img
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.1 # Simple contrast control
    beta = 20    # Simple brightness control
    
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_image, cmap='gray')
    plt.title('New Image '), plt.xticks([]), plt.yticks([])
    cv2.imwrite('contrast_adjustments1.jpg',new_image)
    # Wait until user press some key
    return new_image
def get_threshold(img_grey):
    ret3,th3 = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.title('threshold'), plt.xticks([]), plt.yticks([])
    plt.imshow(th3, interpolation='nearest')
    plt.axis('off')
    plt.show()
    cv2.imwrite('threshold.jpg',th3)
    cv2.waitKey(0)
    return th3

def get_erosion(th3):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(th3,kernel,iterations = 1)
    plt.title('erosion'), plt.xticks([]), plt.yticks([])
    plt.imshow(erosion, interpolation='nearest')
    plt.axis('off')
    plt.show()
    cv2.imwrite('erosion.jpg',erosion )
    cv2.waitKey(0)
    return erosion

def get_dilation(erosion):
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    plt.title('dilation'), plt.xticks([]), plt.yticks([])
    plt.imshow(dilation, interpolation='nearest')
    plt.axis('off')
    plt.show()
    cv2.imwrite('dilation.jpg',dilation )
    cv2.waitKey(0)
    return dilation

def get_canny(image,dilate, low_threshold, high_threshold):
    """Applies the Canny transform"""
    edges = cv2.Canny(dilate, low_threshold, high_threshold)
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    cv2.imwrite('edges.jpg',edges )
    return edges