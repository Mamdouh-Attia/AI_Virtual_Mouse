

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
<<<<<<< HEAD
from skimage.color import rgb2gray,rgb2hsv
=======
from skimage.color import rgb2gray, rgb2hsv

>>>>>>> e697d073a442b7a467be28ceee55456363966960

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
<<<<<<< HEAD
from skimage.measure import label,find_contours
from skimage.color import label2rgb
from skimage.morphology import skeletonize,binary_dilation,binary_erosion



# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
=======
from skimage.measure import label, find_contours
from skimage.color import label2rgb

# Edges
from skimage.filters import sobel_h, sobel, sobel_v, roberts, prewitt

# Show the figures / plots inside the notebook


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
>>>>>>> e697d073a442b7a467be28ceee55456363966960
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
<<<<<<< HEAD
    plt.show() 
=======
    plt.show()
>>>>>>> e697d073a442b7a467be28ceee55456363966960


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
<<<<<<< HEAD
    
=======

>>>>>>> e697d073a442b7a467be28ceee55456363966960
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
