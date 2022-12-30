from commonfunctions import *
import numpy as np
from skimage.morphology import binary_opening, binary_dilation, binary_erosion, skeletonize
import cv2 as cv
import os
import random
import mediapipe as mp

# get threshold value


def calcThreshold(hist, accHist, iFrom, iTo):
    iFrom, iTo = int(iFrom), int(iTo)
    numOfPixels = accHist[iTo] - (accHist[iFrom - 1] if iFrom > 0 else 0)
    mean = np.sum(
        hist[iFrom:iTo+1] * np.arange(iFrom, iTo+1)
    ) / numOfPixels
    return round(mean)


def avgThreshold(hist, accHist, Tinit):
    mean1 = calcThreshold(hist, accHist, 0, Tinit - 1)
    mean2 = calcThreshold(hist, accHist, Tinit, hist.shape[0] - 1)
    newThreshold = round((mean1 + mean2) / 2)
    return newThreshold if Tinit == newThreshold else avgThreshold(hist, accHist, newThreshold)


def getGlobalThreshold(img):
    img2 = np.copy(img)
    hist = np.histogram(img2, bins=np.arange(256))[0]
    accHist = np.cumsum(hist)
    Tinit = calcThreshold(hist, accHist, 0, hist.shape[0] - 1)
    return avgThreshold(hist, accHist, Tinit)  # get threshold


def get_image_with_skin_color(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) * 0.564 + 128
    Cr = (R - Y) * 0.713 + 128
    outImg = (Cb >= 77) * (Cb <= 127) * (Cr >= 133) * (Cr <= 180)
    return outImg


def enhance_image(img):
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
    enhanced_image = np.array(img).astype('uint8')
    enhanced_image = cv.dilate(
        enhanced_image, se, iterations=15
    )
    enhanced_image = cv.erode(
        enhanced_image, se, iterations=15
    )
    return enhanced_image


def captureHand(binaryImg, mainImg):
    contours = find_contours(binaryImg,  fully_connected='high')
    for contour in contours:
        Ymin = int(np.min(contour[:, 1]))
        Ymax = int(np.max(contour[:, 1]))
        Xmin = int(np.min(contour[:, 0]))
        Xmax = int(np.max(contour[:, 0]))
        if Xmax - Xmin >= 80 and Ymax - Ymin >= 80:
            detectedImage = np.array(mainImg[Xmin:Xmax, Ymin:Ymax])
            mp_hands = mp.solutions.hands
            hand = mp_hands.Hands()
            result = hand.process(detectedImage)
            if result.multi_hand_landmarks:
                return np.array(binaryImg[Xmin:Xmax, Ymin:Ymax])
    return None


def detectHand(img):
    outImg = get_image_with_skin_color(img)
    enhanced_image = enhance_image(outImg)
    hand = captureHand(enhanced_image, img)
    return hand

img = io.imread('./images/image8.jpeg')
show_images([detectHand(img)])

